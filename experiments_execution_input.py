from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from model_with_sae import ModelWithSAEModule
from prompts.experiments_execution_prompt import (
    build_input_activation_context,
    build_input_boundary_context,
)


def _extract_designed_sentences(experiment_item: Dict[str, Any]) -> List[str]:
    raw = experiment_item.get("designed_sentences")
    if not isinstance(raw, list):
        raise ValueError("Each input-side experiment item must contain a list field 'designed_sentences'.")
    sentences: List[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            sentences.append(item.strip())
    return sentences


def _extract_boundary_sentences(experiment_item: Dict[str, Any]) -> List[str]:
    raw = experiment_item.get("boundary_sentences")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("Field 'boundary_sentences' must be a list when provided.")
    sentences: List[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            sentences.append(item.strip())
    return sentences


def _extract_max_token(trace: Dict[str, Any]) -> str:
    tokens = trace.get("tokens")
    max_token_index = trace.get("max_token_index")
    if not isinstance(tokens, list) or not isinstance(max_token_index, int):
        return ""
    if max_token_index < 0 or max_token_index >= len(tokens):
        return ""
    token = tokens[max_token_index]
    return token if isinstance(token, str) else str(token)


def _tokenize_sentences(
    *,
    module: ModelWithSAEModule,
    sentences: Sequence[str],
) -> tuple[Any, Any]:
    if module.tokenizer is not None:
        enc = module.tokenizer(
            list(sentences),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        return input_ids, attention_mask
    if hasattr(module.model, "to_tokens"):
        input_ids = module.model.to_tokens(list(sentences))
        attention_mask = None
        return input_ids, attention_mask
    raise RuntimeError("Unable to tokenize sentences for activation analysis.")


def _run_sentence_batch(
    *,
    module: ModelWithSAEModule,
    sentences: Sequence[str],
    non_zero_threshold: float,
    max_activation_scale: float,
) -> Dict[str, Any]:
    sentence_results: List[Dict[str, Any]] = []
    non_zero_count = 0
    activation_sum = 0.0
    activation_max = 0.0

    if not sentences:
        return {
            "sentence_results": [],
            "non_zero_count": 0,
            "total_sentences": 0,
            "score_non_zero_rate": 0.0,
            "mean_summary_activation": 0.0,
            "max_summary_activation": 0.0,
            "runtime_batch": None,
        }

    input_ids, attention_mask = _tokenize_sentences(module=module, sentences=sentences)
    batch_analysis = module.analyze_feature_batch_from_tensors(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_activation_scale=float(max_activation_scale),
    )
    traces = batch_analysis["traces"]

    for sentence_index, sentence in enumerate(sentences, start=1):
        trace = traces[sentence_index - 1]
        # Use no-bos variant to avoid BOS token dominating max activation
        activation_value = float(trace.get("summary_activation_no_bos", trace.get("summary_activation", 0.0)) or 0.0)
        activation_sum += activation_value
        activation_max = max(activation_max, activation_value)
        is_non_zero = activation_value > non_zero_threshold
        if is_non_zero:
            non_zero_count += 1

        sentence_results.append(
            {
                "sentence_index": sentence_index,
                "sentence": sentence,
                "summary_activation": activation_value,
                "summary_activation_mean": float(trace.get("summary_activation_mean", 0.0) or 0.0),
                "summary_activation_sum": float(trace.get("summary_activation_sum", 0.0) or 0.0),
                "max_token_index": int(trace.get("max_token_index", 0) or 0),
                "max_token": _extract_max_token(trace),
                "intervention_token_index": int(trace.get("intervention_token_index", 0) or 0),
                "intervention_token": str(trace.get("intervention_token", "") or ""),
                "intervention_base_activation": float(trace.get("intervention_base_activation", 0.0) or 0.0),
                "intervention_target_activation": float(trace.get("intervention_target_activation", 0.0) or 0.0),
                "is_non_zero": is_non_zero,
            }
        )

    total_sentences = len(sentences)
    non_zero_rate = (non_zero_count / total_sentences) if total_sentences > 0 else 0.0
    mean_activation = (activation_sum / total_sentences) if total_sentences > 0 else 0.0
    return {
        "sentence_results": sentence_results,
        "non_zero_count": non_zero_count,
        "total_sentences": total_sentences,
        "score_non_zero_rate": non_zero_rate,
        "mean_summary_activation": mean_activation,
        "max_summary_activation": activation_max,
        "runtime_batch": {
            "prompts": list(sentences),
            "input_ids": batch_analysis["input_ids"],
            "attention_mask": batch_analysis["attention_mask"],
            "clean_logits": batch_analysis["clean_logits"],
            "selected_token_positions": batch_analysis["selected_positions"],
            "selected_token_base_values": batch_analysis["selected_base_values"],
            "selected_token_target_values": batch_analysis["selected_target_values"],
        },
    }


def execute_input_side_experiments(
    *,
    input_side_experiments: Sequence[Dict[str, Any]],
    module: ModelWithSAEModule,
    non_zero_threshold: float = 0.0,
    max_activation_scale: float = 2.0,
) -> Dict[str, Any]:
    hypothesis_results: List[Dict[str, Any]] = []
    runtime_batches: List[Optional[Dict[str, Any]]] = []

    for hypothesis_index, item in enumerate(input_side_experiments, start=1):
        hypothesis_text = str(item.get("hypothesis", "")).strip()
        sentences = _extract_designed_sentences(item)
        boundary_sentences = _extract_boundary_sentences(item)

        # Run with threshold=0 first to get raw activations
        activation_metrics = _run_sentence_batch(
            module=module,
            sentences=sentences,
            non_zero_threshold=0.0,
            max_activation_scale=float(max_activation_scale),
        )
        # Dynamic threshold: 30% of mean designed-sentence activation, min 1.0
        mean_act = activation_metrics["mean_summary_activation"]
        dynamic_threshold = max(1.0, 0.3 * mean_act) if mean_act > 0 else non_zero_threshold

        # Recompute activation non-zero counts with dynamic threshold
        # and sync per-sentence flags to keep sample-level labels consistent
        # with aggregate scores.
        _act_results = activation_metrics["sentence_results"]
        _act_nz = 0
        for r in _act_results:
            _is_non_zero = float(r.get("summary_activation", 0.0) or 0.0) > dynamic_threshold
            r["is_non_zero"] = _is_non_zero
            if _is_non_zero:
                _act_nz += 1
        _act_total = len(_act_results)
        activation_metrics["non_zero_count"] = _act_nz
        activation_metrics["score_non_zero_rate"] = _act_nz / _act_total if _act_total else 0.0

        boundary_metrics = _run_sentence_batch(
            module=module,
            sentences=boundary_sentences,
            non_zero_threshold=0.0,
            max_activation_scale=float(max_activation_scale),
        )
        # Recompute boundary counts with same dynamic threshold
        # and sync per-sentence flags.
        _bnd_results = boundary_metrics["sentence_results"]
        _bnd_nz = 0
        for r in _bnd_results:
            _is_non_zero = float(r.get("summary_activation", 0.0) or 0.0) > dynamic_threshold
            r["is_non_zero"] = _is_non_zero
            if _is_non_zero:
                _bnd_nz += 1
        _bnd_total = len(_bnd_results)
        boundary_metrics["non_zero_count"] = _bnd_nz
        boundary_metrics["score_non_zero_rate"] = _bnd_nz / _bnd_total if _bnd_total else 0.0

        boundary_non_activation_count = _bnd_total - _bnd_nz
        boundary_non_activation_rate = (
            boundary_non_activation_count / _bnd_total
            if _bnd_total > 0
            else None
        )
        runtime_batches.append(activation_metrics.get("runtime_batch"))

        hypothesis_results.append(
            {
                "hypothesis_index": hypothesis_index,
                "hypothesis": hypothesis_text,
                "designed_sentences": sentences,
                "boundary_sentences": boundary_sentences,
                "input_activation_context": build_input_activation_context(
                    hypothesis=hypothesis_text,
                    designed_sentences=sentences,
                ),
                "input_boundary_context": build_input_boundary_context(
                    hypothesis=hypothesis_text,
                    boundary_sentences=boundary_sentences,
                ),
                "sentence_results": activation_metrics["sentence_results"],
                "non_zero_count": activation_metrics["non_zero_count"],
                "total_sentences": activation_metrics["total_sentences"],
                "score_non_zero_rate": activation_metrics["score_non_zero_rate"],
                "mean_summary_activation": activation_metrics["mean_summary_activation"],
                "max_summary_activation": activation_metrics["max_summary_activation"],
                "dynamic_threshold": dynamic_threshold,
                "boundary_sentence_results": boundary_metrics["sentence_results"],
                "boundary_non_zero_count": boundary_metrics["non_zero_count"],
                "total_boundary_sentences": boundary_metrics["total_sentences"],
                "score_boundary_non_zero_rate": boundary_metrics["score_non_zero_rate"],
                "boundary_non_activation_count": boundary_non_activation_count,
                "score_boundary_non_activation_rate": boundary_non_activation_rate,
                "mean_boundary_summary_activation": boundary_metrics["mean_summary_activation"],
                "max_boundary_summary_activation": boundary_metrics["max_summary_activation"],
            }
        )

    if hypothesis_results:
        overall_score = sum(item["score_non_zero_rate"] for item in hypothesis_results) / len(hypothesis_results)
        boundary_values = [
            item["score_boundary_non_activation_rate"]
            for item in hypothesis_results
            if item.get("score_boundary_non_activation_rate") is not None
        ]
        overall_boundary_score = (
            sum(boundary_values) / len(boundary_values) if boundary_values else None
        )
    else:
        overall_score = 0.0
        overall_boundary_score = None

    return {
        "side": "input",
        "non_zero_threshold": "dynamic (0.3 * mean_activation per hypothesis, min 1.0)",
        "max_activation_scale": float(max_activation_scale),
        "hypothesis_results": hypothesis_results,
        "overall_score_non_zero_rate": overall_score,
        "overall_score_boundary_non_activation_rate": overall_boundary_score,
        "runtime_batches": runtime_batches,
    }
