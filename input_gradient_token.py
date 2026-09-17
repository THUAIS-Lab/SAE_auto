from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from model_with_sae import ModelWithSAEModule
from workflow_step_utils import DEFAULT_MODEL_CHECKPOINT_PATH

DEFAULT_CANONICAL_MAP_PATH = Path("support_info") / "canonical_map.txt"
DEFAULT_GRADIENT_SAE_ROOT = os.environ.get(
    "SAE_ROOT",
    "gemma-scope-2b-pt-res",
)


def _extract_average_l0_from_canonical_map(
    *,
    canonical_map_path: Path,
    layer_id: str,
    width: str,
) -> Optional[str]:
    if not canonical_map_path.exists():
        return None

    target_id = f"layer_{layer_id}/width_{width}/canonical"
    in_target_block = False
    path_pattern = re.compile(
        rf"layer_{re.escape(layer_id)}/width_{re.escape(width)}/average_l0_([0-9]+(?:\.[0-9]+)?)"
    )
    with canonical_map_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("- id:"):
                current_id = line.split(":", 1)[1].strip()
                in_target_block = current_id == target_id
                continue
            if in_target_block and line.startswith("path:"):
                match = path_pattern.search(line.split(":", 1)[1].strip())
                if match:
                    return match.group(1)
                return None
    return None


def _build_default_sae_path(
    *,
    layer_id: str,
    width: str,
    sae_root: str,
    average_l0: Optional[str],
    canonical_map_path: Optional[str],
) -> Tuple[str, str]:
    resolved_average_l0 = average_l0
    if not resolved_average_l0 and canonical_map_path:
        resolved_average_l0 = _extract_average_l0_from_canonical_map(
            canonical_map_path=Path(canonical_map_path),
            layer_id=layer_id,
            width=width,
        )
    if not resolved_average_l0:
        resolved_average_l0 = "70"

    sae_path = (
        Path(str(sae_root))
        / f"layer_{layer_id}"
        / f"width_{width}"
        / f"average_l0_{resolved_average_l0}"
    )
    return str(sae_path), str(resolved_average_l0)


def _resolve_sae_path(args: argparse.Namespace) -> Tuple[str, Optional[str]]:
    if args.sae_path:
        return str(args.sae_path), None

    sae_uri, resolved_average_l0 = _build_default_sae_path(
        layer_id=str(int(args.layer_id)),
        width=str(args.width),
        sae_root=str(args.sae_root),
        average_l0=str(args.sae_average_l0) if args.sae_average_l0 is not None else None,
        canonical_map_path=str(args.sae_canonical_map),
    )
    return sae_uri, str(resolved_average_l0)


def _resolve_bos_token_id(tokenizer) -> Optional[int]:
    bos_id = getattr(tokenizer, "bos_token_id", None)
    return int(bos_id) if bos_id is not None else None


def _select_non_bos_positions(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    bos_token_id: Optional[int],
) -> torch.Tensor:
    valid_positions = attention_mask[0].to(dtype=torch.bool).nonzero(as_tuple=False).flatten()
    if valid_positions.numel() == 0:
        raise RuntimeError("Prompt produced no valid tokens.")

    positions = valid_positions
    first_valid = int(valid_positions[0].item())
    first_token_id = int(input_ids[0, first_valid].item())
    if valid_positions.numel() > 1 and (bos_token_id is None or first_token_id == int(bos_token_id)):
        positions = valid_positions[1:]
    if positions.numel() == 0:
        positions = valid_positions
    return positions


def _token_text_map(tokenizer, token_ids: List[int]) -> Dict[int, str]:
    if not token_ids:
        return {}
    token_texts = tokenizer.convert_ids_to_tokens(token_ids)
    return {int(tok_id): str(tok_text) for tok_id, tok_text in zip(token_ids, token_texts)}


def _run_gradient_scan_with_module(
    args: argparse.Namespace,
    *,
    module: ModelWithSAEModule,
    sae_path: str,
    resolved_average_l0: Optional[str],
) -> Dict[str, Any]:
    if module.model is None or module.tokenizer is None:
        raise RuntimeError("Failed to initialize model/tokenizer.")
    if module.use_hooked_transformer:
        raise RuntimeError("input_gradient_token.py currently supports the HuggingFace model path only.")

    tokenizer = module.tokenizer
    model = module.model
    model.eval()
    model.zero_grad(set_to_none=True)

    encoded = tokenizer(
        str(args.prompt),
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=int(args.max_length),
    )
    input_ids = encoded["input_ids"].to(module.device)
    attention_mask = module._coerce_attention_mask(input_ids, encoded.get("attention_mask"))

    embed_layer = model.get_input_embeddings()
    if embed_layer is None or not hasattr(embed_layer, "weight"):
        raise RuntimeError("Model input embedding layer is not available.")

    inputs_embeds = embed_layer(input_ids).detach().clone()
    inputs_embeds.requires_grad_(True)

    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states.")
    layer_idx = module._resolve_hidden_state_index(len(hidden_states))
    if layer_idx is None:
        raise RuntimeError(f"Layer {module.layer} not available in hidden states.")

    layer_activations = hidden_states[layer_idx]
    sae_features = module._encode_with_sae(layer_activations, return_pre=True)
    if sae_features.ndim != 3:
        raise RuntimeError("SAE pre-activation output must be [batch, seq, n_features].")
    feature_id = int(args.feature_id)
    if feature_id < 0 or feature_id >= sae_features.shape[-1]:
        raise ValueError(f"Feature id {feature_id} out of range for n_features={sae_features.shape[-1]}.")

    positions = _select_non_bos_positions(
        input_ids=input_ids,
        attention_mask=attention_mask,
        bos_token_id=_resolve_bos_token_id(tokenizer),
    )
    feature_activations = sae_features[0, :, feature_id]
    candidate_values = feature_activations.index_select(0, positions)
    max_value, max_local_idx = candidate_values.max(dim=0)
    selected_position = int(positions[int(max_local_idx.item())].item())

    embedding_weight = embed_layer.weight.detach().to(device=inputs_embeds.device, dtype=torch.float32)
    aggregated_ranking_scores = torch.full(
        (int(embedding_weight.shape[0]),),
        float("-inf"),
        dtype=torch.float32,
        device=embedding_weight.device,
    )
    aggregated_signed_scores = torch.zeros_like(aggregated_ranking_scores)
    aggregated_positions = torch.full_like(aggregated_ranking_scores, -1, dtype=torch.long)

    position_records: List[Dict[str, Any]] = []
    position_list = [int(x) for x in positions.detach().cpu().tolist()]
    for idx, position in enumerate(position_list):
        target_value = feature_activations[position]
        grad_tensor = torch.autograd.grad(
            target_value,
            inputs_embeds,
            retain_graph=idx < len(position_list) - 1,
        )[0]
        grad_vec = grad_tensor[0, position].detach().to(dtype=torch.float32)
        grad_norm = float(torch.linalg.vector_norm(grad_vec).item())
        if grad_norm == 0.0:
            continue

        vocab_scores_for_position = torch.matmul(embedding_weight, grad_vec)
        if str(args.rank_by) == "positive":
            ranking_scores_for_position = vocab_scores_for_position
        elif str(args.rank_by) == "negative":
            ranking_scores_for_position = -vocab_scores_for_position
        else:
            ranking_scores_for_position = vocab_scores_for_position.abs()

        better_mask = ranking_scores_for_position > aggregated_ranking_scores
        aggregated_ranking_scores = torch.where(
            better_mask,
            ranking_scores_for_position,
            aggregated_ranking_scores,
        )
        aggregated_signed_scores = torch.where(
            better_mask,
            vocab_scores_for_position,
            aggregated_signed_scores,
        )
        aggregated_positions = torch.where(
            better_mask,
            torch.full_like(aggregated_positions, int(position)),
            aggregated_positions,
        )

        token_id = int(input_ids[0, position].item())
        token_text = str(tokenizer.convert_ids_to_tokens([token_id])[0])
        position_records.append(
            {
                "position": int(position),
                "token_id": int(token_id),
                "token_text": token_text,
                "pre_activation": float(target_value.detach().cpu().item()),
                "gradient_norm": float(grad_norm),
            }
        )

    if not position_records:
        raise RuntimeError("All selected input-position gradient norms are zero.")

    vocab_scores = aggregated_signed_scores
    excluded_token_ids: List[int] = []
    if not bool(args.include_special_tokens) and hasattr(tokenizer, "all_special_ids"):
        excluded_token_ids = [
            int(x) for x in getattr(tokenizer, "all_special_ids", []) if 0 <= int(x) < int(vocab_scores.shape[0])
        ]

    ranking_scores = aggregated_ranking_scores
    if excluded_token_ids:
        ranking_scores = ranking_scores.clone()
        ranking_scores[torch.tensor(excluded_token_ids, device=ranking_scores.device)] = float("-inf")

    top_k = min(int(args.top_k), int(ranking_scores.shape[0]))
    top_scores, top_ids = torch.topk(ranking_scores, k=top_k, dim=0)
    top_token_ids = [int(x) for x in top_ids.detach().cpu().tolist()]
    token_texts = _token_text_map(tokenizer, top_token_ids)
    signed_scores = vocab_scores.index_select(0, top_ids).detach().cpu().tolist()
    source_positions = aggregated_positions.index_select(0, top_ids).detach().cpu().tolist()

    entries: List[Dict[str, Any]] = []
    for rank, (token_id, rank_score, signed_score, source_position) in enumerate(
        zip(top_token_ids, top_scores.detach().cpu().tolist(), signed_scores, source_positions),
        start=1,
    ):
        entries.append(
            {
                "rank": int(rank),
                "token_id": int(token_id),
                "token_text": token_texts.get(int(token_id), ""),
                "gradient_score": float(signed_score),
                "gradient_magnitude": float(abs(float(signed_score))),
                "ranking_score": float(rank_score),
                "source_position": int(source_position),
            }
        )

    prompt_token_ids = [int(x) for x in input_ids[0].detach().cpu().tolist()]
    prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_token_ids)
    selected_token_id = int(input_ids[0, selected_position].item())
    selected_token_text = str(tokenizer.convert_ids_to_tokens([selected_token_id])[0])

    prompt_id = str(args.prompt_id).strip()
    output_root = Path(str(args.output_root))
    feature_dir = (
        output_root
        / f"layer-{int(args.layer_id)}"
        / f"feature-{feature_id}"
        / "gradient_token"
        / prompt_id
    )
    feature_dir.mkdir(parents=True, exist_ok=True)

    prompt_payload = {
        "prompt_id": prompt_id,
        "prompt": str(args.prompt),
        "prompt_tokens": [
            {"position": int(i), "token_id": int(tok_id), "token_text": str(tok_text)}
            for i, (tok_id, tok_text) in enumerate(zip(prompt_token_ids, prompt_tokens))
        ],
    }
    (feature_dir / "prompt.json").write_text(
        json.dumps(prompt_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    payload = {
        "layer_id": int(args.layer_id),
        "feature_id": int(feature_id),
        "source": "gradient_token",
        "prompt_id": prompt_id,
        "prompt": str(args.prompt),
        "objective": "max_over_positions_sae_pre_activation_gradients",
        "rank_by": str(args.rank_by),
        "aggregation": "max_over_non_bos_positions",
        "top_k": int(top_k),
        "selected_position": int(selected_position),
        "selected_token_id": int(selected_token_id),
        "selected_token_text": selected_token_text,
        "selected_pre_activation": float(max_value.detach().cpu().item()),
        "analyzed_position_count": int(len(position_records)),
        "position_gradients": position_records,
        "prompt_tokens": [
            {"position": int(i), "token_id": int(tok_id), "token_text": str(tok_text)}
            for i, (tok_id, tok_text) in enumerate(zip(prompt_token_ids, prompt_tokens))
        ],
        "top_tokens": entries,
    }
    (feature_dir / "top_tokens.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "layer_id": int(args.layer_id),
        "feature_id": int(feature_id),
        "model_checkpoint_path": str(args.model_checkpoint_path),
        "sae_path": str(sae_path),
        "resolved_average_l0": resolved_average_l0,
        "source": "gradient_token",
        "prompt_id": prompt_id,
        "prompt": str(args.prompt),
        "objective": "max_over_positions_sae_pre_activation_gradients",
        "rank_by": str(args.rank_by),
        "aggregation": "max_over_non_bos_positions",
        "top_k": int(top_k),
        "selected_position": int(selected_position),
        "selected_token_id": int(selected_token_id),
        "selected_token_text": selected_token_text,
        "selected_pre_activation": float(max_value.detach().cpu().item()),
        "analyzed_position_count": int(len(position_records)),
        "outputs": {
            "output_root": str(output_root.resolve()),
            "prompt_path": str((feature_dir / "prompt.json").resolve()),
            "top_tokens_path": str((feature_dir / "top_tokens.json").resolve()),
        },
    }
    return summary


def run_gradient_scan(args: argparse.Namespace) -> Dict[str, Any]:
    sae_path, resolved_average_l0 = _resolve_sae_path(args)

    module = ModelWithSAEModule(
        llm_name=str(args.model_checkpoint_path),
        sae_path=str(sae_path),
        sae_layer=int(args.layer_id),
        feature_index=int(args.feature_id),
        device=str(args.device),
    )
    return _run_gradient_scan_with_module(
        args,
        module=module,
        sae_path=str(sae_path),
        resolved_average_l0=resolved_average_l0,
    )


def _parse_int_list(values: Optional[List[str]], *, name: str) -> List[int]:
    if not values:
        raise ValueError(f"--{name} must contain at least one value in batch mode.")
    parsed: List[int] = []
    for raw in values:
        for part in str(raw).split(","):
            item = part.strip()
            if item:
                parsed.append(int(item))
    if not parsed:
        raise ValueError(f"--{name} must contain at least one integer in batch mode.")
    return parsed


def _build_batch_jobs(args: argparse.Namespace) -> List[argparse.Namespace]:
    layers = _parse_int_list(args.layers, name="layers")
    features = _parse_int_list(args.features, name="features")
    prompt_ids = list(args.prompt_ids or [])
    prompts = list(args.prompts or [])
    if not prompt_ids:
        raise ValueError("--prompt-ids must contain at least one value in batch mode.")
    if len(prompt_ids) != len(prompts):
        raise ValueError("--prompt-ids and --prompts must have the same length in batch mode.")
    for prompt_id in prompt_ids:
        prompt_id_str = str(prompt_id).strip()
        if not prompt_id_str:
            raise ValueError("--prompt-ids must not contain empty values.")
        if "/" in prompt_id_str or "\\" in prompt_id_str or prompt_id_str in {".", ".."}:
            raise ValueError("--prompt-ids values must be single directory names, not paths.")

    sae_paths = list(args.sae_paths or [])
    if sae_paths and len(sae_paths) != len(layers):
        raise ValueError("--sae-paths must be omitted or have the same length as --layers.")
    sae_path_by_layer = {int(layer): str(path) for layer, path in zip(layers, sae_paths)}

    worker_count = int(args.worker_count)
    worker_id = int(args.worker_id)
    if worker_count <= 0:
        raise ValueError("--worker-count must be > 0 in batch mode.")
    if worker_id < 0 or worker_id >= worker_count:
        raise ValueError("--worker-id must satisfy 0 <= worker_id < worker_count in batch mode.")

    jobs: List[argparse.Namespace] = []
    idx = 0
    for layer in layers:
        for feature in features:
            for prompt_id, prompt in zip(prompt_ids, prompts):
                if idx % worker_count == worker_id:
                    job_args = argparse.Namespace(**vars(args))
                    job_args.batch_mode = False
                    job_args.layer_id = int(layer)
                    job_args.feature_id = int(feature)
                    job_args.prompt_id = str(prompt_id)
                    job_args.prompt = str(prompt)
                    if layer in sae_path_by_layer:
                        job_args.sae_path = sae_path_by_layer[layer]
                    jobs.append(job_args)
                idx += 1
    return jobs


def _write_batch_job_log(
    *,
    args: argparse.Namespace,
    job_args: argparse.Namespace,
    payload: Dict[str, Any],
) -> None:
    job_log_dir = getattr(args, "job_log_dir", None)
    if not job_log_dir:
        return

    log_dir = Path(str(job_log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (
        log_dir
        / f"layer{int(job_args.layer_id)}_feature{int(job_args.feature_id)}_{str(job_args.prompt_id)}.log"
    )
    log_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_gradient_batch(args: argparse.Namespace) -> Dict[str, Any]:
    jobs = _build_batch_jobs(args)
    if not jobs:
        raise RuntimeError("No batch jobs were assigned to this worker.")

    current_key: Optional[Tuple[int, str]] = None
    module: Optional[ModelWithSAEModule] = None
    shared_model = None
    shared_tokenizer = None
    ok_count = 0
    fail_count = 0
    failures: List[Dict[str, Any]] = []

    start_time = time.time()
    try:
        for job_index, job_args in enumerate(jobs, start=1):
            sae_path, resolved_average_l0 = _resolve_sae_path(job_args)
            key = (int(job_args.layer_id), str(sae_path))
            if module is None or current_key != key:
                if module is not None:
                    shared_model = module.model
                    shared_tokenizer = module.tokenizer
                    module.model = None
                    module.tokenizer = None
                    del module
                    module = None
                    if str(args.device).startswith("cuda"):
                        torch.cuda.empty_cache()

                print(
                    json.dumps(
                        {
                            "event": "load_module",
                            "worker_id": int(args.worker_id),
                            "layer_id": int(job_args.layer_id),
                            "sae_path": str(sae_path),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                module = ModelWithSAEModule(
                    llm_name=str(args.model_checkpoint_path),
                    sae_path=str(sae_path),
                    sae_layer=int(job_args.layer_id),
                    feature_index=-1,
                    device=str(args.device),
                    model=shared_model,
                    tokenizer=shared_tokenizer,
                )
                current_key = key
                shared_model = module.model
                shared_tokenizer = module.tokenizer

            print(
                json.dumps(
                    {
                        "event": "start_job",
                        "worker_id": int(args.worker_id),
                        "job_index": int(job_index),
                        "job_count": int(len(jobs)),
                        "layer_id": int(job_args.layer_id),
                        "feature_id": int(job_args.feature_id),
                        "prompt_id": str(job_args.prompt_id),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            job_start = time.time()
            try:
                assert module is not None
                summary = _run_gradient_scan_with_module(
                    job_args,
                    module=module,
                    sae_path=str(sae_path),
                    resolved_average_l0=resolved_average_l0,
                )
                job_elapsed = round(time.time() - job_start, 2)
                ok_count += 1
                _write_batch_job_log(
                    args=args,
                    job_args=job_args,
                    payload={"status": "ok", "elapsed_seconds": job_elapsed, **summary},
                )
                print(
                    json.dumps(
                        {
                            "event": "ok_job",
                            "worker_id": int(args.worker_id),
                            "elapsed_seconds": job_elapsed,
                            "layer_id": int(job_args.layer_id),
                            "feature_id": int(job_args.feature_id),
                            "prompt_id": str(job_args.prompt_id),
                            "top_tokens_path": summary["outputs"]["top_tokens_path"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception as exc:
                job_elapsed = round(time.time() - job_start, 2)
                fail_count += 1
                failure = {
                    "layer_id": int(job_args.layer_id),
                    "feature_id": int(job_args.feature_id),
                    "prompt_id": str(job_args.prompt_id),
                    "error": repr(exc),
                }
                failures.append(failure)
                _write_batch_job_log(
                    args=args,
                    job_args=job_args,
                    payload={"status": "fail", "elapsed_seconds": job_elapsed, **failure},
                )
                print(
                    json.dumps(
                        {
                            "event": "fail_job",
                            "worker_id": int(args.worker_id),
                            "elapsed_seconds": job_elapsed,
                            **failure,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        if module is not None:
            shared_model = module.model
            shared_tokenizer = module.tokenizer
            module.model = None
            module.tokenizer = None
            del module
        del shared_model
        del shared_tokenizer
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    return {
        "status": "ok" if fail_count == 0 else "partial",
        "worker_id": int(args.worker_id),
        "worker_count": int(args.worker_count),
        "assigned_job_count": int(len(jobs)),
        "ok_count": int(ok_count),
        "fail_count": int(fail_count),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "failures": failures[:20],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use input-embedding gradients from all non-BOS SAE pre-activations to rank vocab tokens."
        )
    )
    parser.add_argument(
        "--model-checkpoint-path",
        type=str,
        default=DEFAULT_MODEL_CHECKPOINT_PATH,
        help="Model name/path for HuggingFace loading.",
    )
    parser.add_argument("--layer-id", type=int, default=0, help="Target layer id.")
    parser.add_argument("--feature-id", type=int, default=0, help="Target SAE feature id.")
    parser.add_argument("--prompt", type=str, default="", help="Input sentence/prompt for gradient analysis.")
    parser.add_argument(
        "--prompt-id",
        type=str,
        default="prompt-0001",
        help="Directory-safe id used under gradient_token/ for this prompt.",
    )
    parser.add_argument(
        "--sae-path",
        type=str,
        default=None,
        help="Direct local SAE path. If omitted, built from SAE root/width/average_l0 options.",
    )
    parser.add_argument(
        "--sae-root",
        type=str,
        default=DEFAULT_GRADIENT_SAE_ROOT,
        help="Local SAE root used when --sae-path is omitted.",
    )
    parser.add_argument("--width", type=str, default="16k", help="SAE width used when --sae-path is omitted.")
    parser.add_argument(
        "--sae-average-l0",
        type=str,
        default=None,
        help="Optional average_l0 override used when --sae-path is omitted.",
    )
    parser.add_argument(
        "--sae-canonical-map",
        type=str,
        default=str(DEFAULT_CANONICAL_MAP_PATH),
        help="Path to canonical_map.txt for average_l0 resolution.",
    )
    parser.add_argument("--top-k", type=int, default=50, help="Top-k vocab tokens to keep.")
    parser.add_argument(
        "--rank-by",
        choices=["abs", "positive", "negative"],
        default="positive",
        help="How to rank vocab directional-gradient scores.",
    )
    parser.add_argument(
        "--include-special-tokens",
        action="store_true",
        help="Include tokenizer special tokens in vocab ranking.",
    )
    parser.add_argument("--max-length", type=int, default=128, help="Maximum prompt token length.")
    parser.add_argument("--output-root", type=str, default="initial_observation", help="Output root directory.")
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device to run model inference on.",
    )
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Run a persistent worker over many layer/feature/prompt jobs.",
    )
    parser.add_argument("--worker-id", type=int, default=0, help="Batch worker id, zero-indexed.")
    parser.add_argument("--worker-count", type=int, default=1, help="Total batch worker count.")
    parser.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help="Layer ids for batch mode. Values may be space-separated or comma-separated.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Feature ids for batch mode. Values may be space-separated or comma-separated.",
    )
    parser.add_argument(
        "--prompt-ids",
        nargs="+",
        default=None,
        help="Prompt ids for batch mode. Must align with --prompts.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="Prompt strings for batch mode. Must align with --prompt-ids.",
    )
    parser.add_argument(
        "--sae-paths",
        nargs="+",
        default=None,
        help="Optional SAE paths for batch mode, aligned with --layers.",
    )
    parser.add_argument(
        "--job-log-dir",
        type=str,
        default=None,
        help="Optional directory for per-job summary logs in batch mode.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if bool(args.batch_mode):
        _build_batch_jobs(args)
        return

    if int(args.layer_id) < 0:
        raise ValueError("--layer-id must be >= 0.")
    if int(args.feature_id) < 0:
        raise ValueError("--feature-id must be >= 0.")
    if int(args.top_k) <= 0:
        raise ValueError("--top-k must be > 0.")
    if int(args.max_length) <= 0:
        raise ValueError("--max-length must be > 0.")
    if not str(args.prompt).strip():
        raise ValueError("--prompt must be non-empty.")
    prompt_id = str(args.prompt_id).strip()
    if not prompt_id:
        raise ValueError("--prompt-id must be non-empty.")
    if "/" in prompt_id or "\\" in prompt_id or prompt_id in {".", ".."}:
        raise ValueError("--prompt-id must be a single directory name, not a path.")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)

    start_time = time.time()
    if bool(args.batch_mode):
        summary = run_gradient_batch(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        summary = run_gradient_scan(args)
        elapsed = time.time() - start_time
        print(json.dumps({"status": "ok", "elapsed_seconds": round(elapsed, 2), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
