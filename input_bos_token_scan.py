from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from model_with_sae import ModelWithSAEModule
from workflow_step_utils import DEFAULT_MODEL_CHECKPOINT_PATH

DEFAULT_CANONICAL_MAP_PATH = Path("support_info") / "canonical_map.txt"


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
    release: str,
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

    sae_uri = (
        "sae-lens://"
        f"release={release};"
        f"sae_id=layer_{layer_id}/width_{width}/average_l0_{resolved_average_l0}"
    )
    return sae_uri, str(resolved_average_l0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan <bos>+token prompts through model+SAE and keep per-feature top-k "
            "activating tokens (token position only)."
        )
    )
    parser.add_argument(
        "--model-checkpoint-path",
        type=str,
        default=DEFAULT_MODEL_CHECKPOINT_PATH,
        help="Model name/path for HuggingFace or TransformerLens loading.",
    )
    parser.add_argument("--layer-id", type=int, required=True, help="Target layer id.")
    parser.add_argument(
        "--sae-path",
        type=str,
        default=None,
        help="Direct SAE path/URI. If omitted, built from SAE release/width/average_l0 options.",
    )
    parser.add_argument(
        "--sae-release",
        type=str,
        default="gemma-scope-2b-pt-res",
        help="SAE release used when --sae-path is omitted.",
    )
    parser.add_argument(
        "--width",
        type=str,
        default="16k",
        help="SAE width used when --sae-path is omitted.",
    )
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
    parser.add_argument(
        "--feature-ids-file",
        type=str,
        default=None,
        help="Text file with one feature id per line. Ignored by --all-features.",
    )
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="Collect top-k for all SAE features instead of a subset from --feature-ids-file.",
    )
    parser.add_argument(
        "--manual-token-texts-file",
        type=str,
        default=None,
        help="Text file containing tokenizer token texts (one token text per line).",
    )
    parser.add_argument(
        "--random-sample-size",
        type=int,
        default=0,
        help="Randomly sample this many vocab token ids and merge with manual tokens.",
    )
    parser.add_argument(
        "--scan-full-vocab",
        action="store_true",
        help="Scan full vocabulary token ids (merged with manual token texts).",
    )
    parser.add_argument(
        "--include-special-tokens",
        action="store_true",
        help="Include special tokens in random/full-vocab candidate pool.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for token sampling.")
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="<bos>",
        help=(
            "Prompt template for each candidate token. Use {token} as the insertion point; "
            "if omitted, the candidate token is appended after the template. Default preserves <bos>+token."
        ),
    )
    parser.add_argument(
        "--prompt-id",
        type=str,
        default="prompt-0001",
        help="Directory-safe id used under bos_token/ for this prompt template.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for prompt-template token prompts.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-k tokens to keep per feature.")
    parser.add_argument(
        "--activation-threshold",
        type=float,
        default=0.0,
        help="Only activations greater than this threshold are recorded.",
    )
    parser.add_argument(
        "--feature-chunk-size",
        type=int,
        default=4096,
        help="Feature chunk size for top-k updates to control memory.",
    )
    parser.add_argument(
        "--save-all-activated",
        action="store_true",
        help="Stream all activated (feature, token, activation) events to JSONL for debugging.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="initial_observation",
        help="Output root directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device to run model inference on.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.layer_id) < 0:
        raise ValueError("--layer-id must be >= 0.")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be > 0.")
    if int(args.top_k) <= 0:
        raise ValueError("--top-k must be > 0.")
    if int(args.feature_chunk_size) <= 0:
        raise ValueError("--feature-chunk-size must be > 0.")
    if int(args.random_sample_size) < 0:
        raise ValueError("--random-sample-size must be >= 0.")
    if not bool(args.all_features) and not args.feature_ids_file:
        raise ValueError("Either --all-features or --feature-ids-file must be provided.")
    if not bool(args.scan_full_vocab) and int(args.random_sample_size) == 0 and not args.manual_token_texts_file:
        raise ValueError(
            "No tokens to scan: provide --scan-full-vocab, --random-sample-size > 0, or --manual-token-texts-file."
        )
    if str(args.prompt_template).count("{token}") > 1:
        raise ValueError("--prompt-template may contain at most one {token} placeholder.")
    if "<bos>" not in str(args.prompt_template):
        raise ValueError("--prompt-template must contain <bos>.")
    prompt_id = str(args.prompt_id).strip()
    if not prompt_id:
        raise ValueError("--prompt-id must be non-empty.")
    if "/" in prompt_id or "\\" in prompt_id or prompt_id in {".", ".."}:
        raise ValueError("--prompt-id must be a single directory name, not a path.")


def _resolve_sae_path(args: argparse.Namespace) -> Tuple[str, Optional[str]]:
    if args.sae_path:
        return str(args.sae_path), None

    sae_uri, resolved_average_l0 = _build_default_sae_path(
        layer_id=str(int(args.layer_id)),
        width=str(args.width),
        release=str(args.sae_release),
        average_l0=str(args.sae_average_l0) if args.sae_average_l0 is not None else None,
        canonical_map_path=str(args.sae_canonical_map),
    )
    return sae_uri, str(resolved_average_l0)


def _load_feature_ids(feature_ids_file: str) -> List[int]:
    path = Path(feature_ids_file)
    if not path.exists():
        raise FileNotFoundError(f"feature ids file not found: {path}")

    ids: List[int] = []
    seen: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        value = int(stripped)
        if value < 0:
            raise ValueError(f"Invalid feature id < 0: {value}")
        if value not in seen:
            ids.append(value)
            seen.add(value)
    if not ids:
        raise ValueError(f"No valid feature ids found in {path}")
    return sorted(ids)


def _load_manual_token_texts(token_texts_file: str) -> List[str]:
    path = Path(token_texts_file)
    if not path.exists():
        raise FileNotFoundError(f"manual token file not found: {path}")

    texts: List[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw:
            continue
        if raw.strip().startswith("#"):
            continue
        token_text = raw.rstrip("\r\n")
        if token_text not in seen:
            texts.append(token_text)
            seen.add(token_text)
    return texts


def _resolve_bos_token_id(tokenizer) -> int:
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None:
        return int(bos_id)

    fallback_token = getattr(tokenizer, "bos_token", None)
    if not fallback_token:
        fallback_token = getattr(tokenizer, "cls_token", None)
    if not fallback_token:
        fallback_token = getattr(tokenizer, "eos_token", None)
    if not fallback_token:
        raise RuntimeError("Tokenizer bos token id/token is required but not available.")

    encoded = tokenizer(str(fallback_token), return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.get("input_ids")
    if input_ids is None or input_ids.numel() == 0:
        raise RuntimeError("Failed to infer bos token id from fallback special token.")
    return int(input_ids[0, 0].item())


def _encode_template_fragment(tokenizer, text: str, bos_token_id: int) -> List[int]:
    if not text:
        return []

    token_ids: List[int] = []
    parts = text.split("<bos>")
    for idx, part in enumerate(parts):
        if part:
            encoded = tokenizer(part, add_special_tokens=False, return_tensors=None)
            raw_ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
            if raw_ids and isinstance(raw_ids[0], list):
                raw_ids = raw_ids[0]
            token_ids.extend(int(x) for x in raw_ids)
        if idx < len(parts) - 1:
            token_ids.append(int(bos_token_id))
    return token_ids


def _resolve_prompt_template(
    *,
    tokenizer,
    template: str,
    bos_token_id: int,
) -> Tuple[List[int], List[int], int]:
    if "{token}" in template:
        prefix, suffix = template.split("{token}", 1)
    else:
        prefix, suffix = template, ""

    prefix_ids = _encode_template_fragment(tokenizer, prefix, bos_token_id)
    suffix_ids = _encode_template_fragment(tokenizer, suffix, bos_token_id)
    return prefix_ids, suffix_ids, len(prefix_ids)


def _build_prompt_batch(
    *,
    batch_token_ids: Sequence[int],
    prefix_ids: Sequence[int],
    suffix_ids: Sequence[int],
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = [
        [*prefix_ids, int(token_id), *suffix_ids]
        for token_id in batch_token_ids
    ]
    if not rows or not rows[0]:
        raise RuntimeError("Prompt template produced empty input ids.")
    input_ids = torch.tensor(rows, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return input_ids, attention_mask


def _resolve_manual_token_ids(tokenizer, manual_token_texts: Sequence[str]) -> Tuple[List[int], List[str]]:
    if not manual_token_texts:
        return [], []

    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None
    if not isinstance(vocab, dict):
        raise RuntimeError("Tokenizer.get_vocab() is required for manual token text resolution.")

    ids: List[int] = []
    seen: set[int] = set()
    missing: List[str] = []
    for token_text in manual_token_texts:
        token_id = vocab.get(token_text)
        if token_id is None:
            missing.append(token_text)
            continue
        token_id = int(token_id)
        if token_id not in seen:
            ids.append(token_id)
            seen.add(token_id)
    return ids, missing


def _build_token_scan_list(
    *,
    tokenizer,
    manual_token_ids: Sequence[int],
    scan_full_vocab: bool,
    random_sample_size: int,
    include_special_tokens: bool,
    seed: int,
) -> Tuple[List[int], Dict[str, int]]:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        raise RuntimeError("Tokenizer.vocab_size is required.")
    vocab_size = int(vocab_size)
    if vocab_size <= 0:
        raise RuntimeError("Tokenizer.vocab_size must be positive.")

    pool_ids = list(range(vocab_size))
    if not include_special_tokens and hasattr(tokenizer, "all_special_ids"):
        special_ids = {int(x) for x in getattr(tokenizer, "all_special_ids", [])}
        pool_ids = [x for x in pool_ids if x not in special_ids]

    manual_unique: List[int] = []
    manual_seen: set[int] = set()
    for token_id in manual_token_ids:
        value = int(token_id)
        if value < 0 or value >= vocab_size:
            continue
        if value not in manual_seen:
            manual_unique.append(value)
            manual_seen.add(value)

    pool_without_manual = [x for x in pool_ids if x not in manual_seen]
    sampled: List[int] = []
    if scan_full_vocab:
        sampled = pool_without_manual
    elif random_sample_size > 0:
        sample_size = min(int(random_sample_size), len(pool_without_manual))
        rng = random.Random(int(seed))
        sampled = rng.sample(pool_without_manual, sample_size)

    final_ids = manual_unique + sampled
    return final_ids, {
        "vocab_size": vocab_size,
        "pool_size": len(pool_ids),
        "manual_token_count": len(manual_unique),
        "sampled_token_count": len(sampled),
        "evaluated_token_count": len(final_ids),
    }


def _get_sae_features(
    module: ModelWithSAEModule,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if module.use_hooked_transformer:
        hook_name = module.hook_name
        if hook_name is None:
            raise RuntimeError("hook_name is required for HookedTransformer path.")
        _, cache = module.model.run_with_cache(input_ids, names_filter=[hook_name])
        layer_activations = cache[hook_name]
    else:
        if module.layer is None:
            raise RuntimeError("layer is required for HuggingFace path.")
        hf_attention_mask = module._coerce_attention_mask(input_ids, attention_mask)
        outputs = module.model(
            input_ids=input_ids,
            attention_mask=hf_attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None or module.layer >= len(hidden_states):
            raise RuntimeError(f"Layer {module.layer} not available in hidden states.")
        layer_activations = hidden_states[module.layer]

    sae_features = module._encode_with_sae(layer_activations)
    if sae_features is None or sae_features.ndim != 3:
        raise RuntimeError("SAE encode output must be 3D [batch, seq, n_features].")
    return sae_features.to(dtype=torch.float32)


def _init_topk_state(feature_count: int, top_k: int) -> Dict[str, torch.Tensor]:
    top_scores = torch.full((feature_count, top_k), float("-inf"), dtype=torch.float32, device="cpu")
    top_token_ids = torch.full((feature_count, top_k), -1, dtype=torch.int32, device="cpu")
    activated_token_counts = torch.zeros(feature_count, dtype=torch.int64, device="cpu")
    max_activations = torch.full((feature_count,), float("-inf"), dtype=torch.float32, device="cpu")
    return {
        "top_scores": top_scores,
        "top_token_ids": top_token_ids,
        "activated_token_counts": activated_token_counts,
        "max_activations": max_activations,
    }


def _update_topk_state(
    *,
    state: Dict[str, torch.Tensor],
    selected_acts: torch.Tensor,  # [batch, selected_feature_count], token position only
    batch_token_ids: Sequence[int],
    top_k: int,
    activation_threshold: float,
    feature_chunk_size: int,
) -> None:
    if selected_acts.ndim != 2:
        raise RuntimeError("selected_acts must be 2D [batch, selected_feature_count].")

    active_mask = selected_acts > float(activation_threshold)
    state["activated_token_counts"] += active_mask.sum(dim=0).to("cpu", dtype=torch.int64)

    batch_max = selected_acts.max(dim=0).values
    state["max_activations"] = torch.maximum(state["max_activations"], batch_max.to("cpu", dtype=torch.float32))

    batch_size = selected_acts.shape[0]
    feature_count = selected_acts.shape[1]

    token_ids_tensor = torch.tensor(batch_token_ids, dtype=torch.int32, device=selected_acts.device).view(1, batch_size)

    for start in range(0, feature_count, int(feature_chunk_size)):
        end = min(start + int(feature_chunk_size), feature_count)

        batch_scores = selected_acts[:, start:end].transpose(0, 1)  # [chunk, batch]
        batch_scores = torch.where(
            batch_scores > float(activation_threshold),
            batch_scores,
            torch.full_like(batch_scores, float("-inf")),
        )

        curr_scores = state["top_scores"][start:end].to(selected_acts.device)
        curr_ids = state["top_token_ids"][start:end].to(selected_acts.device)
        batch_ids = token_ids_tensor.expand(end - start, -1)

        candidate_scores = torch.cat([curr_scores, batch_scores], dim=1)
        candidate_ids = torch.cat([curr_ids, batch_ids], dim=1)

        new_scores, new_idx = torch.topk(candidate_scores, k=top_k, dim=1)
        new_ids = torch.gather(candidate_ids, dim=1, index=new_idx)

        state["top_scores"][start:end] = new_scores.to("cpu", dtype=torch.float32)
        state["top_token_ids"][start:end] = new_ids.to("cpu", dtype=torch.int32)


def _append_all_activated_records(
    *,
    writer,
    selected_acts: torch.Tensor,  # [batch, selected_feature_count]
    selected_feature_ids: Sequence[int],
    batch_token_ids: Sequence[int],
    activation_threshold: float,
) -> int:
    active_indices = torch.nonzero(selected_acts > float(activation_threshold), as_tuple=False)
    if active_indices.numel() == 0:
        return 0

    acts_cpu = selected_acts.detach().to("cpu")
    count = 0
    for idx in active_indices.to("cpu").tolist():
        row, col = int(idx[0]), int(idx[1])
        writer.write(
            json.dumps(
                {
                    "feature_id": int(selected_feature_ids[col]),
                    "token_id": int(batch_token_ids[row]),
                    "activation": float(acts_cpu[row, col].item()),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        count += 1
    return count


def _build_feature_payloads(
    *,
    layer_id: int,
    tokenizer,
    selected_feature_ids: Sequence[int],
    state: Dict[str, torch.Tensor],
    top_k: int,
    evaluated_token_count: int,
    activation_threshold: float,
    prompt_template: str,
    prompt_id: str,
    target_token_position: int,
) -> List[Dict[str, object]]:
    top_scores = state["top_scores"]
    top_token_ids = state["top_token_ids"]
    activated_token_counts = state["activated_token_counts"]
    max_activations = state["max_activations"]

    valid_token_ids: set[int] = set()
    for row_scores, row_ids in zip(top_scores.tolist(), top_token_ids.tolist()):
        for score, token_id in zip(row_scores, row_ids):
            if float(score) != float("-inf") and int(token_id) >= 0:
                valid_token_ids.add(int(token_id))

    token_text_map: Dict[int, str] = {}
    if valid_token_ids:
        sorted_ids = sorted(valid_token_ids)
        token_texts = tokenizer.convert_ids_to_tokens(sorted_ids)
        token_text_map = {int(tok_id): str(tok_text) for tok_id, tok_text in zip(sorted_ids, token_texts)}

    payloads: List[Dict[str, object]] = []
    for row_idx, feature_id in enumerate(selected_feature_ids):
        entries: List[Dict[str, object]] = []
        row_scores = top_scores[row_idx].tolist()
        row_token_ids = top_token_ids[row_idx].tolist()

        for rank, (score, token_id) in enumerate(zip(row_scores, row_token_ids), start=1):
            if float(score) == float("-inf") or int(token_id) < 0:
                continue
            tok_id = int(token_id)
            entries.append(
                {
                    "rank": int(rank),
                    "token_id": tok_id,
                    "token_text": token_text_map.get(tok_id, ""),
                    "activation": float(score),
                }
            )

        payload = {
            "layer_id": int(layer_id),
            "feature_id": int(feature_id),
            "source": "bos_token",
            "prompt_id": str(prompt_id),
            "prompt_template": str(prompt_template),
            "scoring_position": "template_token",
            "target_token_position": int(target_token_position),
            "top_k": int(top_k),
            "activation_threshold": float(activation_threshold),
            "evaluated_token_count": int(evaluated_token_count),
            "activated_token_count": int(activated_token_counts[row_idx].item()),
            "max_activation_seen": float(max_activations[row_idx].item()),
            "top_tokens": entries,
        }
        payloads.append(payload)
    return payloads


def _write_feature_outputs(
    *,
    output_root: Path,
    layer_id: int,
    tokenizer,
    selected_feature_ids: Sequence[int],
    state: Dict[str, torch.Tensor],
    top_k: int,
    evaluated_token_count: int,
    activation_threshold: float,
    prompt_template: str,
    prompt_id: str,
    target_token_position: int,
) -> int:
    payloads = _build_feature_payloads(
        layer_id=layer_id,
        tokenizer=tokenizer,
        selected_feature_ids=selected_feature_ids,
        state=state,
        top_k=top_k,
        evaluated_token_count=evaluated_token_count,
        activation_threshold=activation_threshold,
        prompt_template=prompt_template,
        prompt_id=prompt_id,
        target_token_position=target_token_position,
    )
    for payload in payloads:
        feature_dir = (
            output_root
            / f"layer-{int(layer_id)}"
            / f"feature-{int(payload['feature_id'])}"
            / "bos_token"
            / str(prompt_id)
        )
        feature_dir.mkdir(parents=True, exist_ok=True)
        (feature_dir / "top_tokens.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return len(payloads)


@torch.no_grad()
def scan_tokens_with_module(
    *,
    module,
    layer_id: int,
    requested_feature_ids: Optional[Sequence[int]],
    prompt_template: str,
    prompt_id: str,
    manual_token_texts: Sequence[str],
    scan_full_vocab: bool,
    random_sample_size: int,
    include_special_tokens: bool,
    seed: int,
    batch_size: int,
    top_k: int,
    activation_threshold: float,
    feature_chunk_size: int,
    all_activated_writer=None,
    show_progress: bool = False,
) -> Dict[str, object]:
    if module.model is None or module.tokenizer is None:
        raise RuntimeError("Failed to initialize model/tokenizer.")
    if int(batch_size) <= 0 or int(top_k) <= 0 or int(feature_chunk_size) <= 0:
        raise ValueError("batch_size, top_k, and feature_chunk_size must be positive")
    if str(prompt_template).count("{token}") > 1:
        raise ValueError("prompt_template may contain at most one {token} placeholder")
    if "<bos>" not in str(prompt_template):
        raise ValueError("prompt_template must contain <bos>")

    tokenizer = module.tokenizer
    bos_token_id = _resolve_bos_token_id(tokenizer)
    prefix_ids, suffix_ids, target_token_position = _resolve_prompt_template(
        tokenizer=tokenizer,
        template=str(prompt_template),
        bos_token_id=int(bos_token_id),
    )
    manual_token_ids, missing_manual_tokens = _resolve_manual_token_ids(
        tokenizer,
        list(manual_token_texts),
    )
    token_ids, token_stats = _build_token_scan_list(
        tokenizer=tokenizer,
        manual_token_ids=manual_token_ids,
        scan_full_vocab=bool(scan_full_vocab),
        random_sample_size=int(random_sample_size),
        include_special_tokens=bool(include_special_tokens),
        seed=int(seed),
    )
    if not token_ids:
        raise RuntimeError("No valid token ids to scan.")

    requested = None if requested_feature_ids is None else [int(value) for value in requested_feature_ids]
    if requested is not None and not requested:
        raise ValueError("requested_feature_ids must not be empty")

    selected_feature_ids: Optional[List[int]] = None
    selected_feature_tensor: Optional[torch.Tensor] = None
    state: Optional[Dict[str, torch.Tensor]] = None
    n_features_total: Optional[int] = None
    all_activated_count = 0
    total_batches = (len(token_ids) + int(batch_size) - 1) // int(batch_size)

    for start in tqdm(
        range(0, len(token_ids), int(batch_size)),
        total=total_batches,
        desc="Scanning prompt-template token batches",
        unit="batch",
        disable=not bool(show_progress),
    ):
        batch_token_ids = token_ids[start : start + int(batch_size)]
        input_ids, attention_mask = _build_prompt_batch(
            batch_token_ids=batch_token_ids,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            device=module.device,
        )
        sae_features = _get_sae_features(module, input_ids=input_ids, attention_mask=attention_mask)
        if sae_features.shape[1] <= int(target_token_position):
            raise RuntimeError("Prompt template target token position is outside SAE sequence length.")
        token_position_acts = sae_features[:, int(target_token_position), :]
        if token_position_acts.ndim != 2:
            raise RuntimeError("Token-position activations must be 2D [batch, n_features].")

        if selected_feature_ids is None:
            n_features_total = int(token_position_acts.shape[-1])
            selected_feature_ids = list(range(n_features_total)) if requested is None else list(requested)
            bad = [feature_id for feature_id in selected_feature_ids if feature_id < 0 or feature_id >= n_features_total]
            if bad:
                raise ValueError(
                    f"Requested feature ids out of range for n_features={n_features_total}: {bad[:10]}"
                )
            selected_feature_tensor = torch.tensor(
                selected_feature_ids,
                dtype=torch.long,
                device=token_position_acts.device,
            )
            state = _init_topk_state(feature_count=len(selected_feature_ids), top_k=int(top_k))

        assert selected_feature_tensor is not None
        assert selected_feature_ids is not None
        assert state is not None
        selected_acts = token_position_acts.index_select(dim=1, index=selected_feature_tensor)
        _update_topk_state(
            state=state,
            selected_acts=selected_acts,
            batch_token_ids=batch_token_ids,
            top_k=int(top_k),
            activation_threshold=float(activation_threshold),
            feature_chunk_size=int(feature_chunk_size),
        )
        if all_activated_writer is not None:
            all_activated_count += _append_all_activated_records(
                writer=all_activated_writer,
                selected_acts=selected_acts,
                selected_feature_ids=selected_feature_ids,
                batch_token_ids=batch_token_ids,
                activation_threshold=float(activation_threshold),
            )

    if state is None or selected_feature_ids is None or n_features_total is None:
        raise RuntimeError("No batches were processed.")

    prompt = {
        "prompt_id": str(prompt_id),
        "prompt_template": str(prompt_template),
        "prefix_token_ids": [int(value) for value in prefix_ids],
        "suffix_token_ids": [int(value) for value in suffix_ids],
        "target_token_position": int(target_token_position),
    }
    summary = {
        "layer_id": int(layer_id),
        "bos_token_id": int(bos_token_id),
        "prompt_id": str(prompt_id),
        "prompt_template": str(prompt_template),
        "prompt_template_prefix_token_count": len(prefix_ids),
        "prompt_template_suffix_token_count": len(suffix_ids),
        "scoring_position": "template_token",
        "target_token_position": int(target_token_position),
        "scan_all_features": requested is None,
        "selected_feature_count": len(selected_feature_ids),
        "n_features_total": int(n_features_total),
        "top_k": int(top_k),
        "activation_threshold": float(activation_threshold),
        "token_scan": {
            **token_stats,
            "scan_full_vocab": bool(scan_full_vocab),
            "include_special_tokens": bool(include_special_tokens),
            "seed": int(seed),
        },
        "manual_tokens": {
            "provided_count": len(manual_token_texts),
            "resolved_count": len(manual_token_ids),
            "missing_count": len(missing_manual_tokens),
            "missing_examples": missing_manual_tokens[:20],
        },
        "all_activated_record_count": int(all_activated_count),
    }
    feature_payloads = _build_feature_payloads(
        layer_id=int(layer_id),
        tokenizer=tokenizer,
        selected_feature_ids=selected_feature_ids,
        state=state,
        top_k=int(top_k),
        evaluated_token_count=len(token_ids),
        activation_threshold=float(activation_threshold),
        prompt_template=str(prompt_template),
        prompt_id=str(prompt_id),
        target_token_position=int(target_token_position),
    )
    return {"prompt": prompt, "summary": summary, "feature_payloads": feature_payloads}


@torch.no_grad()
def run_scan(args: argparse.Namespace) -> Dict[str, object]:
    _validate_args(args)
    sae_path, resolved_average_l0 = _resolve_sae_path(args)
    module = ModelWithSAEModule(
        llm_name=str(args.model_checkpoint_path),
        sae_path=str(sae_path),
        sae_layer=int(args.layer_id),
        feature_index=-1,
        device=str(args.device),
    )
    if module.model is None or module.tokenizer is None:
        raise RuntimeError("Failed to initialize model/tokenizer.")
    manual_texts = _load_manual_token_texts(args.manual_token_texts_file) if args.manual_token_texts_file else []
    requested_feature_ids = None if bool(args.all_features) else _load_feature_ids(str(args.feature_ids_file))
    output_root = Path(str(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    prompt_id = str(args.prompt_id).strip()
    layer_prompt_dir = output_root / f"layer-{int(args.layer_id)}" / "bos_token" / prompt_id
    layer_prompt_dir.mkdir(parents=True, exist_ok=True)
    all_activated_writer = None
    if bool(args.save_all_activated):
        all_activated_path = layer_prompt_dir / "all_activated.jsonl"
        all_activated_writer = all_activated_path.open("w", encoding="utf-8")
    try:
        core = scan_tokens_with_module(
            module=module,
            layer_id=int(args.layer_id),
            requested_feature_ids=requested_feature_ids,
            prompt_template=str(args.prompt_template),
            prompt_id=prompt_id,
            manual_token_texts=manual_texts,
            scan_full_vocab=bool(args.scan_full_vocab),
            random_sample_size=int(args.random_sample_size),
            include_special_tokens=bool(args.include_special_tokens),
            seed=int(args.seed),
            batch_size=int(args.batch_size),
            top_k=int(args.top_k),
            activation_threshold=float(args.activation_threshold),
            feature_chunk_size=int(args.feature_chunk_size),
            all_activated_writer=all_activated_writer,
            show_progress=True,
        )
    finally:
        if all_activated_writer is not None:
            all_activated_writer.close()
    prompt = dict(core["prompt"])
    feature_payloads = list(core["feature_payloads"])
    summary = {
        "model_checkpoint_path": str(args.model_checkpoint_path),
        "sae_path": str(sae_path),
        "resolved_average_l0": resolved_average_l0,
        **dict(core["summary"]),
        "outputs": {
            "output_root": str(output_root.resolve()),
            "prompt_path": str((layer_prompt_dir / "prompt.json").resolve()),
            "written_feature_count": len(feature_payloads),
            "save_all_activated": bool(args.save_all_activated),
            "all_activated_record_count": int(core["summary"]["all_activated_record_count"]),
        },
    }
    (layer_prompt_dir / "prompt.json").write_text(
        json.dumps(prompt, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (layer_prompt_dir / "scan_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for payload in feature_payloads:
        feature_dir = (
            output_root
            / f"layer-{int(args.layer_id)}"
            / f"feature-{int(payload['feature_id'])}"
            / "bos_token"
            / prompt_id
        )
        feature_dir.mkdir(parents=True, exist_ok=True)
        (feature_dir / "top_tokens.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (feature_dir / "prompt.json").write_text(
            json.dumps(prompt, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        feature_summary = dict(summary)
        feature_summary["feature_id"] = int(payload["feature_id"])
        (feature_dir / "scan_summary.json").write_text(
            json.dumps(feature_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return summary


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)

    start_time = time.time()
    summary = run_scan(args)
    elapsed = time.time() - start_time

    print(json.dumps({"status": "ok", "elapsed_seconds": round(elapsed, 2), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
