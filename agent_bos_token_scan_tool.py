from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from inference_client import call_bos_token_scan
from input_bos_token_scan import DEFAULT_CANONICAL_MAP_PATH, run_scan


CODE_DIR = Path(__file__).resolve().parent


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (CODE_DIR / path)


def _validate_prompt_id(prompt_id: str) -> str:
    value = str(prompt_id).strip()
    if not value or "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError("prompt_id must be a non-empty single directory name")
    return value


def _dedupe(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _feature_prompt_dir(output_root: Path, layer_id: str, feature_id: str, prompt_id: str) -> Path:
    return (
        output_root
        / f"layer-{int(layer_id)}"
        / f"feature-{int(feature_id)}"
        / "bos_token"
        / prompt_id
    )


def _unique_prompt_id(output_root: Path, layer_id: str, feature_id: str, requested: str) -> str:
    if not _feature_prompt_dir(output_root, layer_id, feature_id, requested).exists():
        return requested
    for index in range(1, 1000):
        candidate = f"{requested}-{index:03d}"
        if not _feature_prompt_dir(output_root, layer_id, feature_id, candidate).exists():
            return candidate
    raise RuntimeError(f"could not allocate a prompt id after 999 collisions: {requested!r}")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_local_scan(
    *,
    layer_id: str,
    feature_id: str,
    model_checkpoint_path: str,
    sae_path: str,
    prompt_template: str,
    prompt_id: str,
    token_texts: Sequence[str],
    scan_full_vocab: bool,
    random_sample_size: int,
    top_k: int,
    batch_size: int,
    activation_threshold: float,
    device: str,
    width: str,
    output_root: Path,
    include_special_tokens: bool,
    seed: int,
    feature_chunk_size: int,
) -> Dict[str, Any]:
    prompt_dir = _feature_prompt_dir(output_root, layer_id, feature_id, prompt_id)
    prompt_dir.mkdir(parents=True, exist_ok=False)
    feature_ids_file = prompt_dir / "feature_ids.txt"
    feature_ids_file.write_text(f"{int(feature_id)}\n", encoding="utf-8")
    candidate_file: Optional[Path] = None
    if token_texts:
        candidate_file = prompt_dir / "candidate_token_texts.txt"
        candidate_file.write_text("\n".join(token_texts) + "\n", encoding="utf-8")
    args = argparse.Namespace(
        model_checkpoint_path=str(model_checkpoint_path),
        layer_id=int(layer_id),
        sae_path=str(sae_path),
        sae_release="gemma-scope-2b-pt-res",
        width=str(width),
        sae_average_l0=None,
        sae_canonical_map=str(_resolve_path(str(DEFAULT_CANONICAL_MAP_PATH))),
        feature_ids_file=str(feature_ids_file),
        all_features=False,
        manual_token_texts_file=str(candidate_file) if candidate_file else None,
        random_sample_size=int(random_sample_size),
        scan_full_vocab=bool(scan_full_vocab),
        include_special_tokens=bool(include_special_tokens),
        seed=int(seed),
        prompt_template=str(prompt_template),
        prompt_id=str(prompt_id),
        batch_size=int(batch_size),
        top_k=int(top_k),
        activation_threshold=float(activation_threshold),
        feature_chunk_size=int(feature_chunk_size),
        save_all_activated=False,
        output_root=str(output_root),
        device=str(device),
    )
    run_scan(args)
    return {
        "prompt": json.loads((prompt_dir / "prompt.json").read_text(encoding="utf-8")),
        "summary": json.loads((prompt_dir / "scan_summary.json").read_text(encoding="utf-8")),
        "feature_payloads": [json.loads((prompt_dir / "top_tokens.json").read_text(encoding="utf-8"))],
    }


def run_agent_bos_token_scan(
    *,
    layer_id: str,
    feature_id: str,
    model_checkpoint_path: str,
    sae_path: str,
    prompt_template: str,
    prompt_id: str,
    candidate_token_texts: Optional[Sequence[str]] = None,
    scan_full_vocab: bool = False,
    random_sample_size: int = 0,
    top_k: int = 50,
    batch_size: int = 64,
    activation_threshold: float = 0.0,
    device: str = "cpu",
    width: str = "16k",
    output_root: str = "initial_observation",
    include_special_tokens: bool = False,
    seed: int = 42,
    feature_chunk_size: int = 4096,
    inference_server_url: str = "http://127.0.0.1:8008",
    inference_timeout_sec: float = 600.0,
    no_inference_server: bool = False,
) -> Dict[str, Any]:
    requested_prompt_id = _validate_prompt_id(prompt_id)
    token_texts = _dedupe(candidate_token_texts or [])
    if not scan_full_vocab and int(random_sample_size) <= 0 and not token_texts:
        raise ValueError("provide candidate_token_texts, random_sample_size > 0, or scan_full_vocab=true")
    resolved_root = _resolve_path(output_root)
    resolved_root.mkdir(parents=True, exist_ok=True)
    resolved_prompt_id = _unique_prompt_id(
        resolved_root,
        str(layer_id),
        str(feature_id),
        requested_prompt_id,
    )
    prompt_dir = _feature_prompt_dir(resolved_root, layer_id, feature_id, resolved_prompt_id)

    if no_inference_server:
        response = _run_local_scan(
            layer_id=layer_id,
            feature_id=feature_id,
            model_checkpoint_path=model_checkpoint_path,
            sae_path=sae_path,
            prompt_template=prompt_template,
            prompt_id=resolved_prompt_id,
            token_texts=token_texts,
            scan_full_vocab=scan_full_vocab,
            random_sample_size=random_sample_size,
            top_k=top_k,
            batch_size=batch_size,
            activation_threshold=activation_threshold,
            device=device,
            width=width,
            output_root=resolved_root,
            include_special_tokens=include_special_tokens,
            seed=seed,
            feature_chunk_size=feature_chunk_size,
        )
    else:
        response = call_bos_token_scan(
            server_url=str(inference_server_url),
            timeout_sec=float(inference_timeout_sec),
            layer_id=int(layer_id),
            feature_ids=[int(feature_id)],
            prompt_template=str(prompt_template),
            prompt_id=resolved_prompt_id,
            candidate_token_texts=token_texts,
            scan_full_vocab=bool(scan_full_vocab),
            random_sample_size=int(random_sample_size),
            top_k=int(top_k),
            batch_size=int(batch_size),
            activation_threshold=float(activation_threshold),
            include_special_tokens=bool(include_special_tokens),
            seed=int(seed),
            feature_chunk_size=int(feature_chunk_size),
        )
        payloads = response.get("feature_payloads") or []
        if len(payloads) != 1 or int(payloads[0].get("feature_id", -1)) != int(feature_id):
            raise RuntimeError("inference server did not return exactly the requested feature")
        prompt_dir.mkdir(parents=True, exist_ok=False)
        _write_json(prompt_dir / "top_tokens.json", payloads[0])
        _write_json(prompt_dir / "prompt.json", dict(response.get("prompt") or {}))
        _write_json(prompt_dir / "scan_summary.json", dict(response.get("summary") or {}))
        (prompt_dir / "feature_ids.txt").write_text(f"{int(feature_id)}\n", encoding="utf-8")
        if token_texts:
            (prompt_dir / "candidate_token_texts.txt").write_text(
                "\n".join(token_texts) + "\n",
                encoding="utf-8",
            )

    feature_payload = dict((response.get("feature_payloads") or [{}])[0])
    summary = dict(response.get("summary") or {})
    top_tokens = list(feature_payload.get("top_tokens") or [])
    result = {
        "status": "ok",
        "layer_id": str(layer_id),
        "feature_id": str(feature_id),
        "requested_prompt_id": requested_prompt_id,
        "prompt_id": resolved_prompt_id,
        "prompt_id_collision_renamed": requested_prompt_id != resolved_prompt_id,
        "prompt_template": str(prompt_template),
        "output_path": str(prompt_dir / "top_tokens.json"),
        "missing_manual_tokens": (summary.get("manual_tokens") or {}).get("missing_examples", []),
        "candidate_token_count": len(token_texts),
        "scan_full_vocab": bool(scan_full_vocab),
        "random_sample_size": int(random_sample_size),
        "top_tokens": top_tokens[:20],
        "scan_summary": summary,
    }
    _write_json(prompt_dir / "tool_summary.json", result)
    return result
