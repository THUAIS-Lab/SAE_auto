from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from function import build_round_dir, normalize_round_id

BASE_URL = "https://www.neuronpedia.org"


def fetch_feature_json(
    model_id: str,
    source: str,
    feature_id: str,
    api_key: Optional[str] = None,
    timeout: int = 30,
    retry_count: int = 3,
    retry_sleep_seconds: float = 3.0,
) -> Dict[str, Any]:
    """Fetch feature payload from Neuronpedia API."""
    url = f"{BASE_URL}/api/feature/{model_id}/{source}/{feature_id}"
    token = api_key or os.getenv("NEURONPEDIA_API_KEY")
    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last_error: Optional[Exception] = None
    total_attempts = max(1, int(retry_count))
    for attempt_idx in range(total_attempts):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            # backup_dir = Path("./neuronpedia_return")
            # backup_dir.mkdir(parents=True, exist_ok=True)
            # backup_path = backup_dir / f"feature_{model_id}_{source}_{feature_id}.json"
            # with backup_path.open("w", encoding="utf-8") as f:
            #     json.dump(payload, f, indent=2, ensure_ascii=False)
            return payload
        except requests.exceptions.HTTPError as exc:
            last_error = exc
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            retryable = status_code == 429 or (isinstance(status_code, int) and status_code >= 500)
            is_last_attempt = attempt_idx == total_attempts - 1
            if not retryable or is_last_attempt:
                break
            print(
                f"Neuronpedia HTTP {status_code} for feature {feature_id} "
                f"(attempt {attempt_idx + 1}/{total_attempts}). "
                f"Sleeping {retry_sleep_seconds} seconds before retry."
            )
            time.sleep(retry_sleep_seconds)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc
            is_last_attempt = attempt_idx == total_attempts - 1
            if is_last_attempt:
                break
            print(
                f"Neuronpedia request failed for feature {feature_id} "
                f"(attempt {attempt_idx + 1}/{total_attempts}): {exc}. "
                f"Sleeping {retry_sleep_seconds} seconds before retry."
            )
            time.sleep(retry_sleep_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to fetch Neuronpedia feature payload for feature {feature_id}.")


def _to_explanation_strings(obj: Any) -> List[str]:
    if obj is None:
        return []
    if isinstance(obj, str):
        text = obj.strip()
        return [text] if text else []
    if isinstance(obj, list):
        out: List[str] = []
        for item in obj:
            out.extend(_to_explanation_strings(item))
        return out
    if isinstance(obj, dict):
        out: List[str] = []
        preferred_keys = ("description", "explanation", "text", "content")
        hit_preferred = False
        for key in preferred_keys:
            if key in obj:
                hit_preferred = True
                out.extend(_to_explanation_strings(obj[key]))
        if not hit_preferred:
            for value in obj.values():
                out.extend(_to_explanation_strings(value))
        return out
    return []


def extract_explanations(
    feature_payload: Dict[str, Any],
    limit: int = 1,
) -> List[str]:
    """
    Extract explanation strings from a Neuronpedia feature payload.

    Handles both `explanation` and `explanations` fields and supports string/list/dict forms.
    """
    if limit <= 0:
        raise ValueError("limit must be a positive integer.")

    raw_candidates: List[str] = []
    for key in ("explanation", "explanations"):
        if key in feature_payload:
            raw_candidates.extend(_to_explanation_strings(feature_payload[key]))

    deduped: List[str] = []
    seen = set()
    for candidate in raw_candidates:
        normalized = " ".join(candidate.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)

    return deduped[:limit]


def _safe_max_token(activation: Dict[str, Any]) -> str:
    tokens = activation.get("tokens")
    max_idx = activation.get("maxValueTokenIndex")
    if not isinstance(tokens, list) or not isinstance(max_idx, int):
        return ""
    if max_idx < 0 or max_idx >= len(tokens):
        return ""
    token = tokens[max_idx]
    return token if isinstance(token, str) else str(token)


def _select_activations_method_1(
    activations: List[Dict[str, Any]],
    m: int,
    n: int,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    total = len(activations)
    first_count = min(m, total)
    selected_indices: List[int] = list(range(first_count))

    # phase 2: token alternation from m onwards
    for idx in range(first_count, total):
        if len(selected_indices) >= first_count + n:
            break
        current_token = _safe_max_token(activations[idx])
        last_token = _safe_max_token(activations[selected_indices[-1]]) if selected_indices else ""
        if current_token != last_token:
            selected_indices.append(idx)

    # fallback fill from m onwards, keep order, skip already selected
    target = first_count + n
    if len(selected_indices) < target:
        for idx in range(first_count, total):
            if len(selected_indices) >= target:
                break
            if idx not in selected_indices:
                selected_indices.append(idx)

    selected = [activations[i] for i in selected_indices]
    return selected, selected_indices


def _select_activations_method_2(
    activations: List[Dict[str, Any]],
    n: int,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    selected_indices: List[int] = []
    for idx, item in enumerate(activations):
        if not selected_indices:
            selected_indices.append(idx)
        else:
            current_token = _safe_max_token(item)
            last_token = _safe_max_token(activations[selected_indices[-1]])
            if current_token != last_token:
                selected_indices.append(idx)
        if len(selected_indices) >= n:
            break

    selected = [activations[i] for i in selected_indices]
    return selected, selected_indices


def _select_activations_method_3(
    activations: List[Dict[str, Any]],
    m: int,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    count = min(m, len(activations))
    selected_indices = list(range(count))
    selected = [activations[i] for i in selected_indices]
    return selected, selected_indices


def _pair_str_values(strings: Any, values: Any) -> List[Dict[str, Any]]:
    if not isinstance(strings, list) or not isinstance(values, list):
        return []
    size = min(len(strings), len(values))
    out: List[Dict[str, Any]] = []
    for i in range(size):
        out.append({
            "str": strings[i],
            "value": values[i],
        })
    return out


def _build_source(layer_id: str, width: str) -> str:
    return f"{layer_id}-gemmascope-res-{width}"


def convert_to_input_observation(parsed_result: Dict[str, Any], layer_id: str = "", feature_id: str = "") -> Dict[str, Any]:
    """
    Convert parsed observation into compact input-observation format.
    """
    input_side = parsed_result.get("input_side_observation", {})
    output_side = parsed_result.get("output_side_observation", {})
    raw_activations = input_side.get("activations", [])

    activation_examples: List[Dict[str, Any]] = []
    if isinstance(raw_activations, list):
        for item in raw_activations:
            if not isinstance(item, dict):
                continue

            activation_payload = item.get("activation", item)
            if not isinstance(activation_payload, dict):
                activation_payload = {}

            tokens = activation_payload.get("tokens", [])
            values = activation_payload.get("values", [])
            if not isinstance(tokens, list):
                tokens = []
            if not isinstance(values, list):
                values = []

            sentence = "".join(str(token) for token in tokens)

            activation_tokens: List[Dict[str, Any]] = []
            pair_count = min(len(tokens), len(values))
            for idx in range(pair_count):
                val = values[idx]
                if val != 0:
                    activation_tokens.append(
                        {
                            "token": tokens[idx],
                            "value": val,
                        }
                    )

            max_token = item.get("max_token")
            if not isinstance(max_token, str) or not max_token:
                max_token = _safe_max_token(activation_payload)

            activation_examples.append(
                {
                    "sentence": sentence,
                    "activation_tokens": activation_tokens,
                    "maxValue": activation_payload.get("maxValue"),
                    "max_token": max_token,
                }
            )
    converted = {
        "input_side_observation": {
            "selected_count": input_side.get("selected_count", 0),
            "activation_examples": activation_examples,
        },
        "output_side_observation": output_side,
    }
    return converted


def fetch_and_parse_feature_observation(
    model_id: str,
    layer_id: str,
    feature_id: str,
    width: str = "16k",
    selection_method: int = 1,
    m: int = 5,
    n: int = 5,
    api_key: Optional[str] = None,
    timeout: int = 30,
    timestamp: Optional[str] = None,
    round_id: Optional[str] = "round_0",
    logs_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Fetch a feature JSON from Neuronpedia, store raw payload, parse observations, and save parsed result.
    """
    if selection_method not in (1, 2, 3):
        raise ValueError("selection_method must be 1, 2, or 3.")
    if m < 0 or n < 0:
        raise ValueError("m and n must be non-negative integers.")

    payload = fetch_feature_json(
        model_id=model_id,
        source=_build_source(layer_id=layer_id, width=width),
        feature_id=feature_id,
        api_key=api_key,
        timeout=timeout,
    )

    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_round_id = normalize_round_id(round_id, round_index=0)
    base_dir = build_round_dir(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=ts,
        round_id=resolved_round_id,
        round_index=0,
        logs_root=logs_root,
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    raw_path = base_dir / f"layer{layer_id}-feature{feature_id}-neuronpedia-raw.json"
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    activations_raw = payload.get("activations", [])
    activations: List[Dict[str, Any]] = [a for a in activations_raw if isinstance(a, dict)]

    if selection_method == 1:
        selected_activations, selected_indices = _select_activations_method_1(activations, m=m, n=n)
    elif selection_method == 2:
        selected_activations, selected_indices = _select_activations_method_2(activations, n=n)
    else:
        selected_activations, selected_indices = _select_activations_method_3(activations, m=m)

    input_observations: List[Dict[str, Any]] = []
    for i, act in enumerate(selected_activations):
        input_observations.append(
            {
                "selected_rank": i + 1,
                "activation_original_index": selected_indices[i],
                "max_token": _safe_max_token(act),
                "activation": act,
            }
        )

    result: Dict[str, Any] = {
        "layer_id": layer_id,
        "feature_id": feature_id,
        "timestamp": ts,
        "round_id": resolved_round_id,
        "input_side_observation": {
            "selection_method": selection_method,
            "m": m,
            "n": n,
            "selected_count": len(input_observations),
            "total_observations": len(activations),
            "activations": input_observations,
        },
    }

    observation_path = base_dir / f"layer{layer_id}-feature{feature_id}-observation.json"
    with observation_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    converted = convert_to_input_observation(result, layer_id, feature_id)
    observation_input_path = base_dir / f"layer{layer_id}-feature{feature_id}-observation-input.json"
    with observation_input_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    return converted


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and parse Neuronpedia feature observation.")
    parser.add_argument("--model-id", default="gemma-2-2b", help="Neuronpedia model id, e.g. gemma-2-2b")
    parser.add_argument("--layer-id", required=True, help="Layer id for log path/file naming")
    parser.add_argument("--feature-id", required=True, help="Feature id for log path/file naming")
    parser.add_argument("--width", default="16k", help="Width in source string, e.g. 16k")
    parser.add_argument("--selection-method", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--neuronpedia-api-key", default=None)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--timestamp", default=None, help="Optional custom timestamp folder name")
    parser.add_argument("--round-id", default="round_0", help="Round directory under timestamp, e.g. round_0")
    parser.add_argument("--logs-root", default="logs")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    result = fetch_and_parse_feature_observation(
        model_id=args.model_id,
        layer_id=args.layer_id,
        feature_id=args.feature_id,
        width=args.width,
        selection_method=args.selection_method,
        m=args.m,
        n=args.n,
        api_key=args.neuronpedia_api_key,
        timeout=args.timeout,
        timestamp=args.timestamp,
        round_id=args.round_id,
        logs_root=args.logs_root,
    )
    print(
        json.dumps(
            {
                "layer_id": result["layer_id"],
                "feature_id": result["feature_id"],
                "input_selected_count": result["input_side_observation"]["selected_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
