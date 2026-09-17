from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

DEFAULT_INFERENCE_SERVER_URL = "http://127.0.0.1:8008"
DEFAULT_INFERENCE_TIMEOUT_SEC = 600.0


class InferenceServerError(RuntimeError):
    pass


def call_bos_token_scan(
    *,
    server_url: str,
    timeout_sec: float,
    layer_id: int,
    feature_ids: List[int],
    prompt_template: str,
    prompt_id: str,
    candidate_token_texts: List[str],
    scan_full_vocab: bool,
    random_sample_size: int,
    top_k: int,
    batch_size: int,
    activation_threshold: float,
    include_special_tokens: bool,
    seed: int,
    feature_chunk_size: int,
) -> Dict[str, Any]:
    return _post_json(
        server_url=server_url,
        endpoint="/bos-token-scan",
        timeout_sec=timeout_sec,
        payload={
            "layer_id": int(layer_id),
            "feature_ids": [int(feature_id) for feature_id in feature_ids],
            "prompt_template": str(prompt_template),
            "prompt_id": str(prompt_id),
            "candidate_token_texts": [str(value) for value in candidate_token_texts],
            "scan_full_vocab": bool(scan_full_vocab),
            "random_sample_size": int(random_sample_size),
            "top_k": int(top_k),
            "batch_size": int(batch_size),
            "activation_threshold": float(activation_threshold),
            "include_special_tokens": bool(include_special_tokens),
            "seed": int(seed),
            "feature_chunk_size": int(feature_chunk_size),
        },
    )


def call_activation_traces(
    *,
    server_url: str,
    timeout_sec: float,
    layer_id: int,
    feature_id: int,
    texts: List[str],
) -> Dict[str, Any]:
    return _post_json(
        server_url=server_url,
        endpoint="/activation-traces",
        timeout_sec=timeout_sec,
        payload={
            "layer_id": int(layer_id),
            "feature_id": int(feature_id),
            "texts": [str(text) for text in texts],
        },
    )


def call_score_input_experiments(
    *,
    server_url: str,
    timeout_sec: float,
    layer_id: int,
    feature_id: int,
    input_side_experiments: List[Dict[str, Any]],
    non_zero_threshold: float,
    max_activation_scale: float,
) -> Dict[str, Any]:
    return _post_json(
        server_url=server_url,
        endpoint="/score-input-experiments",
        timeout_sec=timeout_sec,
        payload={
            "layer_id": int(layer_id),
            "feature_id": int(feature_id),
            "input_side_experiments": input_side_experiments,
            "non_zero_threshold": float(non_zero_threshold),
            "max_activation_scale": float(max_activation_scale),
        },
    )


def call_run_intervention(
    *,
    server_url: str,
    timeout_sec: float,
    layer_id: int,
    feature_id: int,
    input_side_experiments: List[Dict[str, Any]],
    prompts: List[str],
    top_k: int,
    max_steering_prompts: int,
    intervention_scope: str,
    max_activation_scale: float,
    last_token_scale: float,
    custom_steering_prompts: Optional[List[str]],
) -> Dict[str, Any]:
    return _post_json(
        server_url=server_url,
        endpoint="/run-intervention",
        timeout_sec=timeout_sec,
        payload={
            "layer_id": int(layer_id),
            "feature_id": int(feature_id),
            "input_side_experiments": input_side_experiments,
            "prompts": list(prompts),
            "top_k": int(top_k),
            "max_steering_prompts": int(max_steering_prompts),
            "intervention_scope": str(intervention_scope),
            "max_activation_scale": float(max_activation_scale),
            "last_token_scale": float(last_token_scale),
            "custom_steering_prompts": custom_steering_prompts,
        },
    )



def call_output_centric_observation(
    *,
    server_url: str,
    timeout_sec: float,
    layer_id: int,
    feature_id: int,
    token_id_batches: List[List[int]],
    projection_top_k: int = 50,
    token_change_top_k: int = 10,
    clamp_value: float = 10.0,
) -> Dict[str, Any]:
    return _post_json(
        server_url=server_url,
        endpoint="/output-centric-observation",
        timeout_sec=timeout_sec,
        payload={
            "layer_id": int(layer_id),
            "feature_id": int(feature_id),
            "token_id_batches": [[int(token) for token in row] for row in token_id_batches],
            "projection_top_k": int(projection_top_k),
            "token_change_top_k": int(token_change_top_k),
            "clamp_value": float(clamp_value),
        },
    )
def _post_json(
    *,
    server_url: str,
    endpoint: str,
    timeout_sec: float,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    base_url = str(server_url or "").rstrip("/")
    if not base_url:
        raise InferenceServerError("Inference server URL is empty. Pass --no-inference-server to use local loading.")
    url = f"{base_url}{endpoint}"
    try:
        response = requests.post(url, json=payload, timeout=float(timeout_sec))
    except requests.exceptions.Timeout as exc:
        raise InferenceServerError(f"Inference server timed out after {timeout_sec}s: {url}") from exc
    except requests.exceptions.ConnectionError as exc:
        raise InferenceServerError(
            f"Inference server unavailable at {base_url}. "
            "Start inference_server.py or pass --no-inference-server to use local loading."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise InferenceServerError(f"Inference server request failed for {url}: {exc}") from exc

    if response.status_code >= 400:
        detail = _response_detail(response)
        raise InferenceServerError(f"Inference server returned HTTP {response.status_code} for {url}: {detail}")

    try:
        data = response.json()
    except ValueError as exc:
        raise InferenceServerError(f"Inference server returned non-JSON response for {url}: {response.text[:500]}") from exc
    if not isinstance(data, dict):
        raise InferenceServerError(f"Inference server returned {type(data).__name__}, expected JSON object.")
    return data


def _response_detail(response: requests.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(data, dict):
        detail = data.get("detail")
        if detail is not None:
            return str(detail)
    return str(data)[:500]
