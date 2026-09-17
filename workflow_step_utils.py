from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover
    OpenAI = Any  # type: ignore[misc,assignment]

from function import (
    TokenUsageAccumulator,
    build_default_sae_path,
    build_round_dir,
    call_llm,
    call_llm_stream,
    extract_json_object,
    extract_usage_counts,
    read_api_key,
)
from prompts.chain_judge_prompt import (
    build_chain_judge_system_prompt,
    build_chain_judge_user_prompt,
    build_chain_synthesis_system_prompt,
    build_chain_synthesis_user_prompt,
)
from prompts.experiments_design_prompt import (
    build_boundary_system_prompt,
    build_boundary_user_prompt,
    build_hypothesis_system_prompt,
    build_hypothesis_user_prompt,
    build_system_prompt,
    build_user_prompt,
)
from support_info.llm_api_info import api_key_file as DEFAULT_API_KEY_FILE
from support_info.llm_api_info import base_url as DEFAULT_BASE_URL
from support_info.llm_api_info import model_name as DEFAULT_MODEL_NAME

SideType = Literal["input", "output"]

DEFAULT_MODEL_CHECKPOINT_PATH = os.environ.get(
    "SAE_MODEL_CHECKPOINT_PATH",
    "google/gemma-2-2b",
)
DEFAULT_SAE_ROOT = os.environ.get(
    "SAE_ROOT",
    "gemma-scope-2b-pt-res",
)
DEFAULT_CANONICAL_MAP_PATH = Path(__file__).resolve().parent / "support_info" / "canonical_map.txt"

STEP_OUTPUT_NAMES = {
    1: "initial-observation",
    2: "input-hypotheses",
    3: "input-experiments",
    4: "input-experiment-scores",
    5: "intervention-results",
    6: "output-hypotheses",
    7: "output-hypothesis-scores",
    8: "chain-hypotheses",
    9: "chain-explanation",
}


class LLMParseFailure(ValueError):
    def __init__(self, message: str, llm_calls: Optional[List[Dict[str, Any]]] = None):
        super().__init__(message)
        self.llm_calls = llm_calls or []


def call_llm_with_parse_retry(
    *,
    client: OpenAI,
    model: str,
    messages: Sequence[Dict[str, Any]],
    parser: Callable[[str], Any],
    token_counter: TokenUsageAccumulator,
    expected_format: str,
    max_tokens: int,
    max_retries: int = 2,
    retry_backoff_seconds: float = 1.5,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Call an LLM and retry only when its response cannot be parsed."""
    attempts: List[Dict[str, Any]] = []
    retry_messages = [dict(message) for message in messages]
    for attempt in range(max(0, int(max_retries)) + 1):
        raw_output, usage_obj, debug_info = call_llm(
            client=client,
            model=model,
            messages=retry_messages,
            temperature=0.0,
            max_tokens=max_tokens,
            stream=False,
            response_format_text=True,
            return_debug=True,
        )
        usage = token_counter.add(usage_obj)
        record: Dict[str, Any] = {
            "attempt": attempt + 1,
            "raw_output": raw_output,
            "usage": usage,
            "debug": debug_info,
        }
        try:
            parsed = parser(raw_output)
        except (TypeError, ValueError, OverflowError) as exc:
            record["parse_error"] = f"{type(exc).__name__}: {exc}"
            attempts.append(record)
            if attempt >= max(0, int(max_retries)):
                raise LLMParseFailure(
                    f"LLM response was still invalid after {attempt + 1} attempt(s): {exc}",
                    llm_calls=attempts,
                ) from exc
            retry_messages.extend(
                [
                    {"role": "assistant", "content": raw_output},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response could not be parsed. "
                            f"Error: {exc}. Return only valid JSON in exactly this format: "
                            f"{expected_format}"
                        ),
                    },
                ]
            )
            if retry_backoff_seconds > 0:
                time.sleep(float(retry_backoff_seconds))
            continue
        attempts.append(record)
        return parsed, attempts
    raise AssertionError("unreachable")


def round_id_for_step(step_index: int) -> str:
    """Map step index to its phase directory (input / output / chain)."""
    i = int(step_index)
    if i <= 4:
        return "input"
    if i <= 7:
        return "output"
    return "chain"


def step_round_dir(
    *,
    layer_id: str,
    feature_id: str,
    timestamp: str,
    step_index: int,
    logs_root: Optional[Path] = None,
) -> Path:
    return build_round_dir(
        layer_id=str(layer_id),
        feature_id=str(feature_id),
        timestamp=str(timestamp),
        round_id=round_id_for_step(step_index),
        round_index=step_index,
        logs_root=logs_root,
    )


def step_json_path(
    *,
    layer_id: str,
    feature_id: str,
    timestamp: str,
    step_index: int,
    output_name: Optional[str] = None,
    logs_root: Optional[Path] = None,
) -> Path:
    name = output_name or STEP_OUTPUT_NAMES[int(step_index)]
    return (
        step_round_dir(
            layer_id=str(layer_id),
            feature_id=str(feature_id),
            timestamp=str(timestamp),
            step_index=int(step_index),
            logs_root=logs_root,
        )
        / f"layer{layer_id}-feature{feature_id}-step{step_index}-{name}.json"
    )


def write_step_payload(
    *,
    layer_id: str,
    feature_id: str,
    timestamp: str,
    step_index: int,
    step_name: str,
    outputs: Dict[str, Any],
    parameters: Optional[Dict[str, Any]] = None,
    inputs: Optional[Dict[str, Any]] = None,
    output_name: Optional[str] = None,
    logs_root: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Path]:
    path = step_json_path(
        layer_id=str(layer_id),
        feature_id=str(feature_id),
        timestamp=str(timestamp),
        step_index=int(step_index),
        output_name=output_name,
        logs_root=logs_root,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "atomic_workflow_v1",
        "step_index": int(step_index),
        "step_name": str(step_name),
        "layer_id": str(layer_id),
        "feature_id": str(feature_id),
        "timestamp": str(timestamp),
        "round_id": round_id_for_step(step_index),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "parameters": parameters or {},
        "inputs": inputs or {},
        "outputs": outputs,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload, path


def write_step_failure_payload(
    *,
    layer_id: str,
    feature_id: str,
    timestamp: str,
    step_index: int,
    step_name: str,
    error: BaseException,
    parameters: Optional[Dict[str, Any]] = None,
    inputs: Optional[Dict[str, Any]] = None,
    llm_calls: Optional[List[Dict[str, Any]]] = None,
    partial_outputs: Optional[Dict[str, Any]] = None,
    logs_root: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Path]:
    path = (
        step_round_dir(
            layer_id=str(layer_id),
            feature_id=str(feature_id),
            timestamp=str(timestamp),
            step_index=int(step_index),
            logs_root=logs_root,
        )
        / f"layer{layer_id}-feature{feature_id}-step{step_index}-{STEP_OUTPUT_NAMES[int(step_index)]}-failure.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "atomic_workflow_failure_v1",
        "step_index": int(step_index),
        "step_name": str(step_name),
        "layer_id": str(layer_id),
        "feature_id": str(feature_id),
        "timestamp": str(timestamp),
        "round_id": round_id_for_step(step_index),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "parameters": parameters or {},
        "inputs": inputs or {},
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "llm_calls": llm_calls or [],
        "partial_outputs": partial_outputs or {},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return payload, path


def load_step_payload(
    *,
    layer_id: str,
    feature_id: str,
    timestamp: str,
    step_index: int,
    output_name: Optional[str] = None,
    logs_root: Optional[Path] = None,
) -> Dict[str, Any]:
    path = step_json_path(
        layer_id=str(layer_id),
        feature_id=str(feature_id),
        timestamp=str(timestamp),
        step_index=int(step_index),
        output_name=output_name,
        logs_root=logs_root,
    )
    if not path.exists():
        raise FileNotFoundError(f"Step {step_index} output not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def print_step_result(path: Path, extra: Optional[Dict[str, Any]] = None) -> None:
    payload = {"output_json": str(path)}
    if extra:
        payload.update(extra)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def make_llm_client(*, base_url: str, api_key_file: Optional[str]) -> OpenAI:
    return OpenAI(base_url=str(base_url), api_key=read_api_key(api_key_file))


def resolve_sae_path(
    *,
    layer_id: str,
    width: str,
    sae_path: Optional[str],
    sae_root: str = DEFAULT_SAE_ROOT,
    sae_average_l0: Optional[str] = None,
    sae_canonical_map: str = str(DEFAULT_CANONICAL_MAP_PATH),
    sae_release: str = "gemma-scope-2b-pt-res",
    use_sae_lens_uri: bool = False,
) -> str:
    if sae_path:
        return str(sae_path)
    if use_sae_lens_uri:
        uri, _ = build_default_sae_path(
            layer_id=str(layer_id),
            width=str(width),
            release=str(sae_release),
            average_l0=sae_average_l0,
            canonical_map_path=str(sae_canonical_map),
        )
        return uri

    average_l0 = sae_average_l0
    if not average_l0:
        _, average_l0 = build_default_sae_path(
            layer_id=str(layer_id),
            width=str(width),
            release=str(sae_release),
            average_l0=None,
            canonical_map_path=str(sae_canonical_map),
        )
    return str(Path(sae_root) / f"layer_{layer_id}" / f"width_{width}" / f"average_l0_{average_l0}")


def build_layer_sae_paths(
    *,
    layer_ids: Sequence[int],
    width: str,
    sae_root: str = DEFAULT_SAE_ROOT,
    sae_canonical_map: str = str(DEFAULT_CANONICAL_MAP_PATH),
) -> Dict[int, str]:
    """Build canonical local SAE paths for a set of layers from one root."""
    return {
        int(layer_id): resolve_sae_path(
            layer_id=str(layer_id),
            width=str(width),
            sae_path=None,
            sae_root=str(sae_root),
            sae_canonical_map=str(sae_canonical_map),
        )
        for layer_id in layer_ids
    }


def load_model_with_sae(
    *,
    model_checkpoint_path: str,
    sae_path: str,
    layer_id: str,
    feature_id: str,
    device: str,
) -> Any:
    from model_with_sae import ModelWithSAEModule

    return ModelWithSAEModule(
        llm_name=str(model_checkpoint_path),
        sae_path=str(sae_path),
        sae_layer=int(layer_id),
        feature_index=int(feature_id),
        device=str(device),
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_token(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip().lower()
    return cleaned.replace("\u2581", " ").replace("\u0120", " ")


def normalize_display_token(text: str) -> str:
    cleaned = str(text).replace("\u2581", " ").replace("\u0120", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _normalize_sentence(text: str) -> str:
    stripped = str(text).strip().strip('"').strip("'")
    stripped = re.sub(r"^\d+[\).\s-]+", "", stripped)
    return stripped


def _normalize_hypothesis(text: str) -> str:
    stripped = str(text).strip().strip('"').strip("'")
    return " ".join(stripped.split())


def _extract_named_json_array_block(raw_output: str, key: str) -> Optional[str]:
    match = re.search(rf'"{re.escape(key)}"\s*:', raw_output)
    if not match:
        return None

    array_start = raw_output.find("[", match.end())
    if array_start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(array_start, len(raw_output)):
        ch = raw_output[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "[":
            depth += 1
            continue
        if ch == "]":
            depth -= 1
            if depth == 0:
                return raw_output[array_start : idx + 1]
            continue
    return None


def _extract_named_json_array_block_loose(raw_output: str, key: str) -> Optional[str]:
    match = re.search(rf'["\']?{re.escape(key)}["\']?\s*:', raw_output)
    if not match:
        return None
    array_start = raw_output.find("[", match.end())
    if array_start < 0:
        return None
    array_end = raw_output.find("]", array_start + 1)
    if array_end < 0:
        return None
    return raw_output[array_start : array_end + 1]


def _parse_string_list_from_jsonish_array(array_text: str) -> List[str]:
    text = array_text.strip()
    if not text.startswith("[") or not text.endswith("]"):
        return []
    body = text[1:-1]
    results: List[str] = []
    i = 0
    n = len(body)
    while i < n:
        while i < n and (body[i].isspace() or body[i] == ","):
            i += 1
        if i >= n:
            break
        if body[i] not in {'"', "'"}:
            start = i
            while i < n and body[i] != ",":
                i += 1
            normalized = _normalize_hypothesis(body[start:i])
            if normalized:
                results.append(normalized)
            continue
        quote = body[i]
        i += 1
        chars: List[str] = []
        while i < n:
            ch = body[i]
            if ch == quote:
                i += 1
                break
            if ch != "\\":
                chars.append(ch)
                i += 1
                continue
            if i + 1 >= n:
                i += 1
                break
            nxt = body[i + 1]
            if nxt == "u" and i + 5 < n and re.fullmatch(r"[0-9a-fA-F]{4}", body[i + 2 : i + 6]):
                chars.append(chr(int(body[i + 2 : i + 6], 16)))
                i += 6
                continue
            escape_map = {
                '"': '"',
                "\\": "\\",
                "/": "/",
                "b": "\b",
                "f": "\f",
                "n": "\n",
                "r": "\r",
                "t": "\t",
            }
            chars.append(escape_map.get(nxt, nxt))
            i += 2
        normalized = _normalize_hypothesis("".join(chars))
        if normalized:
            results.append(normalized)
    return results


def parse_hypothesis_list(raw_output: str, expected_count: int) -> List[str]:
    parsed = extract_json_object(raw_output)
    hypotheses: List[str] = []
    if isinstance(parsed, dict):
        candidate = parsed.get("hypotheses")
        if isinstance(candidate, list):
            hypotheses = [
                _normalize_hypothesis(item) for item in candidate if _normalize_hypothesis(item)
            ]

    if not hypotheses:
        array_block = _extract_named_json_array_block(raw_output, key="hypotheses")
        if not array_block:
            array_block = _extract_named_json_array_block_loose(raw_output, key="hypotheses")
        if array_block:
            hypotheses = _parse_string_list_from_jsonish_array(array_block)

    if not hypotheses:
        parsed_lines: List[str] = []
        for line in raw_output.splitlines():
            stripped = line.strip().strip(",")
            if not stripped or stripped in {"{", "}", "[", "]"}:
                continue
            if stripped.startswith("{") or stripped.endswith("}"):
                continue
            if not re.search(r"[A-Za-z]", stripped):
                continue
            if re.fullmatch(r'"?hypotheses"?\s*:\s*\[?', stripped):
                continue
            if stripped.startswith('"hypotheses"') and "[" in stripped:
                continue
            normalized = _normalize_hypothesis(stripped)
            if normalized:
                parsed_lines.append(normalized)
        hypotheses = parsed_lines

    if not hypotheses:
        raise ValueError(f"Failed to parse hypotheses from output: {raw_output}")
    if len(hypotheses) < expected_count:
        raise ValueError(
            f"Expected {expected_count} hypotheses, only parsed {len(hypotheses)}: {raw_output}"
        )
    return hypotheses[:expected_count]


def parse_sentence_list(raw_output: str, expected_count: int) -> List[str]:
    parsed = extract_json_object(raw_output)
    sentences: List[str] = []
    if isinstance(parsed, dict):
        candidate = parsed.get("sentences")
        if isinstance(candidate, list):
            sentences = [
                _normalize_sentence(item)
                for item in candidate
                if isinstance(item, str) and _normalize_sentence(item)
            ]

    if not sentences:
        array_block = _extract_named_json_array_block(raw_output, key="sentences")
        if array_block:
            sentences = [
                _normalize_sentence(item)
                for item in _parse_string_list_from_jsonish_array(array_block)
                if _normalize_sentence(item)
            ]

    if not sentences:
        lines = [line for line in raw_output.splitlines() if line.strip()]
        sentences = [_normalize_sentence(line) for line in lines if _normalize_sentence(line)]

    if not sentences:
        raise ValueError(f"Failed to parse sentences from output: {raw_output}")
    if len(sentences) < expected_count:
        raise ValueError(
            f"Expected {expected_count} sentences, only parsed {len(sentences)}: {raw_output}"
        )
    return sentences[:expected_count]


def generate_hypotheses_single_call(
    *,
    side: SideType,
    observation: Dict[str, Any],
    num_hypothesis: int,
    client: OpenAI,
    model: str,
    token_counter: TokenUsageAccumulator,
    temperature: float,
    max_tokens: int,
    extra_guidance: Optional[str] = None,
    max_retries: int = 2,
    retry_backoff_seconds: float = 1.5,
) -> Tuple[List[str], Dict[str, Any]]:
    system_prompt = build_hypothesis_system_prompt(side)
    user_prompt = build_hypothesis_user_prompt(
        side=side,
        observation=observation,
        num_hypothesis=num_hypothesis,
        extra_guidance=extra_guidance,
    )
    llm_calls: List[Dict[str, Any]] = []
    max_attempts = max(1, int(max_retries) + 1)
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        retry_note = ""
        if attempt > 1:
            retry_note = (
                "\n\nSTRICT FORMAT RETRY: Your previous answer could not be parsed. "
                f"Return valid JSON only, exactly in this shape: "
                f'{{"hypotheses": ["hypothesis 1", "..."]}} with {num_hypothesis} string item(s).'
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + retry_note},
        ]
        raw_output, usage_obj = call_llm(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        usage = token_counter.add(usage_obj)
        record: Dict[str, Any] = {
            "call_type": "hypothesis_generation",
            "side": side,
            "messages": messages,
            "raw_output": raw_output,
            "usage": usage,
            "attempt": attempt,
        }
        try:
            hypotheses = parse_hypothesis_list(raw_output, expected_count=num_hypothesis)
        except ValueError as exc:
            last_error = str(exc)
            record["parse_status"] = "failed"
            record["parse_error"] = last_error
            llm_calls.append(record)
            if attempt < max_attempts:
                time.sleep(max(0.0, retry_backoff_seconds) * attempt)
                continue
            raise LLMParseFailure(
                f"{side} hypothesis generation failed to parse after {max_attempts} attempts.",
                llm_calls=llm_calls,
            ) from exc
        record["parse_status"] = "ok"
        llm_calls.append(record)
        summary_record = dict(record)
        summary_record["attempt_count"] = attempt
        summary_record["attempts"] = [dict(item) for item in llm_calls]
        return hypotheses, summary_record

    raise LLMParseFailure(
        f"{side} hypothesis generation failed to parse after {max_attempts} attempts. {last_error}",
        llm_calls=llm_calls,
    )



def design_sentences_for_input(
    *,
    hypotheses: Sequence[str],
    num_sentences: int,
    client: OpenAI,
    model: str,
    token_counter: TokenUsageAccumulator,
    temperature: float,
    max_tokens: int,
    max_retries: int = 2,
    retry_backoff_seconds: float = 1.5,
    observation: Optional[Dict[str, Any]] = None,
) -> Tuple[List[List[str]], List[Dict[str, Any]]]:
    system_prompt = build_system_prompt("input")
    max_attempts = max(1, int(max_retries) + 1)
    _activation_examples: List[Dict[str, Any]] = []
    if observation:
        _inner = observation.get("input_side_observation", observation)
        _raw = _inner.get("activation_examples", [])
        if isinstance(_raw, list):
            def _max_val(ex: Dict[str, Any]) -> float:
                toks = ex.get("activation_tokens", [])
                return max((float(t.get("value", 0)) for t in toks if isinstance(t, dict)), default=0.0)
            _activation_examples = sorted(_raw, key=_max_val, reverse=True)[:5]

    _lock = threading.Lock()

    def _design_one(index: int, hypothesis: str) -> Tuple[int, List[str], List[Dict[str, Any]]]:
        user_prompt = build_user_prompt(
            side="input",
            hypothesis=hypothesis,
            num_sentences=num_sentences,
            activation_examples=_activation_examples if _activation_examples else None,
        )
        records: List[Dict[str, Any]] = []
        for attempt in range(1, max_attempts + 1):
            retry_note = ""
            if attempt > 1:
                _fmt = '{{"sentences": ["sentence 1", "..."]}}' + f" with {num_sentences} string item(s)."
                retry_note = (
                    "\n\nSTRICT FORMAT RETRY: Your previous answer could not be parsed. "
                    f"Return valid JSON only, exactly in this shape: {_fmt}"
                )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + retry_note},
            ]
            raw_output, usage_obj = call_llm_stream(
                client, model, messages, temperature=temperature, max_tokens=max_tokens
            )
            with _lock:
                usage = token_counter.add(usage_obj)
            record: Dict[str, Any] = {
                "call_type": "sentence_design",
                "side": "input",
                "hypothesis_index": index,
                "hypothesis_text": hypothesis,
                "messages": messages,
                "raw_output": raw_output,
                "usage": usage,
                "attempt": attempt,
            }
            try:
                designed_sentences = parse_sentence_list(raw_output, expected_count=num_sentences)
            except ValueError as exc:
                record["parse_status"] = "failed"
                record["parse_error"] = str(exc)
                records.append(record)
                if attempt < max_attempts:
                    time.sleep(max(0.0, retry_backoff_seconds) * attempt)
                    continue
                raise LLMParseFailure(
                    f"Input sentence design failed to parse after {max_attempts} attempts for hypothesis {index}.",
                    llm_calls=records,
                ) from exc
            record["parse_status"] = "ok"
            records.append(record)
            return index, designed_sentences, records
        raise LLMParseFailure(f"Exhausted attempts for hypothesis {index}.", llm_calls=records)

    hyp_list = list(hypotheses)
    n = len(hyp_list)
    results: Dict[int, List[str]] = {}
    all_calls: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = {executor.submit(_design_one, i, h): i for i, h in enumerate(hyp_list, start=1)}
        for future in as_completed(futures):
            idx, sentences, records = future.result()
            results[idx] = sentences
            all_calls.extend(records)

    all_sentences = [results[i] for i in range(1, n + 1)]
    all_calls.sort(key=lambda r: r.get("hypothesis_index", 0))
    return all_sentences, all_calls


def _extract_json_any(text: str) -> Optional[Any]:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            continue
        try:
            return json.loads(match.group(0))
        except Exception:
            continue
    return None


def _extract_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(_extract_string_list(item))
        return out
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(_extract_string_list(item))
        return out
    return []


def generate_boundary_contexts(
    *,
    client: OpenAI,
    model: str,
    explanation: str,
    boundary_case_count: int,
    max_tokens: int,
    token_counter: TokenUsageAccumulator,
    trigger_tokens: Optional[List[str]] = None,
    call_metadata: Optional[Dict[str, Any]] = None,
    max_retries: int = 2,
    retry_backoff_seconds: float = 1.5,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    if boundary_case_count <= 0:
        raise ValueError("boundary_case_count must be a positive integer.")

    trigger_block = ""
    if trigger_tokens:
        quoted = ", ".join(f'"{t}"' for t in trigger_tokens[:8])
        trigger_block = (
            f"\n\nKnown trigger tokens (these strongly activate the feature): {quoted}\n"
            "IMPORTANT: Do not use any of these tokens or their morphological variants "
            "in your boundary sentences. Use different words to convey a related concept."
        )

    system_prompt = build_boundary_system_prompt()
    user_prompt = build_boundary_user_prompt(
        hypothesis=explanation,
        boundary_case_count=boundary_case_count,
        trigger_block=trigger_block,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    def _collect_candidates(raw_text: str) -> List[str]:
        parsed = _extract_json_any(raw_text)
        candidates: List[str] = []
        if isinstance(parsed, dict):
            for key in ("boundary_cases", "cases", "examples", "contexts", "items"):
                if key in parsed:
                    candidates.extend(
                        _normalize_sentence(item)
                        for item in _extract_string_list(parsed[key])
                        if _normalize_sentence(item)
                    )
            if not candidates:
                candidates.extend(
                    _normalize_sentence(item)
                    for item in _extract_string_list(parsed)
                    if _normalize_sentence(item)
                )
        elif parsed is not None:
            candidates.extend(
                _normalize_sentence(item)
                for item in _extract_string_list(parsed)
                if _normalize_sentence(item)
            )

        if not candidates:
            for key in ("boundary_cases", "cases", "examples", "contexts", "items"):
                array_block = _extract_named_json_array_block(raw_text, key=key)
                if not array_block:
                    continue
                parsed_items = [
                    _normalize_sentence(item)
                    for item in _parse_string_list_from_jsonish_array(array_block)
                    if _normalize_sentence(item)
                ]
                if parsed_items:
                    candidates.extend(parsed_items)
                    break

        if not candidates:
            for line in raw_text.splitlines():
                cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
                if cleaned:
                    candidates.append(cleaned)
        deduped: List[str] = []
        seen = set()
        for item in candidates:
            if item and item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped

    llm_calls: List[Dict[str, Any]] = []
    max_attempts = max(1, int(max_retries) + 1)
    last_content = ""
    last_count = 0
    for attempt in range(1, max_attempts + 1):
        content, usage_obj = call_llm(
            client=client,
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            stream=False,
        )
        content = content.strip().strip("`")
        usage = token_counter.add(usage_obj)
        record: Dict[str, Any] = {
            "call_type": "boundary_context_generation",
            "side": "input",
            "messages": messages,
            "raw_output": content,
            "usage": usage,
            "attempt": attempt,
        }
        if isinstance(call_metadata, dict):
            record.update(call_metadata)
        llm_calls.append(record)

        deduped = _collect_candidates(content)
        if len(deduped) >= boundary_case_count:
            return deduped[:boundary_case_count], llm_calls

        last_content = content
        last_count = len(deduped)
        if attempt < max_attempts:
            time.sleep(max(0.0, retry_backoff_seconds) * attempt)

    raise LLMParseFailure(
        f"Boundary case generator returned {last_count} cases, "
        f"but {boundary_case_count} required after {max_attempts} attempts. "
        f"Raw output is recorded in llm_calls.",
        llm_calls=llm_calls,
    )


def design_boundary_sentences_for_input(
    *,
    hypotheses: Sequence[str],
    num_sentences: int,
    client: OpenAI,
    model: str,
    token_counter: TokenUsageAccumulator,
    max_tokens: int,
    trigger_tokens: Optional[List[str]] = None,
) -> Tuple[List[List[str]], List[Dict[str, Any]]]:
    def _boundary_one(index: int, hypothesis: str) -> Tuple[int, List[str], List[Dict[str, Any]], TokenUsageAccumulator]:
        local_counter = TokenUsageAccumulator()
        sentences, calls = generate_boundary_contexts(
            client=client,
            model=model,
            explanation=hypothesis,
            boundary_case_count=num_sentences,
            max_tokens=max_tokens,
            token_counter=local_counter,
            trigger_tokens=trigger_tokens,
            call_metadata={"hypothesis_index": index, "hypothesis_text": hypothesis},
        )
        return index, sentences, calls, local_counter

    hyp_list = list(hypotheses)
    n = len(hyp_list)
    results: Dict[int, List[str]] = {}
    all_calls: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = {executor.submit(_boundary_one, i, h): i for i, h in enumerate(hyp_list, start=1)}
        for future in as_completed(futures):
            idx, sentences, calls, local_counter = future.result()
            results[idx] = sentences
            all_calls.extend(calls)
            token_counter.add(local_counter.as_dict())

    all_sentences = [results[i] for i in range(1, n + 1)]
    return all_sentences, all_calls



def extract_trigger_tokens(observation: Dict[str, Any], limit: int = 10) -> List[str]:
    input_side = observation.get("input_side_observation", observation)
    examples = input_side.get("activation_examples", [])
    seen: List[str] = []

    def add_token(token: Any) -> None:
        text = str(token or "").strip()
        if text and text not in seen:
            seen.append(text)

    for ex in examples:
        if not isinstance(ex, dict):
            continue
        for tok in ex.get("activation_tokens", []):
            if not isinstance(tok, dict):
                continue
            add_token(tok.get("token"))
        add_token(ex.get("max_token"))

    for key in ("bos_token_top_tokens", "gradient_token_top_tokens"):
        for item in input_side.get(key, []):
            if not isinstance(item, dict):
                continue
            add_token(item.get("token_text"))
            add_token(item.get("token"))
    return seen[: int(limit)]


def load_bos_token_observation(
    *,
    bos_root: str,
    layer_id: str,
    feature_id: str,
    prompt_id: str = "prompt-0001",
) -> Dict[str, Any]:
    base = Path(bos_root)
    top_tokens_path = (
        base / f"layer-{layer_id}" / f"feature-{feature_id}" / "bos_token" / prompt_id / "top_tokens.json"
    )
    legacy_top_tokens_path = (
        base / f"layer-{layer_id}" / f"feature-{feature_id}" / "bos_token" / "top_tokens.json"
    )
    if not top_tokens_path.exists() and legacy_top_tokens_path.exists():
        top_tokens_path = legacy_top_tokens_path
    if not top_tokens_path.exists():
        raise FileNotFoundError(f"BOS top_tokens.json not found: {top_tokens_path}")

    payload = json.loads(top_tokens_path.read_text(encoding="utf-8"))
    top_tokens = payload.get("top_tokens", [])
    if not isinstance(top_tokens, list):
        top_tokens = []

    feature_summary_path = top_tokens_path.parent / "scan_summary.json"
    summary_path = (
        feature_summary_path
        if feature_summary_path.exists()
        else base / f"layer-{layer_id}" / "bos_token" / prompt_id / "scan_summary.json"
    )
    legacy_summary_path = base / f"layer-{layer_id}" / "bos_token_scan_summary.json"
    if not summary_path.exists() and legacy_summary_path.exists():
        summary_path = legacy_summary_path
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        raw = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            summary = raw

    return {
        "source": "bos_token",
        "selected_count": len(top_tokens),
        "bos_token_top_tokens": top_tokens,
        "bos_token_scan_meta": {
            "prompt_id": payload.get("prompt_id", prompt_id),
            "prompt_template": payload.get("prompt_template"),
            "activation_threshold": payload.get("activation_threshold"),
            "evaluated_token_count": payload.get("evaluated_token_count"),
            "max_activation_seen": payload.get("max_activation_seen"),
            "summary": summary,
        },
    }


def load_gradient_token_observation(
    *,
    gradient_root: str,
    layer_id: str,
    feature_id: str,
    prompt_id: str = "prompt-0001",
) -> Dict[str, Any]:
    base = Path(gradient_root)
    top_tokens_path = (
        base
        / f"layer-{layer_id}"
        / f"feature-{feature_id}"
        / "gradient_token"
        / prompt_id
        / "top_tokens.json"
    )
    legacy_top_tokens_path = (
        base / f"layer-{layer_id}" / f"feature-{feature_id}" / "gradient_token" / "top_tokens.json"
    )
    if not top_tokens_path.exists() and legacy_top_tokens_path.exists():
        top_tokens_path = legacy_top_tokens_path
    if not top_tokens_path.exists():
        raise FileNotFoundError(f"Gradient token top_tokens.json not found: {top_tokens_path}")

    payload = json.loads(top_tokens_path.read_text(encoding="utf-8"))
    top_tokens = payload.get("top_tokens", [])
    if not isinstance(top_tokens, list):
        top_tokens = []

    return {
        "source": "gradient_token",
        "selected_count": len(top_tokens),
        "gradient_token_top_tokens": top_tokens,
        "gradient_token_scan_meta": {
            "prompt_id": payload.get("prompt_id"),
            "prompt": payload.get("prompt"),
            "objective": payload.get("objective"),
            "rank_by": payload.get("rank_by"),
            "aggregation": payload.get("aggregation"),
            "top_k": payload.get("top_k"),
            "selected_position": payload.get("selected_position"),
            "selected_token_id": payload.get("selected_token_id"),
            "selected_token_text": payload.get("selected_token_text"),
            "selected_pre_activation": payload.get("selected_pre_activation"),
            "analyzed_position_count": payload.get("analyzed_position_count"),
            "position_gradients": payload.get("position_gradients", []),
            "prompt_tokens": payload.get("prompt_tokens", []),
        },
    }


def limit_input_token_observation(
    observation: Dict[str, Any],
    token_count: int,
) -> Dict[str, Any]:
    """Return the minimal token observation sent to input-hypothesis LLM calls."""
    limit = int(token_count)
    source = str(observation.get("source", "")).strip()

    def select_rows(rows: Any) -> List[Dict[str, Any]]:
        if not isinstance(rows, list):
            return []
        selected = rows if limit <= 0 else rows[:limit]
        return [row for row in selected if isinstance(row, dict)]

    def project_row(row: Dict[str, Any], fields: Sequence[str]) -> Dict[str, Any]:
        return {field: row[field] for field in fields if field in row}

    if source == "bos_token":
        rows = [
            project_row(row, ("rank", "token_text", "activation"))
            for row in select_rows(observation.get("bos_token_top_tokens", []))
        ]
        return {
            "source": "bos_token",
            "selected_count": len(rows),
            "bos_token_top_tokens": rows,
        }

    if source == "gradient_token":
        rows = [
            project_row(row, ("rank", "token_text", "gradient_score"))
            for row in select_rows(observation.get("gradient_token_top_tokens", []))
        ]
        return {
            "source": "gradient_token",
            "selected_count": len(rows),
            "gradient_token_top_tokens": rows,
        }

    return observation


def extract_max_activation(observation: Dict[str, Any]) -> Optional[float]:
    """Temporarily deprecated: fixed workflow interventions use Step 3 prompts.

    This observation-based helper is kept only for legacy callers. It is not
    appropriate for gradient-token observations because they rank directional
    sensitivity evidence rather than direct feature activations.
    """
    input_side = observation.get("input_side_observation", observation)
    examples = input_side.get("activation_examples", [])
    values = [float(ex["maxValue"]) for ex in examples if isinstance(ex, dict) and "maxValue" in ex]
    if not values:
        values = [
            float(ex["activation_value"])
            for ex in examples
            if isinstance(ex, dict) and "activation_value" in ex
        ]
    bos_tokens = input_side.get("bos_token_top_tokens", [])
    for item in bos_tokens:
        if isinstance(item, dict) and "activation_value" in item:
            values.append(float(item["activation_value"]))
        elif isinstance(item, dict) and "activation" in item:
            values.append(float(item["activation"]))
    return max(values) if values else None


def select_steering_prompts(
    *,
    experiment_item: Dict[str, Any],
    fallback_prompts: Sequence[str],
    max_prompts: int,
    custom_prompts: Optional[List[str]] = None,
) -> List[str]:
    if custom_prompts:
        prompts = [str(p).strip() for p in custom_prompts if str(p).strip()]
        return prompts[:max_prompts] if max_prompts > 0 else prompts
    raw = experiment_item.get("designed_sentences") if isinstance(experiment_item, dict) else None
    prompts: List[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                prompts.append(item.strip())
    if not prompts:
        prompts = [str(p).strip() for p in fallback_prompts if str(p).strip()]
    return prompts[:max_prompts] if max_prompts > 0 else prompts


def compute_tokenchange(
    *,
    module: Any,
    prompts: Sequence[str],
    feature_id: int,
    top_k: int,
    intervention_scope: str = "last_token_only",
    kl_tolerance: float = 0.1,
    kl_max_steps: int = 12,
    max_clamp_value: Optional[float] = None,
    max_activation_scale: float = 2.0,
    last_token_scale: float = 1.0,
    target_kl: float = 0.25,
) -> Dict[str, Any]:
    import torch

    def _ids_to_tokens(token_ids: List[int]) -> List[str]:
        if module.tokenizer is not None:
            return module.tokenizer.convert_ids_to_tokens(token_ids)
        return [str(x) for x in token_ids]

    attention_mask = None
    if module.tokenizer is not None:
        enc = module.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
        prompt_tokens = enc["input_ids"].to(module.device)
        raw_attention_mask = enc.get("attention_mask")
        if raw_attention_mask is not None:
            attention_mask = raw_attention_mask.to(module.device)
    elif hasattr(module.model, "to_tokens"):
        prompt_tokens = module.model.to_tokens(list(prompts))
        attention_mask = torch.ones_like(prompt_tokens, dtype=torch.long, device=module.device)
    else:
        raise RuntimeError("Unable to tokenize prompts for intervention.")

    clean_logits = module.run_logits(prompt_tokens, attention_mask=attention_mask)
    clean_mean_logits = clean_logits.mean(dim=(0, 1)).detach().float().cpu()

    selected_token_positions: List[int] = []
    selected_token_base_values: List[float] = []
    selected_token_target_values: List[float] = []
    intervention_value: Optional[float] = None

    # max_clamp_value is retained for older callers, but fixed-workflow
    # interventions now use the max activation observed on the Step 3 prompts.
    details = module.get_max_activation_intervention_details(
        input_ids=prompt_tokens,
        feature_index=feature_id,
        attention_mask=attention_mask,
        scale=float(max_activation_scale),
    )
    base_values = details["base_values"].detach().float().cpu()
    target_values = details["target_values"].detach().float().cpu()
    selected_token_positions = [int(x) for x in details["positions"].detach().cpu().tolist()]
    selected_token_base_values = [float(x) for x in base_values.tolist()]

    if intervention_scope == "max_activation_token":
        selected_token_target_values = [float(x) for x in target_values.tolist()]
        clamp_value = float(target_values.mean().item()) if target_values.numel() > 0 else 0.0
        mean_base_value = float(base_values.mean().item()) if base_values.numel() > 0 else 0.0
        mean_target_value = float(target_values.mean().item()) if target_values.numel() > 0 else 0.0
        clamp_value_source = "step3_prompt_max_activation_token"
    else:
        step3_max_activation = float(base_values.max().item()) if base_values.numel() > 0 else 0.0
        clamp_value = step3_max_activation * float(max_activation_scale)
        if intervention_scope == "last_token_only":
            clamp_value = step3_max_activation * float(last_token_scale)
        intervention_value = clamp_value
        selected_token_target_values = [float(clamp_value) for _ in selected_token_base_values]
        mean_base_value = step3_max_activation
        mean_target_value = float(clamp_value)
        clamp_value_source = "step3_steering_prompt_max_activation"

    steered_logits = module.run_logits_with_feature_intervention(
        input_ids=prompt_tokens,
        feature_index=feature_id,
        value=intervention_value,
        mode="clamp",
        attention_mask=attention_mask,
        intervention_scope=intervention_scope,
        max_activation_scale=float(max_activation_scale),
    )

    clean_logprob = torch.nn.functional.log_softmax(clean_logits, dim=-1)
    steered_logprob = torch.nn.functional.log_softmax(steered_logits, dim=-1)
    actual_kl = float((clean_logprob.exp() * (clean_logprob - steered_logprob)).sum(dim=-1).mean().item())

    steered_mean_logits = steered_logits.mean(dim=(0, 1)).detach().float().cpu()
    delta_logits = steered_mean_logits - clean_mean_logits
    abs_delta = delta_logits.abs()
    total_mass = float(abs_delta.sum().item())
    k_all = min(int(top_k), int(abs_delta.shape[0]))
    topk_values, topk_ids = torch.topk(abs_delta, k=k_all)
    topk_mass = float(topk_values.sum().item()) if total_mass > 0 else 0.0
    topk_ratio = (topk_mass / total_mass) if total_mass > 0 else 0.0
    prob = (abs_delta / total_mass).clamp(min=1e-12) if total_mass > 0 else abs_delta
    entropy = float(-(prob * prob.log()).sum().item()) if total_mass > 0 else 0.0

    pos_mask = delta_logits > 0
    pos_ids_all = pos_mask.nonzero(as_tuple=False).flatten()
    if pos_ids_all.numel() > 0:
        pos_values_all = delta_logits[pos_ids_all]
        pos_k = min(int(top_k), int(pos_values_all.shape[0]))
        pos_top_values, pos_top_local_ids = torch.topk(pos_values_all, k=pos_k)
        pos_top_ids = pos_ids_all[pos_top_local_ids]
    else:
        pos_top_values = torch.empty(0, dtype=delta_logits.dtype)
        pos_top_ids = torch.empty(0, dtype=torch.long)

    neg_mask = delta_logits < 0
    neg_ids_all = neg_mask.nonzero(as_tuple=False).flatten()
    if neg_ids_all.numel() > 0:
        neg_values_all = delta_logits[neg_ids_all]
        neg_k = min(int(top_k), int(neg_values_all.shape[0]))
        neg_top_abs_values, neg_top_local_ids = torch.topk(-neg_values_all, k=neg_k)
        neg_top_ids = neg_ids_all[neg_top_local_ids]
        neg_top_values = neg_values_all[neg_top_local_ids]
    else:
        neg_top_abs_values = torch.empty(0, dtype=delta_logits.dtype)
        neg_top_values = torch.empty(0, dtype=delta_logits.dtype)
        neg_top_ids = torch.empty(0, dtype=torch.long)

    topk_tokens = _ids_to_tokens(topk_ids.tolist())
    topk_positive_tokens = _ids_to_tokens(pos_top_ids.tolist())
    topk_negative_tokens = _ids_to_tokens(neg_top_ids.tolist())

    intervention_strength_ratio: Optional[float] = None
    intervention_direction: Optional[str] = None
    if mean_base_value is not None and mean_target_value is not None:
        if abs(float(mean_base_value)) > 1e-12:
            intervention_strength_ratio = float(mean_target_value / mean_base_value)
        if mean_target_value > mean_base_value:
            intervention_direction = "increase"
        elif mean_target_value < mean_base_value:
            intervention_direction = "decrease"
        else:
            intervention_direction = "neutral"

    return {
        "clamp_value": clamp_value,
        "clamp_value_source": clamp_value_source,
        "intervention_scope": intervention_scope,
        "max_activation_scale": float(max_activation_scale),
        "last_token_scale": float(last_token_scale),
        "actual_kl": actual_kl,
        "topk_ratio": topk_ratio,
        "delta_entropy": entropy,
        "selected_token_positions": selected_token_positions,
        "selected_token_base_values": selected_token_base_values,
        "selected_token_target_values": selected_token_target_values,
        "mean_selected_token_base_value": mean_base_value,
        "mean_selected_token_target_value": mean_target_value,
        "intervention_strength_ratio": intervention_strength_ratio,
        "intervention_direction": intervention_direction,
        "topk_tokens": [
            {"token_id": int(tok_id), "token": str(tok), "delta_abs": float(val)}
            for tok_id, tok, val in zip(topk_ids.tolist(), topk_tokens, topk_values.tolist())
        ],
        "topk_positive_tokens": [
            {
                "token_id": int(tok_id),
                "token": str(tok),
                "delta": float(val),
                "delta_abs": float(abs(float(val))),
            }
            for tok_id, tok, val in zip(
                pos_top_ids.tolist(),
                topk_positive_tokens,
                pos_top_values.tolist(),
            )
        ],
        "topk_negative_tokens": [
            {
                "token_id": int(tok_id),
                "token": str(tok),
                "delta": float(val),
                "delta_abs": float(abs_val),
            }
            for tok_id, tok, val, abs_val in zip(
                neg_top_ids.tolist(),
                topk_negative_tokens,
                neg_top_values.tolist(),
                neg_top_abs_values.tolist(),
            )
        ],
    }


def build_output_observation_from_tokenchange(
    token_change: Dict[str, Any],
    *,
    steering_prompts: Sequence[str],
    top_k: int,
) -> Dict[str, Any]:
    return {
        "source": "tokenchange",
        "output_top_tokens": [
            {
                "rank": i + 1,
                "token_text": str(item.get("token", "")),
                "delta": float(item.get("delta", 0.0)),
                "delta_abs": float(item.get("delta_abs", 0.0)),
                "token_id": int(item.get("token_id", 0)),
            }
            for i, item in enumerate(token_change.get("topk_positive_tokens", []))
            if isinstance(item, dict)
        ],
        "tokenchange_meta": {
            "prompts": list(steering_prompts),
            "actual_kl": float(token_change.get("actual_kl", 0.0)),
            "clamp_value": float(token_change.get("clamp_value", 0.0)),
            "clamp_value_source": str(token_change.get("clamp_value_source", "")),
            "intervention_scope": str(token_change.get("intervention_scope", "last_token_only")),
            "max_activation_scale": float(token_change.get("max_activation_scale", 2.0)),
            "last_token_scale": float(token_change.get("last_token_scale", 1.0)),
            "intervention_strength_ratio": token_change.get("intervention_strength_ratio"),
            "top_k": int(top_k),
        },
    }


def generate_output_hypothesis(
    *,
    observation: Dict[str, Any],
    client: OpenAI,
    model: str,
    token_counter: TokenUsageAccumulator,
    extra_guidance: Optional[str],
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    hypotheses, call = generate_hypotheses_single_call(
        side="output",
        observation=observation,
        num_hypothesis=1,
        client=client,
        model=model,
        token_counter=token_counter,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_guidance=extra_guidance,
    )
    call["call_type"] = "output_hypothesis_generation"
    return hypotheses[0], call


def resolve_strength_ratio(token_change: Dict[str, Any]) -> Optional[float]:
    ratio = token_change.get("intervention_strength_ratio")
    if isinstance(ratio, (float, int)):
        return float(ratio)
    base_value = token_change.get("mean_selected_token_base_value")
    target_value = token_change.get("mean_selected_token_target_value")
    try:
        base_float = float(base_value)
        target_float = float(target_value)
    except (TypeError, ValueError):
        return None
    if abs(base_float) <= 1e-12:
        return None
    return float(target_float / base_float)


def choose_candidate_rows(item: Dict[str, Any]) -> Dict[str, Any]:
    token_change = item.get("token_change", {}) if isinstance(item, dict) else {}
    intervention_scope = str(token_change.get("intervention_scope", "last_token_only"))
    strength_ratio = resolve_strength_ratio(token_change)
    last_token_scale = _safe_float(token_change.get("last_token_scale", 1.0), 1.0)

    if intervention_scope == "max_activation_token" and strength_ratio is not None and strength_ratio < 1.0:
        token_sources = [("topk_negative_tokens", "negative")]
        direction = "decrease"
    elif intervention_scope == "last_token_only" and last_token_scale < 0.0:
        token_sources = [("topk_negative_tokens", "negative")]
        direction = "decrease"
    else:
        token_sources = [
            ("topk_positive_tokens", "positive"),
            ("topk_negative_tokens", "negative"),
        ]
        if intervention_scope == "max_activation_token":
            direction = "increase" if strength_ratio is None or strength_ratio > 1.0 else "neutral"
        else:
            direction = "increase"

    candidate_rows: List[Dict[str, Any]] = []
    for token_source, side in token_sources:
        raw_rows = token_change.get(token_source, [])
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            delta = _safe_float(row.get("delta", 0.0))
            token_raw = str(row.get("token", ""))
            candidate_rows.append(
                {
                    "id": 0,
                    "token": token_raw,
                    "token_display": normalize_display_token(token_raw) or token_raw,
                    "delta": delta,
                    "delta_abs": abs(delta),
                    "side": side,
                    "token_source": token_source,
                }
            )

    candidate_rows.sort(key=lambda row: (str(row["token_display"]).lower(), str(row["token"]), row["side"]))
    for local_idx, row in enumerate(candidate_rows, start=1):
        row["id"] = local_idx

    return {
        "intervention_scope": intervention_scope,
        "intervention_direction": direction,
        "intervention_strength_ratio": strength_ratio,
        "candidate_token_source": "+".join(source for source, _ in token_sources),
        "candidate_rows": candidate_rows,
    }


def build_selection_prompts(
    *,
    output_hypothesis: str,
    candidate_rows: Sequence[Dict[str, Any]],
) -> Tuple[str, str]:
    system_prompt = (
        "You judge which vocabulary tokens clearly support an output hypothesis. "
        "Be conservative. Only select tokens that directly express or strongly imply the hypothesis. "
        "Judge only the token text; no model-delta direction is provided. "
        "Do NOT analyze tokens one by one. Do NOT write step-by-step reasoning. "
        "Scan the list, decide immediately, then output JSON. "
        'Return JSON only: {"matched_token_ids":[1,2], "reason":"one sentence"}'
    )
    token_lines = []
    for row in candidate_rows:
        token_lines.append(
            f'{int(row["id"])}. raw="{row["token"]}" display="{row["token_display"]}"'
        )
    user_prompt = (
        f"Output hypothesis:\n{output_hypothesis.strip()}\n\n"
        "Candidate tokens:\n"
        + "\n".join(token_lines)
        + "\n\nOutput JSON immediately. No per-token commentary."
    )
    return system_prompt, user_prompt


def parse_matched_token_ids(raw_output: str, valid_ids: Sequence[int]) -> Tuple[List[int], str]:
    parsed = extract_json_object(raw_output)
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")
    if "matched_token_ids" not in parsed or not isinstance(parsed["matched_token_ids"], list):
        raise ValueError("matched_token_ids must be a JSON list")
    if "reason" not in parsed or not isinstance(parsed["reason"], str):
        raise ValueError("reason must be a JSON string")

    matched: List[int] = []
    for value in parsed["matched_token_ids"]:
        if isinstance(value, bool):
            raise ValueError("matched_token_ids must contain integer ids")
        try:
            matched.append(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError("matched_token_ids must contain integer ids") from exc
    reason = parsed["reason"].strip()

    valid_id_set = set(int(v) for v in valid_ids)
    deduped: List[int] = []
    for token_id in matched:
        if token_id in valid_id_set and token_id not in deduped:
            deduped.append(token_id)
    return deduped, reason


def compute_support_ratio(
    *,
    candidate_rows: Sequence[Dict[str, Any]],
    matched_ids: Sequence[int],
) -> Tuple[Optional[float], float, float, List[Dict[str, Any]]]:
    positive_delta_sum = float(
        sum(
            _safe_float(row.get("delta", 0.0))
            for row in candidate_rows
            if str(row.get("side", "positive")) == "positive"
        )
    )
    all_delta_sum = float(sum(_safe_float(row.get("delta", 0.0)) for row in candidate_rows))
    candidate_delta_sum = positive_delta_sum if abs(positive_delta_sum) > 1e-12 else all_delta_sum
    matched_set = set(int(x) for x in matched_ids)
    matched_rows = [row for row in candidate_rows if int(row.get("id", 0)) in matched_set]
    matched_delta_sum = float(sum(_safe_float(row.get("delta", 0.0)) for row in matched_rows))
    if not candidate_rows or abs(candidate_delta_sum) <= 1e-12:
        return None, matched_delta_sum, candidate_delta_sum, matched_rows
    return float(matched_delta_sum / candidate_delta_sum), matched_delta_sum, candidate_delta_sum, matched_rows


def score_output_hypotheses(
    *,
    token_change_rows: Sequence[Dict[str, Any]],
    client: OpenAI,
    model: str,
    token_counter: TokenUsageAccumulator,
    max_retries: int = 2,
    retry_backoff_seconds: float = 1.5,
) -> Dict[str, Any]:
    per_hypothesis: List[Dict[str, Any]] = []
    llm_calls: List[Dict[str, Any]] = []
    for idx, item in enumerate(token_change_rows, start=1):
        if not isinstance(item, dict):
            continue
        output_hypothesis = str(item.get("output_hypothesis", "")).strip()
        choice = choose_candidate_rows(item)
        candidate_rows = choice["candidate_rows"]
        valid_ids = [int(row["id"]) for row in candidate_rows]

        matched_ids: List[int] = []
        llm_reason = ""
        raw_output = ""
        if output_hypothesis and candidate_rows:
            system_prompt, user_prompt = build_selection_prompts(
                output_hypothesis=output_hypothesis,
                candidate_rows=candidate_rows,
            )
            parsed_match, attempts = call_llm_with_parse_retry(
                client=client,
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                parser=lambda raw: parse_matched_token_ids(raw, valid_ids),
                token_counter=token_counter,
                expected_format='{"matched_token_ids":[1,2],"reason":"one sentence"}',
                max_tokens=4000,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            matched_ids, llm_reason = parsed_match
            for attempt_record in attempts:
                llm_calls.append({
                    "idx": idx,
                    "output_hypothesis": output_hypothesis,
                    "candidate_token_source": choice["candidate_token_source"],
                    **attempt_record,
                })

        support_ratio, matched_delta_sum, candidate_delta_sum, matched_rows = compute_support_ratio(
            candidate_rows=candidate_rows,
            matched_ids=matched_ids,
        )
        matched_positive_delta = float(sum(
            _safe_float(row.get("delta", 0.0))
            for row in matched_rows
            if str(row.get("side", "positive")) == "positive"
        ))
        matched_negative_penalty = float(sum(
            abs(_safe_float(row.get("delta", 0.0)))
            for row in matched_rows
            if str(row.get("side", "positive")) == "negative"
        ))
        if support_ratio is not None and candidate_delta_sum > 0:
            final_score = (matched_positive_delta - matched_negative_penalty) / candidate_delta_sum
        else:
            final_score = support_ratio
        per_hypothesis.append(
            {
                "idx": int(item.get("hypothesis_index", idx)),
                "input_hypothesis": str(item.get("input_hypothesis", "")),
                "output_hypothesis": output_hypothesis,
                "intervention_scope": str(choice["intervention_scope"]),
                "intervention_direction": str(choice["intervention_direction"]),
                "intervention_strength_ratio": choice["intervention_strength_ratio"],
                "candidate_token_source": str(choice["candidate_token_source"]),
                "candidate_tokens": candidate_rows,
                "matched_token_ids": matched_ids,
                "matched_tokens": matched_rows,
                "matched_token_count": len(matched_rows),
                "matched_delta_sum": matched_delta_sum,
                "matched_negative_penalty": matched_negative_penalty,
                "candidate_delta_sum": candidate_delta_sum,
                "support_ratio": support_ratio,
                "final_score": final_score,
                "llm_reason": llm_reason,
            }
        )

    ranked = [row for row in per_hypothesis if isinstance(row.get("final_score"), (float, int))]
    if ranked:
        ranked.sort(key=lambda x: float(x["final_score"]), reverse=True)
        best_idx = int(ranked[0]["idx"])
    else:
        best_idx = 1
    return {
        "metric_version": "output_llm_token_match_v3",
        "best_hypothesis_idx_by_final_score": best_idx,
        "per_hypothesis": per_hypothesis,
        "llm_calls": llm_calls,
        "token_usage": token_counter.as_dict(),
    }


def judge_chain_pair(
    *,
    input_hypothesis: str,
    output_hypothesis: str,
    token_change: Dict[str, Any],
    client: OpenAI,
    model: str,
    token_counter: TokenUsageAccumulator,
    extra_chain_guidance: Optional[str] = None,
    max_retries: int = 2,
    retry_backoff_seconds: float = 1.5,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    system_prompt = build_chain_judge_system_prompt()
    user_prompt = build_chain_judge_user_prompt(
        input_hypothesis=input_hypothesis,
        output_hypothesis=output_hypothesis,
        top_tokens=[item["token"] for item in token_change.get("topk_tokens", [])],
        extra_guidance=extra_chain_guidance,
    )
    def parse_chain_judgement(raw_output: str) -> Dict[str, Any]:
        parsed = extract_json_object(raw_output)
        if not isinstance(parsed, dict):
            raise ValueError("expected a JSON object")
        score = parsed.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or int(score) != score
            or not 1 <= int(score) <= 5
        ):
            raise ValueError("score must be an integer from 1 to 5")
        reason = parsed.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        return {
            **parsed,
            "score": int(score),
            "reason": reason.strip(),
        }

    parsed, attempts = call_llm_with_parse_retry(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        parser=parse_chain_judgement,
        token_counter=token_counter,
        expected_format='{"score":3,"reason":"short reason"}',
        max_tokens=3000,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    call = {
        "call_type": "chain_judge",
        "input_hypothesis": input_hypothesis,
        "output_hypothesis": output_hypothesis,
        "attempt_count": len(attempts),
        "attempts": attempts,
        **attempts[-1],
    }
    return parsed, call


def output_scores_by_idx(score_payload: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    rows = score_payload.get("per_hypothesis", [])
    return {
        int(row.get("idx", idx + 1)): row
        for idx, row in enumerate(rows)
        if isinstance(row, dict)
    }


def pick_best_pair(
    pairs: Sequence[Dict[str, Any]],
    *,
    chain_threshold: int = 4,
    output_threshold: float = 0.5,
) -> int:
    """Select the best (input, output) hypothesis pair.

    Priority (descending):
      1. Number of gates met: chain_judge_score >= chain_threshold (+1),
         output_final_score >= output_threshold (+1).
      2. chain_judge_score (higher is better causal explanation).
      3. output_final_score (higher is better output evidence).
    """
    best_idx = 1
    best_key: tuple = (-1, -1, -1.0e18)
    for pair in pairs:
        idx = int(pair.get("idx", best_idx))
        chain_score = int(pair.get("chain_judge_score", 0) or 0)
        output_score = pair.get("output_final_score")
        output_score_f = float(output_score) if isinstance(output_score, (float, int)) else -1.0e18
        thresholds_met = (
            (1 if chain_score >= chain_threshold else 0)
            + (1 if output_score_f >= output_threshold else 0)
        )
        key = (thresholds_met, chain_score, output_score_f)
        if key > best_key:
            best_key = key
            best_idx = idx
    return best_idx


def synthesize_chain_explanation(
    *,
    best_pair: Dict[str, Any],
    client: OpenAI,
    model: str,
    token_counter: TokenUsageAccumulator,
    max_retries: int = 2,
    retry_backoff_seconds: float = 1.5,
) -> Tuple[str, Dict[str, Any]]:
    token_change = best_pair.get("token_change", {}) or {}
    parsed = best_pair.get("chain_judge_parsed") or {}
    score = int(parsed.get("score", best_pair.get("chain_judge_score", 0)) or 0) if isinstance(parsed, dict) else 0
    reason = str(parsed.get("reason", best_pair.get("chain_judge_reason", ""))) if isinstance(parsed, dict) else ""
    topk_tokens = [t["token"] for t in token_change.get("topk_tokens", []) if isinstance(t, dict)]

    system_prompt = build_chain_synthesis_system_prompt()
    user_prompt = build_chain_synthesis_user_prompt(
        input_hypothesis=str(best_pair.get("input_hypothesis", "")),
        output_hypothesis=str(best_pair.get("output_hypothesis", "")),
        chain_judge_score=score,
        chain_judge_reason=reason,
        topk_delta_tokens=topk_tokens,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    def parse_chain_explanation(raw_output: str) -> str:
        parsed_out = extract_json_object(raw_output)
        if not isinstance(parsed_out, dict):
            raise ValueError("expected a JSON object")
        explanation = parsed_out.get("chain_explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("chain_explanation must be a non-empty string")
        explanation = explanation.strip()
        if len(explanation.split()) > 50:
            raise ValueError("chain_explanation must contain at most 50 words")
        return explanation

    explanation, attempts = call_llm_with_parse_retry(
        client=client,
        model=model,
        messages=messages,
        parser=parse_chain_explanation,
        token_counter=token_counter,
        expected_format='{"chain_explanation":"at most 50 words"}',
        max_tokens=3000,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    return explanation, {
        "call_type": "chain_synthesis",
        "messages": messages,
        "attempt_count": len(attempts),
        "attempts": attempts,
        **attempts[-1],
    }
