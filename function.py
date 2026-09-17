from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - allows local unit tests of non-LLM helpers
    OpenAI = Any  # type: ignore[misc,assignment]

DEFAULT_CANONICAL_MAP_PATH = Path("support_info") / "canonical_map.txt"


def append_text_locked(path: Union[str, Path], text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(str(text))
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def append_line_locked(path: Union[str, Path], line: str) -> None:
    text = str(line)
    if not text.endswith("\n"):
        text += "\n"
    append_text_locked(path, text)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def extract_usage_counts(usage: Any) -> Dict[str, int]:
    """
    Extract prompt/completion/total token counts from OpenAI usage-like objects.
    """
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if isinstance(usage, Mapping):
        prompt_tokens = _safe_int(usage.get("prompt_tokens"))
        completion_tokens = _safe_int(usage.get("completion_tokens"))
        total_tokens = _safe_int(usage.get("total_tokens"))
    else:
        prompt_tokens = _safe_int(getattr(usage, "prompt_tokens", 0))
        completion_tokens = _safe_int(getattr(usage, "completion_tokens", 0))
        total_tokens = _safe_int(getattr(usage, "total_tokens", 0))

    if total_tokens == 0 and (prompt_tokens > 0 or completion_tokens > 0):
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def normalize_round_id(round_id: Optional[str], *, round_index: Optional[int] = None) -> str:
    text = str(round_id).strip() if round_id is not None else ""
    if text:
        return text
    if round_index is not None:
        return f"round_{int(round_index)}"
    return "round_1"


def build_feature_dir(
    *,
    layer_id: str,
    feature_id: str,
    logs_root: Optional[Union[str, Path]] = None,
) -> Path:
    root = Path(logs_root) if logs_root is not None else Path("logs")
    return root / f"layer-{str(layer_id)}" / f"feature-{str(feature_id)}"


def build_round_dir(
    *,
    layer_id: str,
    feature_id: str,
    timestamp: str,
    round_id: Optional[str] = None,
    round_index: Optional[int] = None,
    logs_root: Optional[Union[str, Path]] = None,
) -> Path:
    resolved_round_id = normalize_round_id(round_id, round_index=round_index)
    return (
        build_feature_dir(layer_id=layer_id, feature_id=feature_id, logs_root=logs_root)
        / str(timestamp)
        / resolved_round_id
    )


def extract_average_l0_from_canonical_map(
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


def build_default_sae_path(
    *,
    layer_id: str,
    width: str,
    release: str,
    average_l0: Optional[str],
    canonical_map_path: Optional[Union[str, Path]],
) -> Tuple[str, str]:
    resolved_average_l0 = average_l0
    if not resolved_average_l0 and canonical_map_path:
        resolved_average_l0 = extract_average_l0_from_canonical_map(
            canonical_map_path=Path(canonical_map_path),
            layer_id=layer_id,
            width=width,
        )

    if not resolved_average_l0:
        # Keep backward-compatible fallback.
        resolved_average_l0 = "70"

    sae_uri = (
        "sae-lens://"
        f"release={release};"
        f"sae_id=layer_{layer_id}/width_{width}/average_l0_{resolved_average_l0}"
    )
    return sae_uri, resolved_average_l0


@dataclass
class TokenUsageAccumulator:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: Any) -> Dict[str, int]:
        counts = extract_usage_counts(usage)
        self.prompt_tokens += counts["prompt_tokens"]
        self.completion_tokens += counts["completion_tokens"]
        self.total_tokens += counts["total_tokens"]
        return counts

    def as_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def read_api_key(api_key_file: Optional[str]) -> str:
    env_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key.strip()

    if api_key_file:
        key = Path(api_key_file).read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError(f"API key file is empty: {api_key_file}")
        return key

    raise ValueError("Missing API key: set LLM_API_KEY or OPENAI_API_KEY.")


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = text.strip()
    if not raw:
        return None

    try:
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None

    try:
        loaded = json.loads(match.group(0))
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        return None
    return None


def _extract_text_from_message(message: Any) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    text = ""
    content_type = "missing"
    message_dump: Optional[Dict[str, Any]] = None

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip(), "content_str", None

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue

            if isinstance(item, dict):
                maybe_text = item.get("text")
                if isinstance(maybe_text, str):
                    parts.append(maybe_text)
                    continue
                maybe_content = item.get("content")
                if isinstance(maybe_content, str):
                    parts.append(maybe_content)
                    continue

            maybe_text = getattr(item, "text", None)
            if isinstance(maybe_text, str):
                parts.append(maybe_text)
                continue
            maybe_content = getattr(item, "content", None)
            if isinstance(maybe_content, str):
                parts.append(maybe_content)
                continue

            if hasattr(item, "model_dump"):
                try:
                    item_dump = item.model_dump()
                    if isinstance(item_dump, dict):
                        if isinstance(item_dump.get("text"), str):
                            parts.append(item_dump["text"])
                        elif isinstance(item_dump.get("content"), str):
                            parts.append(item_dump["content"])
                except Exception:
                    pass

        joined = "".join(parts).strip()
        if joined:
            return joined, "content_list", None

    reasoning_content = getattr(message, "reasoning_content", None)
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return reasoning_content.strip(), "reasoning_content_str", None

    if hasattr(message, "model_dump"):
        try:
            message_dump = message.model_dump()
            if isinstance(message_dump, dict):
                for key in ("output_text", "text", "content", "reasoning_content"):
                    value = message_dump.get(key)
                    if isinstance(value, str) and value.strip():
                        text = value.strip()
                        content_type = f"message_dump_{key}"
                        break
        except Exception:
            message_dump = None

    return text, content_type, message_dump


def call_llm(
    client: OpenAI,
    model: str,
    messages: Sequence[Dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    stream: bool = False,
    response_format_text: bool = False,
    return_debug: bool = False,
) -> Union[Tuple[str, Any], Tuple[str, Any, Dict[str, Any]]]:
    if stream:
        response_stream = client.chat.completions.create(
            model=model,
            messages=list(messages),
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            stream_options={"include_usage": True},
            
        )

        output_parts = []
        usage_obj = None
        for chunk in response_stream:
            if getattr(chunk, "choices", None):
                delta = chunk.choices[0].delta.content
                if delta:
                    output_parts.append(delta)
            if getattr(chunk, "usage", None) is not None:
                usage_obj = chunk.usage

        text = "".join(output_parts).strip()
        if return_debug:
            return text, usage_obj, {"mode": "stream"}
        return text, usage_obj

    response = None
    request_mode = "without_response_format"
    request_error = None
    request_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
        
    }
    if response_format_text:
        request_mode = "with_response_format_text"
        request_kwargs["response_format"] = {"type": "text"}

    try:
        response = client.chat.completions.create(**request_kwargs)
    except Exception as exc:
        if not response_format_text:
            raise
        request_mode = "fallback_without_response_format"
        request_error = repr(exc)
        fallback_kwargs = {
            "model": model,
            "messages": list(messages),
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
            
        }
        response = client.chat.completions.create(**fallback_kwargs)

    text = ""
    content_type = "missing"
    finish_reason = None
    message_dump = None

    if getattr(response, "choices", None):
        choice0 = response.choices[0]
        finish_reason = getattr(choice0, "finish_reason", None)
        message = choice0.message
        text, content_type, message_dump = _extract_text_from_message(message)

    usage_obj = getattr(response, "usage", None)
    if return_debug:
        return text, usage_obj, {
            "mode": "non_stream",
            "request_mode": request_mode,
            "request_error": request_error,
            "content_type": content_type,
            "finish_reason": finish_reason,
            "message_dump": message_dump,
        }
    return text, usage_obj


def call_llm_stream(
    client: OpenAI,
    model: str,
    messages: Sequence[Dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Any]:
    text, usage_obj = call_llm(
        client=client,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    return text, usage_obj
