"""
agent_proposer.py — Tool-use agent loop for Phase 3 harness proposal.

The agent has direct access to file system tools and investigates raw trace
files autonomously before proposing a new harness configuration.
"""
from __future__ import annotations

import json
import re
from contextlib import nullcontext
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - allows local tests without OpenAI installed
    OpenAI = Any  # type: ignore[misc,assignment]

from function import TokenUsageAccumulator, append_text_locked
from skill_lock import shared_skill_lock
from skill_maintenance import maybe_run_auto_skill_maintenance
from skill_case_store import (
    CASE_FILES,
    append_agent_case,
)
from trace_metrics import extract_trace_metrics

SKILLS_DIR = Path(__file__).parent / "skills"
_CURRENT_SKILLS_DIR: ContextVar[Path] = ContextVar("agent_proposer_skills_dir", default=SKILLS_DIR)
_CURRENT_LOGS_ROOT: ContextVar[Path] = ContextVar(
    "agent_proposer_logs_root",
    default=Path(__file__).parent / "logs",
)
_SKILL_LEARNING_ENABLED: ContextVar[bool] = ContextVar("agent_proposer_skill_learning_enabled", default=True)
_CURRENT_FEATURE_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar("agent_proposer_feature_context", default={})
_CURRENT_FEATURE_OBSERVATION_DIR: ContextVar[Optional[Path]] = ContextVar(
    "agent_proposer_feature_observation_dir",
    default=None,
)
_CURRENT_TOOL_EVENTS: ContextVar[List[Dict[str, Any]]] = ContextVar("agent_proposer_tool_events", default=[])
_CURRENT_SKILL_MAINTENANCE: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "agent_proposer_skill_maintenance",
    default=None,
)
_MAX_EVIDENCE_CHARS = 3000  # cap pre-injected evidence to avoid token bloat

# Per-mode caps for one proposal round. Stronger models need room to inspect files before finishing.
DEFAULT_MAX_TOOL_CALLS = 50
_MODE_MAX_TOOL_CALLS = {
    "chain":           15,
    "input_validate":  50,
    "output_validate": 50,
}
_MODE_THINKING_BUDGET = {
    "chain":           6000,
    "input_validate":  2000,
    "output_validate": 2000,
}


# ── Tool schemas ──────────────────────────────────────────────────────────────

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file from the current feature's logs/ directory or from skills/. "
                "Use this to inspect raw activation samples, per-sentence results, "
                "token change data, hypothesis texts, or skill guides."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path relative to the code directory. "
                            "e.g. 'logs/layer-6/feature-100/20260420_175328/round_4/layer6-feature100-step4-input-experiment-scores.json' "
                            "or 'skills/diagnosis_guide.md'"
                        ),
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": (
                "List files and subdirectories under the current feature's logs/ directory or under skills/."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path, e.g. 'logs/layer-6/feature-100/20260420_175328'",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_json_value",
            "description": (
                "Extract one value from a JSON file using a dot-separated key path. "
                "The JSON file must be under the current feature's logs/ directory or under skills/. "
                "Use integer indices for arrays, e.g. 'chain.pairs.0.chain_judge_score'. "
                "Faster than read_file when you only need one metric."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to JSON file"},
                    "key_path": {
                        "type": "string",
                        "description": "Dot-separated key path, e.g. 'input_round.eval.overall_activation_rate'",
                    },
                },
                "required": ["path", "key_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Output your final diagnosis and harness configuration. "
                "Call this once you have gathered enough evidence to make a confident decision."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["retry", "stop"],
                        "description": (
                            "'retry' to run another pipeline iteration with the given harness config. "
                            "'stop' if the score already meets the threshold or further iteration is unlikely to help."
                        ),
                    },
                    "diagnosis": {
                        "type": "string",
                        "description": (
                            "One paragraph explaining what failed and why, "
                            "citing specific evidence from the files you read (e.g. actual activation rates, "
                            "specific sentences that failed, token examples)."
                        ),
                    },
                    "harness": {
                        "type": "object",
                        "description": "Required if action='retry'. Specifies what to change in the next pipeline run.",
                        "properties": {
                            "skip_observation_and_design": {
                                "type": "boolean",
                                "description": (
                                    "If true, reuse previous observation, hypotheses, input experiments, and input scores; "
                                    "only re-run steps 5-9 (intervention, output hypotheses, scoring, chain selection, synthesis). "
                                    "Use when input-side artifacts are OK but steering prompts or output scoring needs fixing."
                                ),
                            },
                            "custom_steering_prompts": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "5–7 short neutral sentence prefixes to replace designed input-side sentences "
                                    "as steering context. e.g. ['The answer is', 'In conclusion,', 'Therefore,']"
                                ),
                            },
                            "intervention_scope": {
                                "type": "string",
                                "enum": ["last_token_only", "all_tokens", "max_activation_token"],
                                "description": (
                                    "Which token position to steer during intervention. "
                                    "max_activation_token: steers where the SAE feature activates most strongly. "
                                    "last_token_only: steers the penultimate token (position -2). "
                                    "Consult the output skill for when to switch between these. "
                                    "Do not use all_tokens."
                                ),
                            },
                            "max_activation_scale": {
                                "type": "number",
                                "description": (
                                    "Scale factor for max-activation-token intervention. "
                                    "Only used when intervention_scope='max_activation_token'."
                                ),
                            },
                            "last_token_scale": {
                                "type": "number",
                                "description": (
                                    "Scale factor applied to the observed max activation before clamping the last token. "
                                    "Only used when intervention_scope='last_token_only'."
                                ),
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of top delta tokens to track (default 30).",
                            },
                            "extra_input_guidance": {
                                "type": "string",
                                "description": "Additional instruction appended to the input hypothesis generation prompt.",
                            },
                            "bos_prompt_id": {
                                "type": "string",
                                "description": (
                                    "BOS runs only: select a feature-local prompt id returned by "
                                    "run_bos_token_scan, then rerun the input side from Step 1."
                                ),
                            },
                            "extra_output_guidance": {
                                "type": "string",
                                "description": "Additional instruction appended to the step6 output hypothesis generation prompt.",
                            },
                            "extra_chain_guidance": {
                                "type": "string",
                                "description": (
                                    "Additional instruction appended to the step8 chain-judge prompt. "
                                    "Use when Gates 1+2 pass but chain_judge_score < 4. "
                                    "Triggers rerun of steps 8-9 only. "
                                    "Example: 'The input is a lexical feature on token X. Focus the causal chain "
                                    "on the specific syntactic trigger and its downstream token-generation effect.'"
                                ),
                            },
                        },
                    },
                    "rerun_steps": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "List of step numbers to re-run (e.g. [6, 7, 8, 9]). "
                            "Step 9 is always added automatically if missing. "
                            "If omitted, the system infers steps from harness parameters (backward compatible)."
                        ),
                    },
                    "used_skill_cases": {
                        "type": "array",
                        "description": (
                            "Skill case IDs actually used to choose this harness. "
                            "Only include cases whose concrete experience influenced the proposed change."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "case_id": {"type": "string"},
                                "applied_experience": {"type": "string"},
                                "expected_effect": {"type": "string"},
                            },
                            "required": ["case_id"],
                        },
                    },
                },
                "required": ["action", "diagnosis"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bos_token_scan",
            "description": (
                "Run a BOS-token prompt-template scan for the current feature. "
                "This tool is available only when the run started from bos_token evidence. "
                "Use the returned prompt_id as harness.bos_prompt_id when the evidence is better."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt_template": {
                        "type": "string",
                        "description": (
                            "Use {token} as the insertion point; if omitted, append the candidate token."
                        ),
                    },
                    "prompt_id": {"type": "string", "description": "Feature-local scan id."},
                    "candidate_token_texts": {"type": "array", "items": {"type": "string"}},
                    "scan_full_vocab": {"type": "boolean"},
                    "random_sample_size": {"type": "integer"},
                    "top_k": {"type": "integer"},
                    "batch_size": {"type": "integer"},
                },
                "required": ["prompt_template", "prompt_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_skill",
            "description": (
                "Write a discovered technique to the skills/ knowledge base. "
                "Call this when you find evidence in PREVIOUS rounds that a specific approach "
                "improved a metric by a meaningful amount (>= 0.1 rate improvement or >= 1 chain score step). "
                "Choose the right layer:\n"
                "  Framework files (skill_input.md, skill_output.md, skill_chain.md): "
                "ONLY for truly general patterns not already covered — add or update a ## section.\n"
                "  Cases files (skill_input_cases.md, skill_output_cases.md, skill_chain_cases.md): "
                "for specific feature instances or narrow patterns that illustrate an existing rule.\n"
                "Call write_skill() BEFORE calling finish()."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": (
                            "Target .md filename in skills/. "
                            "Framework: 'skill_input.md', 'skill_output.md', 'skill_chain.md'. "
                            "Cases: 'skill_input_cases.md', 'skill_output_cases.md', 'skill_chain_cases.md'."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Full markdown content to write. Must include: "
                            "(1) a ## section heading, "
                            "(2) when to apply (the trace pattern that triggers this technique), "
                            "(3) what to change (specific harness parameter values), "
                            "(4) evidence (cite metric values before and after)."
                        ),
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["append", "create"],
                        "description": "'append' adds to an existing file; 'create' makes a new skill file.",
                    },
                },
                "required": ["filename", "content", "mode"],
            },
        },
    },
]

# ── Tool execution ─────────────────────────────────────────────────────────────

def _active_skills_dir() -> Path:
    return _CURRENT_SKILLS_DIR.get()


def _active_logs_root() -> Path:
    return _CURRENT_LOGS_ROOT.get()


def _skill_learning_enabled() -> bool:
    return bool(_SKILL_LEARNING_ENABLED.get())


def _feature_context() -> Dict[str, Any]:
    return dict(_CURRENT_FEATURE_CONTEXT.get() or {})


def _active_feature_logs_dir() -> Optional[Path]:
    context = _feature_context()
    layer_id = context.get("layer_id")
    feature_id = context.get("feature_id")
    if layer_id is None or feature_id is None:
        return None
    return (
        _active_logs_root()
        / f"layer-{layer_id}"
        / f"feature-{feature_id}"
    ).resolve()


def _tool_events() -> List[Dict[str, Any]]:
    return _CURRENT_TOOL_EVENTS.get()


def _record_tool_event(event: Dict[str, Any]) -> None:
    try:
        payload = dict(_feature_context())
        payload.update(event)
        _tool_events().append(payload)
    except Exception:
        pass


def _maybe_maintain_skills() -> Dict[str, Any]:
    config = _CURRENT_SKILL_MAINTENANCE.get()
    if not config or config.get("mode") != "auto":
        return {"status": "disabled"}
    result = maybe_run_auto_skill_maintenance(
        config["client"],
        str(config["model"]),
        skills_dir=_active_skills_dir(),
        archive_dir=Path(config["archive_dir"]),
        maintenance_name=str(config["maintenance_name"]),
        verbose=True,
    )
    _record_tool_event({"event": "skill_maintenance", **result})
    return result


def _skill_read_guard(path: Path):
    skills_dir = _active_skills_dir().resolve()
    try:
        if _is_relative_to(path.resolve(), skills_dir):
            return shared_skill_lock(skills_dir)
    except OSError:
        pass
    return nullcontext(True)


def _skill_file_kind(filename: str) -> str:
    if filename in set(CASE_FILES.values()):
        return "case"
    if filename.startswith("skill_") and filename.endswith(".md"):
        return "skill"
    if filename.endswith(".md"):
        return "skill_markdown"
    return "other_skill_file"


def _is_skill_archive_file(path: Path) -> bool:
    return path.name.endswith("_archive.md")


def _record_skill_write_event(
    *,
    filename: str,
    mode: str,
    result: str,
    event: Optional[str] = None,
    case_id: Optional[str] = None,
    trigger: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    payload: Dict[str, Any] = {
        "event": event or ("case_write" if filename in set(CASE_FILES.values()) else "skill_write"),
        "filename": filename,
        "mode": mode,
        "kind": _skill_file_kind(filename),
        "result": result,
    }
    if case_id:
        payload["case_id"] = case_id
    if trigger:
        payload["trigger"] = trigger
    if error:
        payload["error"] = error
    _record_tool_event(payload)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _tools_for_input_source(input_source: Optional[str]) -> List[Dict[str, Any]]:
    excluded = set()
    if str(input_source or "").strip() != "bos_token":
        excluded.add("run_bos_token_scan")
    if not _skill_learning_enabled():
        excluded.add("write_skill")
    return [
        tool
        for tool in _TOOLS
        if tool.get("function", {}).get("name") not in excluded
    ]


def _tools_for_current_context() -> List[Dict[str, Any]]:
    return _tools_for_input_source(_CURRENT_FEATURE_CONTEXT.get().get("input_source"))


def _safe_path(rel_path: str) -> Optional[Path]:
    base = Path(__file__).parent
    skills_dir = _active_skills_dir().resolve()
    logs_root = _active_logs_root().resolve()
    feature_logs_dir = _active_feature_logs_dir()
    feature_observation_dir = _CURRENT_FEATURE_OBSERVATION_DIR.get()
    if feature_observation_dir is not None:
        feature_observation_dir = Path(feature_observation_dir).resolve()
    raw = str(rel_path or "").strip()
    if not raw:
        return None
    try:
        raw_path = Path(raw)
        if raw_path.is_absolute():
            p = raw_path.resolve()
        elif raw == "skills" or raw.startswith("skills/"):
            suffix = raw.split("/", 1)[1] if "/" in raw else ""
            p = (skills_dir / suffix).resolve() if suffix else skills_dir
        elif raw == "logs" and feature_logs_dir is not None:
            p = feature_logs_dir
        elif raw == "logs" or raw.startswith("logs/"):
            suffix = raw.split("/", 1)[1] if "/" in raw else ""
            p = (logs_root / suffix).resolve() if suffix else logs_root
        elif raw == "initial_observation" and feature_observation_dir is not None:
            p = feature_observation_dir
        elif raw.startswith("initial_observation/") and feature_observation_dir is not None:
            suffix = raw.split("/", 1)[1]
            p = (feature_observation_dir / suffix).resolve()
        else:
            p = (base / raw).resolve()
    except Exception:
        return None
    if _is_relative_to(p, skills_dir) and (
        _is_skill_archive_file(p) or p.name == ".maintenance.lock"
    ):
        return None
    allowed = [skills_dir]
    if feature_logs_dir is not None:
        allowed.append(feature_logs_dir)
    if feature_observation_dir is not None:
        allowed.append(feature_observation_dir)
    return p if any(_is_relative_to(p, root) for root in allowed) else None


def _run_read_file(path: str) -> str:
    p = _safe_path(path)
    if p is None:
        return f"ERROR: path not allowed or outside current feature logs/skills: {path!r}"
    if not p.exists():
        return f"ERROR: file not found: {path!r}"
    if not p.is_file():
        return f"ERROR: not a file: {path!r}"
    with _skill_read_guard(p):
        content = p.read_text(encoding="utf-8", errors="replace")
    try:
        skills_dir = _active_skills_dir().resolve()
        resolved = p.resolve()
        if _is_relative_to(resolved, skills_dir):
            try:
                rel_path = str(resolved.relative_to(skills_dir))
            except ValueError:
                rel_path = p.name
            _record_tool_event({
                "event": "skill_read_file",
                "path": f"skills/{rel_path}",
                "filename": p.name,
                "kind": _skill_file_kind(p.name),
                "chars": len(content),
                "truncated": False,
            })
    except Exception:
        pass
    return content


def _run_list_directory(path: str) -> str:
    p = _safe_path(path)
    if p is None:
        return f"ERROR: path not allowed: {path!r}"
    if not p.exists():
        return f"ERROR: path not found: {path!r}"
    with _skill_read_guard(p):
        entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
    try:
        skills_dir = _active_skills_dir().resolve()
        if _is_relative_to(p.resolve(), skills_dir):
            entries = [
                e
                for e in entries
                if not _is_skill_archive_file(e) and e.name != ".maintenance.lock"
            ]
    except Exception:
        pass
    lines = [f"{'FILE' if e.is_file() else 'DIR '}  {e.name}" for e in entries]
    return "\n".join(lines) or "(empty directory)"


def _run_get_json_value(path: str, key_path: str) -> str:
    p = _safe_path(path)
    if p is None:
        return f"ERROR: path not allowed: {path!r}"
    if not p.exists():
        return f"ERROR: file not found: {path!r}"
    try:
        with _skill_read_guard(p):
            data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return f"ERROR: invalid JSON: {e}"
    val: Any = data
    for k in key_path.split("."):
        if isinstance(val, list):
            try:
                val = val[int(k)]
            except (ValueError, IndexError) as e:
                return f"ERROR: {e} at segment '{k}'"
        elif isinstance(val, dict):
            if k not in val:
                avail = list(val.keys())[:10]
                return f"ERROR: key '{k}' not found. Available keys: {avail}"
            val = val[k]
        else:
            return f"ERROR: cannot index into {type(val).__name__} at segment '{k}'"
    return json.dumps(val, ensure_ascii=False, indent=2)


def _run_write_skill(filename: str, content: str, mode: str) -> str:
    if not _skill_learning_enabled():
        _record_skill_write_event(filename=filename, mode=mode, result="error", error="skill learning is disabled")
        return "ERROR: skill learning is disabled for this run"
    if not filename.endswith(".md"):
        _record_skill_write_event(filename=filename, mode=mode, result="error", error="filename must end with .md")
        return "ERROR: filename must end with .md"
    if any(c in filename for c in ("/", "\\", "..")):
        _record_skill_write_event(filename=filename, mode=mode, result="error", error="invalid filename")
        return "ERROR: invalid filename"
    skills_dir = _active_skills_dir()
    skill_path = skills_dir / filename
    skills_dir.mkdir(parents=True, exist_ok=True)
    content_to_write = content.strip() + "\n"

    case_side = next((side for side, case_file in CASE_FILES.items() if case_file == filename), None)
    if case_side and mode == "append":
        case_id = append_agent_case(
            skills_dir,
            side=case_side,
            content=content_to_write,
            source_feature=_feature_context(),
            post_write=_maybe_maintain_skills,
        )
        _record_skill_write_event(
            filename=filename,
            mode=mode,
            result="ok",
            event="case_write",
            case_id=case_id,
            trigger="agent_decision",
        )
        return f"OK: appended case {case_id} to skills/{filename}"

    if mode == "create":
        with shared_skill_lock(skills_dir):
            if skill_path.exists():
                error = f"'{filename}' already exists - use mode='append' to add a section"
                _record_skill_write_event(filename=filename, mode=mode, result="error", error=error)
                return f"ERROR: {error}"
            append_text_locked(skill_path, content_to_write)
        _maybe_maintain_skills()
        _record_skill_write_event(filename=filename, mode=mode, result="ok")
        return f"OK: created skills/{filename}"
    if mode == "append":
        with shared_skill_lock(skills_dir):
            if not skill_path.exists():
                error = f"'{filename}' not found - use mode='create' for a new file"
                _record_skill_write_event(filename=filename, mode=mode, result="error", error=error)
                return f"ERROR: {error}"
            prefix = ""
            try:
                if skill_path.read_text(encoding="utf-8").strip():
                    prefix = "\n\n"
            except OSError:
                pass
            append_text_locked(skill_path, prefix + content_to_write)
        _maybe_maintain_skills()
        _record_skill_write_event(filename=filename, mode=mode, result="ok")
        return f"OK: appended to skills/{filename}"
    _record_skill_write_event(filename=filename, mode=mode, result="error", error=f"unknown mode '{mode}'")
    return f"ERROR: unknown mode '{mode}'"


def _run_bos_token_scan(args: Dict[str, Any]) -> str:
    context = _feature_context()
    if context.get("input_source") != "bos_token":
        return "ERROR: run_bos_token_scan is available only in a bos_token run"
    required = ("sae_path", "model_path", "bos_token_root")
    missing = [key for key in required if not context.get(key)]
    if missing:
        return f"ERROR: BOS scan context is missing: {', '.join(missing)}"
    try:
        from agent_bos_token_scan_tool import run_agent_bos_token_scan

        raw_candidates = args.get("candidate_token_texts") or []
        if isinstance(raw_candidates, str):
            raw_candidates = [raw_candidates]
        result = run_agent_bos_token_scan(
            layer_id=str(context["layer_id"]),
            feature_id=str(context["feature_id"]),
            model_checkpoint_path=str(context["model_path"]),
            sae_path=str(context["sae_path"]),
            prompt_template=str(args.get("prompt_template") or ""),
            prompt_id=str(args.get("prompt_id") or ""),
            candidate_token_texts=[str(value) for value in raw_candidates],
            scan_full_vocab=bool(args.get("scan_full_vocab", False)),
            random_sample_size=int(args.get("random_sample_size", 0) or 0),
            top_k=int(args.get("top_k", 50) or 50),
            batch_size=int(args.get("batch_size", 64) or 64),
            device=str(context.get("device") or "cpu"),
            width=str(context.get("sae_width") or "16k"),
            output_root=str(context["bos_token_root"]),
            inference_server_url=str(context.get("inference_server_url") or ""),
            inference_timeout_sec=float(context.get("inference_timeout_sec") or 600.0),
            no_inference_server=bool(context.get("no_inference_server", False)),
        )
        _record_tool_event({
            "event": "bos_token_scan",
            "prompt_id": result.get("prompt_id"),
            "prompt_template": result.get("prompt_template"),
            "output_path": result.get("output_path"),
        })
        slim = {
            key: result.get(key)
            for key in (
                "status",
                "requested_prompt_id",
                "prompt_id",
                "prompt_id_collision_renamed",
                "prompt_template",
                "output_path",
                "missing_manual_tokens",
                "candidate_token_count",
                "scan_full_vocab",
                "random_sample_size",
                "top_tokens",
            )
        }
        return json.dumps(slim, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"ERROR: BOS token scan failed: {type(exc).__name__}: {exc}"


def _dispatch(name: str, raw_args: str) -> str:
    try:
        args = json.loads(raw_args) if raw_args else {}
    except Exception:
        return f"ERROR: could not parse tool arguments: {raw_args!r}"
    if name == "read_file":
        return _run_read_file(args.get("path", ""))
    if name == "list_directory":
        return _run_list_directory(args.get("path", ""))
    if name == "get_json_value":
        return _run_get_json_value(args.get("path", ""), args.get("key_path", ""))
    if name == "write_skill":
        return _run_write_skill(args.get("filename", ""), args.get("content", ""), args.get("mode", ""))
    if name == "run_bos_token_scan":
        return _run_bos_token_scan(args)
    return f"ERROR: unknown tool '{name}'"


def _parse_finish_arguments(raw_args: str) -> Dict[str, Any]:
    try:
        args = json.loads(raw_args)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("finish arguments must be valid JSON") from exc
    if not isinstance(args, dict):
        raise ValueError("finish arguments must be a JSON object")
    if args.get("action") not in {"retry", "stop"}:
        raise ValueError("finish.action must be 'retry' or 'stop'")
    if "harness" in args and args["harness"] is not None and not isinstance(args["harness"], dict):
        raise ValueError("finish.harness must be a JSON object")
    return args


_OUTPUT_INTERVENTION_KEYS = {
    "custom_steering_prompts",
    "intervention_scope",
    "max_activation_scale",
    "last_token_scale",
    "top_k",
    "skip_observation_and_design",
}


def _normalize_finish_for_mode(args: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Normalize a finish proposal and enforce the active gate's rerun boundary.

    Some OpenAI-compatible providers do not enforce the JSON schema strictly. In
    particular, models may place ``rerun_steps`` inside ``harness`` or propose a
    chain-only retry while Gate 2 is still failing. The runner must not execute a
    retry that cannot change the active metric.
    """
    normalized = dict(args)
    harness = dict(normalized.get("harness") or {})

    nested_rerun_steps = harness.pop("rerun_steps", None)
    if normalized.get("rerun_steps") is None and nested_rerun_steps is not None:
        normalized["rerun_steps"] = nested_rerun_steps

    if normalized.get("action") == "retry" and mode == "output_validate":
        has_output_change = bool(harness.get("extra_output_guidance")) or any(
            harness.get(key) not in (None, "", [], {}) for key in _OUTPUT_INTERVENTION_KEYS
        )

        # A common malformed proposal describes how to rewrite the output
        # hypothesis but stores that instruction under the chain-only field.
        # Preserve the instruction while routing it to Step 6, where it can
        # actually affect Gate 2.
        if not has_output_change and harness.get("extra_chain_guidance"):
            harness["extra_output_guidance"] = harness.pop("extra_chain_guidance")
            has_output_change = True

        if not has_output_change:
            raise ValueError(
                "Gate 2 is failing, so a retry must change output generation or "
                "intervention settings; a chain-only retry cannot improve output_score"
            )

        required_start = 5 if any(
            harness.get(key) not in (None, "", [], {}) for key in _OUTPUT_INTERVENTION_KEYS
        ) else 6
        steps = normalized.get("rerun_steps")
        required_steps = set(range(required_start, 10))
        if not isinstance(steps, list) or not required_steps.issubset(steps):
            normalized["rerun_steps"] = list(range(required_start, 10))

    normalized["harness"] = harness
    return normalized


# ── System prompt ──────────────────────────────────────────────────────────────

def _build_system_prompt(mode: str, *, input_source: Optional[str] = None) -> str:
    skills_dir = _active_skills_dir()
    skill_learning = _skill_learning_enabled()
    with shared_skill_lock(skills_dir):
        if skills_dir.exists():
            skill_names = [md.stem for md in sorted(skills_dir.glob("*.md")) if not _is_skill_archive_file(md)]
            skills_index = ", ".join(f"skills/{n}.md" for n in skill_names) if skill_names else "(none)"
        else:
            skills_index = "(none)"

        # Embed the primary skill for this mode so agent doesn't spend a tool call reading it
        skill_for_mode = {
            "input_validate": "skill_input.md",
            "output_validate": "skill_output.md",
            "chain": "skill_chain.md",
        }
        embedded_skill = ""
        skill_filename = skill_for_mode.get(mode)
        if skill_filename:
            skill_path = skills_dir / skill_filename
            if skill_path.exists():
                content = skill_path.read_text(encoding="utf-8")
                embedded_skill = f"\n\n## Reference: {skill_filename} (pre-loaded)\n{content}"

    if skill_learning:
        skill_update_guidance = "If evidence contradicts a skill, trust the current evidence and update the skill."
        skill_write_guidance = "If previous rounds show a pattern worth documenting, call write_skill() before finish()."
        skill_tool_row = "| write_skill(...) | Record evidence: framework files for general rules, cases files for specific instances. |\n"
        skill_final_guidance = "One discovery per write_skill() call. Call it before finish()."
    else:
        skill_update_guidance = (
            "If evidence contradicts a skill, trust the current evidence for this feature, "
            "but do not update skills in this run."
        )
        skill_write_guidance = "Skill learning is disabled: do not write or update skills; call finish() directly after deciding."
        skill_tool_row = ""
        skill_final_guidance = "Skill learning is disabled for this run; finish without writing skills."

    if input_source == "bos_token":
        source_guidance = (
            "This run is permanently BOS-token initialized. Do not request or infer Neuronpedia evidence. "
            "You may inspect only this feature's initial_observation/ directory. If the current BOS tokens "
            "are incoherent or under-specific, call run_bos_token_scan before changing the hypothesis; "
            "select useful fresh evidence with harness.bos_prompt_id."
        )
        bos_tool_row = (
            '| run_bos_token_scan(...) | Design a feature-local BOS template scan; use its returned prompt_id. |\n'
        )
    else:
        source_guidance = (
            "This run is permanently Neuronpedia initialized. The BOS-token scan tool and BOS observations "
            "are unavailable."
        )
        bos_tool_row = ""

    return f"""You are an expert SAE (Sparse Autoencoder) feature interpretation expert with file system access.

Your task: figure out why this feature interpretation is failing and propose a concrete fix for the next atomic workflow retry.

## Input observation source
source={input_source or 'neuronpedia'}
{source_guidance}

## How to work

### Phase 1 - Read and reason independently
Read trace.json first. Based on the numbers, form your own diagnostic hypothesis before reading anything else.
Ask:
- Which gate is failing, and by how much?
- What does this pattern suggest about the root cause?
- Which one or two round files answer the key question?

### Phase 2 - Gather targeted evidence
Read only the files that answer your specific question.

Key questions and files:
- Gate 1 input failure: read round_4/*-step4-input-experiment-scores.json for per-hypothesis score_non_zero_rate, boundary rate, per-sentence max_token, summary_activation, and is_non_zero.
- Hypothesis or sentence text: read round_2/*-step2-input-hypotheses.json and round_3/*-step3-input-experiments.json.
- Raw Neuronpedia/BOS evidence: read round_1/*-observation-input.json or trace.json -> input_round.observation.input_top_activations.
- Gate 2 output failure: read round_7/*-step7-output-hypothesis-scores.json for per-hypothesis output_score (or final_score in older traces), matched_tokens, and match_reason.
- Intervention evidence: read round_5/*-step5-intervention-results.json for token_change_by_hypothesis, top positive tokens, KL, and intervention settings.
- Gate 3 chain failure: read round_8/*-step8-chain-hypotheses.json for chain_judge_score and judge reasons.
- Final concise explanation: read round_9/*-step9-chain-explanation.json.
- Previous retry comparison: list logs/layer-X/feature-Y, then compare the same step file across TIMESTAMP, TIMESTAMP_r1, TIMESTAMP_r2, etc.

Stop reading once your hypothesis is confirmed or you have enough evidence to propose a fix.

### Phase 3 - Cross-check skill knowledge when useful
Skills are a two-layer reference library:
  Framework (always read first): {skills_index}
  Cases (read only when the framework doesn't match your pattern):
    skills/skill_input_cases.md, skills/skill_output_cases.md, skills/skill_chain_cases.md

Read the relevant framework file first. Only open a cases file if you need specific worked examples. If a case materially influences your harness, include its case_id in finish(..., used_skill_cases=[...]).
{skill_update_guidance}

### Phase 4 - Propose and update knowledge
Call finish() with a diagnosis that cites:
- Specific metric values.
- Which hypotheses/sentences/tokens failed.
- What harness change you propose and why.

{skill_write_guidance}

## Target metrics, fixed in order
Gate 1 (INPUT, fix first):
  - input_round.eval.overall_activation_rate >= 0.8
  - input_round.eval.overall_boundary_non_activation_rate >= 0.8

Gate 2 (OUTPUT, fix after Gate 1 passes):
  - best output score >= 0.5, from output_round.per_hypothesis[*].output_score

Gate 3 (CHAIN, optimize only when Gates 1+2 both pass):
  - chain.pairs[best].chain_judge_score >= 4/5

Do NOT propose chain-specific fixes while input/output gates are failing.

## Fast file map
| File | When to use it |
|------|----------------|
| read_file("logs/.../trace.json") | Always first. Gets overall gate metrics and final chain summary. |
| read_file("logs/.../round_1/*-observation-input.json") | Raw observation tokens and examples. |
| read_file("logs/.../round_2/*-step2-input-hypotheses.json") | Input hypotheses. |
| read_file("logs/.../round_3/*-step3-input-experiments.json") | Designed activation and boundary sentences. |
| read_file("logs/.../round_4/*-step4-input-experiment-scores.json") | Gate 1 per-sentence and per-hypothesis input scores. |
| read_file("logs/.../round_5/*-step5-intervention-results.json") | Steering prompts, intervention settings, token_change evidence. |
| read_file("logs/.../round_6/*-step6-output-hypotheses.json") | Output hypotheses generated from intervention evidence. |
| read_file("logs/.../round_7/*-step7-output-hypothesis-scores.json") | Gate 2 token-match scores. |
| read_file("logs/.../round_8/*-step8-chain-hypotheses.json") | Gate 3 chain judge scores and reasons. |
| get_json_value(path, key_path) | Fast metric lookup, e.g. input_round.eval.overall_activation_rate. |
{bos_tool_row}
{skill_tool_row}
## Efficient re-run strategy
When proposing a fix via finish(), specify rerun_steps to only re-run the minimum necessary steps.
Step 9 is always appended automatically if missing.

**COST RULE: If Gate 1 activation_rate < 0.6, use rerun_steps=[2,3,4] only.**
The pipeline automatically detects Gate 1 failure after step 4 and writes a partial trace — steps 5-9 are skipped automatically. This saves GPU (step 5) and output LLM calls (steps 6-9).
Only use [2,3,4,5,6,7,8,9] when you are confident Gate 1 will pass this round (activation_rate >= 0.6 in prior round).

| Fix target | rerun_steps | Why |
|------------|-------------|-----|
| Fix input hypotheses — Gate 1 < 0.6, Gate 2 was good (>= 0.5) | [2,3,4,8,9] | Skip GPU step 5 + output steps 6-7; inherit old Gate 2 data; redo chain with new Gate 1 + old output |
| Fix input hypotheses — Gate 1 < 0.6, Gate 2 also failing | [2,3,4,5,6,7,8,9] | Full rerun; if Gate 1 still fails at step 4, pipeline auto-skips 5-7 and runs 8-9 with inherited data |
| Fix input hypotheses — Gate 1 >= 0.6 (likely to pass) | [2,3,4,5,6,7,8,9] | Run full pipeline |
| Fix experiment sentences only | [3,4,5,6,7,8,9] | Sentences affect scoring + intervention |
| Re-score input only (hypotheses + sentences OK) | [4] | Just re-run scoring, skip GPU steps 5-9 |
| Fix intervention params (steering, scope, scale) | [5,6,7,8,9] | Intervention affects output hypotheses onward |
| Fix output hypotheses (extra_output_guidance) | [6,7,8,9] | Only re-run output side |
| Fix chain judgment only | [8,9] | Minimal: just re-judge the chain |

## Harness parameters
| Parameter | Effect |
|-----------|--------|
| skip_observation_and_design | Reuse step1-step4 and rerun step5-step9. |
| extra_input_guidance | Rerun from step2; appended to input hypothesis generation. |
| bos_prompt_id | BOS only; select fresh evidence and force rerun of steps 1-4. |
| extra_output_guidance | Rerun from step6; appended to output hypothesis generation. |
| custom_steering_prompts | Rerun from step5; replaces designed sentences as steering context. |
| intervention_scope | Rerun from step5; controls steered token positions. |
| max_activation_scale | Rerun from step5; scales max-activation-token intervention strength. |
| last_token_scale | Rerun from step5; scales last-token-only intervention. |
| top_k | Rerun from step5; controls number of token deltas recorded. |

{skill_final_guidance}
{embedded_skill}"""


# ── Mode helpers ───────────────────────────────────────────────────────────────

def _load_trace(trace_path_str: str) -> Optional[Dict[str, Any]]:
    p = _safe_path(trace_path_str)
    if p is None or not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _metrics_from_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    m = extract_trace_metrics(trace)
    g1_act = m["input_activation_rate"]
    g1_bnd = m["input_boundary_non_activation_rate"]
    g2_best = round(m["output_score"], 4)
    g3_best = int(m["best_chain_score"])
    return {
        "gate1_activation_rate": g1_act,
        "gate1_boundary_rate": g1_bnd,
        "gate1_pass": g1_act >= 0.8 and g1_bnd >= 0.8,
        "gate2_best_score": g2_best,
        "gate2_pass": g2_best >= 0.5,
        "gate3_best_score": g3_best,
        "gate3_pass": g3_best >= 4,
    }


def _extract_current_metrics(trace_path_str: str) -> Optional[Dict[str, Any]]:
    trace = _load_trace(trace_path_str)
    return _metrics_from_trace(trace) if trace is not None else None


def _harness_targeted_gate(harness: Optional[Dict[str, Any]]) -> str:
    """Infer which side (input/output/chain) a harness config was targeting."""
    if not harness:
        return "unknown"
    if harness.get("extra_input_guidance"):
        return "input"
    if harness.get("extra_chain_guidance"):
        return "chain"
    output_keys = ("intervention_scope", "max_activation_scale", "last_token_scale",
                   "custom_steering_prompts", "extra_output_guidance", "top_k",
                   "skip_observation_and_design")
    if any(harness.get(k) for k in output_keys):
        return "output"
    return "unknown"


def _determine_mode(
    prev_harness: Optional[Dict[str, Any]],
    metrics: Optional[Dict[str, Any]],
) -> str:
    """
    input_validate  — Gate 1 is the active problem
    output_validate — Gate 1 passes but Gate 2 fails
    chain           — Gates 1+2 pass; focus on Gate 3 / chain quality

    Metrics-first: always use the current gate state to determine mode so that
    when a gate transitions from failing → passing the mode advances immediately.
    Prev harness is only consulted as a tiebreaker when metrics are unavailable.
    """
    if metrics:
        if not metrics["gate1_pass"]:
            return "input_validate"
        if not metrics["gate2_pass"]:
            return "output_validate"
        return "chain"
    # Fallback when metrics are unavailable
    targeted = _harness_targeted_gate(prev_harness)
    if targeted == "input":
        return "input_validate"
    if targeted == "output":
        return "output_validate"
    return "chain"


def _build_gate1_evidence(trace: Dict[str, Any]) -> str:
    """Compact per-hypothesis Gate 1 breakdown extracted from trace.json.

    Handles both new schema (input_round.eval.per_hypothesis) and old schema (input_eval.per_hypothesis).
    """
    # New schema first, fall back to old
    eval_data = ((trace.get("input_round") or {}).get("eval") or {}) or (trace.get("input_eval") or {})
    per_hyp = eval_data.get("per_hypothesis") or []
    if not per_hyp:
        return "  (no per-hypothesis Gate 1 data available)"

    lines: List[str] = []
    for h in per_hyp:
        idx = h.get("hypothesis_index", h.get("idx", "?"))
        hyp_text = str(h.get("hypothesis", ""))[:120]
        # New schema uses score_non_zero_rate; old uses activation_rate
        act = h.get("score_non_zero_rate", h.get("activation_rate", "?"))
        bnd = h.get("score_boundary_non_activation_rate", h.get("boundary_non_activation_rate", "?"))
        act_ok = isinstance(act, (int, float)) and act >= 0.8
        bnd_ok = isinstance(bnd, (int, float)) and bnd >= 0.8
        if act_ok and bnd_ok:
            tag = "PASS"
        elif not act_ok and not bnd_ok:
            tag = "ACT+BND_FAIL"
        elif not act_ok:
            tag = "ACT_FAIL"
        else:
            tag = "BND_FAIL"
        lines.append(f"\nH{idx} [{tag}] act={act}, boundary={bnd}")
        lines.append(f'  "{hyp_text}"')

        # New schema: sentence_results[].{sentence, is_non_zero}
        # Old schema: activation_samples[].{sentence, activated}
        sent_results = h.get("sentence_results") or h.get("activation_samples") or []
        failed_act = [s["sentence"][:100] for s in sent_results if not (s.get("is_non_zero") or s.get("activated"))]
        if failed_act:
            lines.append(f"  Failed activation ({len(failed_act)}):")
            for s in failed_act[:3]:
                lines.append(f"    - \"{s}\"")

        bnd_results = h.get("boundary_sentence_results") or h.get("boundary_samples") or []
        failed_bnd = [s["sentence"][:100] for s in bnd_results if (s.get("is_non_zero") or s.get("activated"))]
        if failed_bnd:
            lines.append(f"  Still-activated boundary ({len(failed_bnd)}):")
            for s in failed_bnd[:3]:
                lines.append(f"    - \"{s}\"")

    result = "\n".join(lines)
    if len(result) > _MAX_EVIDENCE_CHARS:
        result = result[:_MAX_EVIDENCE_CHARS] + "\n  [... truncated]"
    return result


def _build_gate2_evidence(trace: Dict[str, Any]) -> str:
    """Compact per-hypothesis Gate 2 breakdown extracted from trace.json.

    Handles both new schema (output_round.per_hypothesis) and old schema (output_llm_token_match_eval.per_hypothesis).
    """
    # New schema first, fall back to old
    per_hyp = (
        (trace.get("output_round") or {}).get("per_hypothesis")
        or (trace.get("output_llm_token_match_eval") or {}).get("per_hypothesis")
        or []
    )
    if not per_hyp:
        return "  (no per-hypothesis Gate 2 data available)"

    lines: List[str] = []
    for h in per_hyp:
        idx = h.get("idx", "?")
        # New schema: output_score; old schema: final_score
        score = h.get("output_score", h.get("final_score", "?"))
        ok = isinstance(score, (int, float)) and score >= 0.5
        in_hyp = str(h.get("input_hypothesis", ""))[:80]
        out_hyp = str(h.get("output_hypothesis", ""))[:80]
        # matched_tokens may be strings (new schema) or dicts with 'token_display' (old schema)
        raw_matched = (h.get("matched_tokens") or [])[:5]
        matched = [t.get("token_display", t) if isinstance(t, dict) else t for t in raw_matched]
        # New schema: match_reason; old schema: llm_reason
        reason = str(h.get("match_reason", h.get("llm_reason", "")))[:120]
        scope = h.get("intervention_scope", "?")
        # topk_positive_tokens may be strings (new schema) or dicts (old schema)
        raw_top_pos = (h.get("topk_positive_tokens") or [])[:8]
        top_pos = [t.get("token_display", t) if isinstance(t, dict) else t for t in raw_top_pos]
        lines.append(f"\nH{idx} [{'PASS' if ok else 'FAIL'}] score={score}")
        if in_hyp:
            lines.append(f'  input_hyp:  "{in_hyp}"')
        lines.append(f'  output_hyp: "{out_hyp}"')
        lines.append(f"  scope={scope}  matched={matched}  top_pos={top_pos}")
        if reason:
            lines.append(f'  reason: "{reason}"')

    result = "\n".join(lines)
    if len(result) > _MAX_EVIDENCE_CHARS:
        result = result[:_MAX_EVIDENCE_CHARS] + "\n  [... truncated]"
    return result


def _format_metrics(metrics: Optional[Dict[str, Any]]) -> str:
    if not metrics:
        return "  (metrics unavailable)"
    g1 = "PASS ✓" if metrics.get("gate1_pass") else "FAIL"
    g2 = "PASS ✓" if metrics.get("gate2_pass") else "FAIL"
    g3 = "PASS ✓" if metrics.get("gate3_pass") else "FAIL"
    return (
        f"  Gate 1  activation={metrics.get('gate1_activation_rate')}  "
        f"boundary={metrics.get('gate1_boundary_rate')}  [{g1}]\n"
        f"  Gate 2  best_output_score={metrics.get('gate2_best_score')}  [{g2}]\n"
        f"  Gate 3  best_chain_score={metrics.get('gate3_best_score')}  [{g3}]"
    )


def _extract_hypothesis_texts(trace: Optional[Dict[str, Any]]) -> List[str]:
    """Pull hypothesis strings from trace for deduplication hints."""
    if not trace:
        return []
    per_hyp = (
        ((trace.get("input_round") or {}).get("eval") or (trace.get("input_eval") or {}))
        .get("per_hypothesis") or []
    )
    texts = [str(h.get("hypothesis", "")).strip() for h in per_hyp if h.get("hypothesis")]
    # Also try top-level hypotheses list
    if not texts:
        texts = [str(h).strip() for h in (trace.get("input_round") or {}).get("hypotheses", []) if h]
    return [t for t in texts if t][:3]


def _build_user_message(
    *,
    mode: str,
    layer_id: str,
    feature_id: str,
    round_idx: int,
    prev_timestamps: List[str],
    prev_harness: Optional[Dict[str, Any]],
    metrics: Optional[Dict[str, Any]],
    trace_path: str,
    trace: Optional[Dict[str, Any]] = None,
    prev_diagnosis: Optional[str] = None,
) -> str:
    input_source = str(_feature_context().get("input_source") or "neuronpedia")
    observation_hint = (
        "  initial_observation/ (this feature's BOS prompts and token evidence)\n"
        if input_source == "bos_token"
        else "  round_1/*-initial-observation.json (raw Neuronpedia evidence)\n"
    )
    header = (
        f"## Feature: layer={layer_id}, feature={feature_id} | agent round={round_idx}\n\n"
        f"## Current metrics\n{_format_metrics(metrics)}\n\n"
    )

    # Validate modes only need the most recent timestamp; chain mode shows full history
    display_timestamps = prev_timestamps[-1:] if mode in ("input_validate", "output_validate") else prev_timestamps
    if display_timestamps:
        prev_line = f"  Previous timestamps: {', '.join(display_timestamps)}\n"
        if prev_harness:
            prev_line += f"  Last harness change: {json.dumps(prev_harness, ensure_ascii=False)}\n"
        header += f"## History\n{prev_line}\n"

    if mode == "input_validate":
        evidence = _build_gate1_evidence(trace) if trace else "  (trace not loaded)"
        # Previous diagnosis for continuity (#2)
        diag_section = f"## Previous diagnosis\n  {prev_diagnosis[:400]}\n\n" if prev_diagnosis else ""
        # Failed hypothesis texts to avoid re-generating similar ones (#3)
        prev_hyp_texts = _extract_hypothesis_texts(trace)
        hyp_avoid = ""
        if prev_hyp_texts:
            listed = "\n".join(f'  - "{t[:120]}"' for t in prev_hyp_texts)
            hyp_avoid = (
                f"\n## Previously tried hypotheses (avoid generating similar ones)\n"
                f"{listed}\n"
                f"  → If proposing extra_input_guidance, explicitly instruct the LLM NOT to repeat "
                f"these conceptual angles.\n"
            )
        return (
            header
            + diag_section
            + f"## Gate 1 evidence (pre-extracted)\n{evidence}\n\n"
            + hyp_avoid
            + f"## Task — propose input fix\n"
            + f"Use the per-hypothesis evidence above. "
            + f"The skill reference (skill_input.md) is pre-loaded in your instructions.\n\n"
            + f"Decision rules:\n"
            + f"  Gate 1 now passes AND Gate 2 passes AND chain_judge_score >= 4 → call finish(action='stop').\n"
            + f"  Gate 1 now passes AND Gate 2 passes AND chain_judge_score < 4 → propose chain fix "
            + f"(use extra_chain_guidance with rerun_steps=[8,9]).\n"
            + f"  Gate 1 now passes AND Gate 2 fails → propose output fix.\n"
            + f"  Gate 1 still fails → propose a new input fix using the evidence above.\n\n"
            + f"If you need raw observation tokens or hypothesis text, read:\n"
            + f"  {trace_path} (full trace)\n"
            + observation_hint
            + f"  round_2/*-input-hypotheses.json (full hypothesis text)\n"
            + f"Call finish() as soon as you have enough evidence.\n"
        )

    if mode == "output_validate":
        evidence = _build_gate2_evidence(trace) if trace else "  (trace not loaded)"
        diag_section = f"## Previous diagnosis\n  {prev_diagnosis[:400]}\n\n" if prev_diagnosis else ""
        return (
            header
            + diag_section
            + f"## Gate 2 evidence (pre-extracted)\n{evidence}\n\n"
            + f"## Task — propose output fix\n"
            + f"Use the per-hypothesis evidence above. "
            + f"The skill reference (skill_output.md) is pre-loaded in your instructions.\n\n"
            + f"Decision rules:\n"
            + f"  Gate 2 now passes AND chain_judge_score >= 4 → call finish(action='stop').\n"
            + f"  Gate 2 now passes AND chain_judge_score < 4 → propose chain fix "
            + f"(use extra_chain_guidance with rerun_steps=[8,9]).\n"
            + f"  Gate 2 still fails → propose a new intervention/output fix.\n\n"
            + f"If you need intervention details or steering prompts, read:\n"
            + f"  round_5/*-intervention-results.json\n"
            + f"Call finish() as soon as you have enough evidence.\n"
        )

    # chain mode: Gates 1+2 pass; focus on Gate 3 (chain_judge_score >= 4)
    return (
        header
        + f"## Task — optimize chain score (Gate 3)\n"
        + f"Gates 1+2 already pass. Focus EXCLUSIVELY on improving chain_judge_score to >= 4.\n\n"
        + f"Read trace.json first to get the current chain judge scores and reasons.\n"
        + f"  trace.json : {trace_path}\n"
        + f"  skills/    : skills/\n\n"
        + f"Key sections:\n"
        + f"  chain.pairs[*].chain_judge_score   — Gate 3 (target >=4/5)\n"
        + f"  chain.pairs[*].chain_judge_reason  — why the judge scored as it did\n"
        + f"  chain.pairs[*].input_hypothesis    — what the agent thinks the feature detects\n"
        + f"  chain.pairs[*].output_hypothesis   — what output effect the feature causes\n"
        + f"  output_round.per_hypothesis[*].output_score  — Gate 2 score (must stay >= 0.5 after fix)\n\n"
        + f"Fix options (in order of preference):\n"
        + f"  1. extra_chain_guidance + rerun_steps=[8,9]: inject guidance into step8 chain-judge to help "
        + f"the LLM form a stronger causal narrative. Use when judge_reason shows it missed the key link.\n"
        + f"  2. extra_output_guidance + rerun_steps=[6,7,8,9]: improve output hypothesis to better match "
        + f"the intervention evidence. Use when output_hypothesis is vague or mismatched.\n"
        + f"  3. custom_steering_prompts + rerun_steps=[5,6,7,8,9]: replace steering prompts to produce "
        + f"cleaner token-change evidence for the chain judge.\n\n"
        + f"IMPORTANT: output_score must not drop below 0.5. Check current output_score before proposing.\n"
        + f"If the feature is polysemantic (input and output have unrelated domains) → call finish(action='stop').\n"
        + f"Call finish() as soon as you have a clear fix plan.\n"
    )


def propose_harness(
    *,
    layer_id: str,
    feature_id: str,
    current_timestamp: str,
    round_idx: int,
    prev_timestamps: List[str],
    prev_harness: Optional[Dict[str, Any]] = None,
    prev_diagnosis: Optional[str] = None,
    client: OpenAI,
    model: str,
    token_counter: TokenUsageAccumulator,
    override_mode: Optional[str] = None,
    skills_dir: Optional[Path] = None,
    skill_learning_enabled: bool = True,
    skill_maintenance_mode: str = "off",
    skill_maintenance_archive_dir: Optional[Path] = None,
    logs_root: Optional[Path] = None,
    input_source: str = "neuronpedia",
    bos_token_root: Optional[Path] = None,
    bos_prompt_id: Optional[str] = None,
    sae_path: Optional[str] = None,
    model_path: Optional[str] = None,
    device: str = "cpu",
    sae_width: str = "16k",
    inference_server_url: str = "http://127.0.0.1:8008",
    inference_timeout_sec: float = 600.0,
    no_inference_server: bool = False,
) -> Dict[str, Any]:
    """
    Mode-aware tool-use agent loop.

    - chain:           Gates 1+2 pass; focus on chain score
    - input_validate:  Gate 1 is the active problem — evidence pre-injected
    - output_validate: Gate 2 is the active problem — evidence pre-injected
    - override_mode:   Force a specific mode (used when Gate 1 is stuck and we want to skip ahead)
    """
    skills_token = _CURRENT_SKILLS_DIR.set(Path(skills_dir) if skills_dir is not None else SKILLS_DIR)
    logs_token = _CURRENT_LOGS_ROOT.set(Path(logs_root) if logs_root is not None else Path(__file__).parent / "logs")
    learning_token = _SKILL_LEARNING_ENABLED.set(bool(skill_learning_enabled))
    tool_events: List[Dict[str, Any]] = []
    events_token = _CURRENT_TOOL_EVENTS.set(tool_events)
    maintenance_archive = (
        Path(skill_maintenance_archive_dir)
        if skill_maintenance_archive_dir is not None
        else (Path(logs_root) if logs_root is not None else Path(__file__).parent / "logs")
        / "skill_case_archive"
    )
    maintenance_token = _CURRENT_SKILL_MAINTENANCE.set(
        {
            "mode": str(skill_maintenance_mode),
            "client": client,
            "model": model,
            "archive_dir": maintenance_archive,
            "maintenance_name": f"run-{current_timestamp}",
        }
    )
    resolved_observation_dir = None
    if input_source == "bos_token" and bos_token_root is not None:
        resolved_observation_dir = (
            Path(bos_token_root)
            / f"layer-{int(layer_id)}"
            / f"feature-{int(feature_id)}"
            / "bos_token"
        )
    observation_token = _CURRENT_FEATURE_OBSERVATION_DIR.set(resolved_observation_dir)
    context_token = _CURRENT_FEATURE_CONTEXT.set({
        "layer_id": str(layer_id),
        "feature_id": str(feature_id),
        "current_timestamp": str(current_timestamp),
        "round_idx": int(round_idx),
        "input_source": str(input_source),
        "bos_token_root": str(bos_token_root) if bos_token_root is not None else None,
        "bos_prompt_id": str(bos_prompt_id) if bos_prompt_id else None,
        "sae_path": str(sae_path) if sae_path else None,
        "model_path": str(model_path) if model_path else None,
        "device": str(device),
        "sae_width": str(sae_width),
        "inference_server_url": str(inference_server_url),
        "inference_timeout_sec": float(inference_timeout_sec),
        "no_inference_server": bool(no_inference_server),
    })
    try:
        _result = _propose_harness_inner(
            layer_id=layer_id,
            feature_id=feature_id,
            current_timestamp=current_timestamp,
            round_idx=round_idx,
            prev_timestamps=prev_timestamps,
            prev_harness=prev_harness,
            prev_diagnosis=prev_diagnosis,
            client=client,
            model=model,
            token_counter=token_counter,
            override_mode=override_mode,
        )
        _result["tool_events"] = list(tool_events)
        return _result
    finally:
        _CURRENT_FEATURE_CONTEXT.reset(context_token)
        _CURRENT_FEATURE_OBSERVATION_DIR.reset(observation_token)
        _CURRENT_SKILL_MAINTENANCE.reset(maintenance_token)
        _CURRENT_TOOL_EVENTS.reset(events_token)
        _SKILL_LEARNING_ENABLED.reset(learning_token)
        _CURRENT_LOGS_ROOT.reset(logs_token)
        _CURRENT_SKILLS_DIR.reset(skills_token)


def _propose_harness_inner(
    *,
    layer_id: str,
    feature_id: str,
    current_timestamp: str,
    round_idx: int,
    prev_timestamps: List[str],
    prev_harness: Optional[Dict[str, Any]] = None,
    prev_diagnosis: Optional[str] = None,
    client: OpenAI,
    model: str,
    token_counter: TokenUsageAccumulator,
    override_mode: Optional[str] = None,
) -> Dict[str, Any]:
    trace_path = f"logs/layer-{layer_id}/feature-{feature_id}/{current_timestamp}/trace.json"

    trace = _load_trace(trace_path)
    metrics = _metrics_from_trace(trace) if trace is not None else None
    mode = override_mode if override_mode else _determine_mode(prev_harness, metrics)

    effective_max_tool_calls = _MODE_MAX_TOOL_CALLS.get(mode, DEFAULT_MAX_TOOL_CALLS)
    thinking_budget = _MODE_THINKING_BUDGET.get(mode, 3000)

    user_msg = _build_user_message(
        mode=mode,
        layer_id=layer_id,
        feature_id=feature_id,
        round_idx=round_idx,
        prev_timestamps=prev_timestamps,
        prev_harness=prev_harness,
        metrics=metrics,
        trace_path=trace_path,
        trace=trace,
        prev_diagnosis=prev_diagnosis,
    )

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": _build_system_prompt(
                mode,
                input_source=str(_feature_context().get("input_source") or "neuronpedia"),
            ),
        },
        {"role": "user", "content": user_msg},
    ]

    tool_call_count = 0
    skills_written: List[str] = []

    while tool_call_count < effective_max_tool_calls:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=_tools_for_current_context(),
            tool_choice="auto",
            max_tokens=8000,
        )
        token_counter.add(getattr(response, "usage", None))

        choice = response.choices[0]
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None) or []

        # Serialize assistant message back into history
        msg_dict: Dict[str, Any] = {"role": "assistant"}
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            msg_dict["reasoning_content"] = reasoning
        content = getattr(msg, "content", None)
        msg_dict["content"] = content or ""
        if tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        messages.append(msg_dict)

        # Check for finish call first. Invalid JSON is returned to the model as
        # a tool error so it can correct the call on the next turn.
        for tc in tool_calls:
            if tc.function.name == "finish":
                try:
                    args = _parse_finish_arguments(tc.function.arguments)
                    args = _normalize_finish_for_mode(args, mode)
                except ValueError as exc:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"ERROR: could not parse finish arguments: {exc}. Return valid finish JSON.",
                    })
                    tool_call_count += 1
                    continue
                return {
                    "action": args.get("action", "stop"),
                    "diagnosis": args.get("diagnosis", ""),
                    "harness": args.get("harness") or {},
                    "rerun_steps": args.get("rerun_steps"),
                    "used_skill_cases": args.get("used_skill_cases") or [],
                    "tool_calls_made": tool_call_count,
                    "skills_written": skills_written,
                    "mode": mode,
                }
        # No tool calls: force the model to convert its plain-text answer into finish(...).
        if not tool_calls:
            raw = content or ""
            messages.append({
                "role": "user",
                "content": (
                    "You responded in plain text instead of calling the finish tool. "
                    "Convert your previous answer into a finish(...) tool call now. "
                    "If your text says retry or proposes a fix, use action='retry' and include "
                    "a concrete harness plus rerun_steps. If your text says stop, use action='stop'. "
                    "Do not answer in plain text."
                ),
            })
            finish_response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=_tools_for_current_context(),
                tool_choice={"type": "function", "function": {"name": "finish"}},
                max_tokens=4000,
            )
            token_counter.add(getattr(finish_response, "usage", None))
            finish_msg = finish_response.choices[0].message
            finish_tool_calls = getattr(finish_msg, "tool_calls", None) or []
            for tc in finish_tool_calls:
                if tc.function.name == "finish":
                    try:
                        args = _parse_finish_arguments(tc.function.arguments)
                        args = _normalize_finish_for_mode(args, mode)
                    except ValueError as exc:
                        messages.append({
                            "role": "assistant",
                            "content": getattr(finish_msg, "content", None) or "",
                            "tool_calls": [{
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }],
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": f"ERROR: could not parse finish arguments: {exc}. Return valid finish JSON.",
                        })
                        tool_call_count += 1
                        break
                    return {
                        "action": args.get("action", "stop"),
                        "diagnosis": args.get("diagnosis", ""),
                        "harness": args.get("harness") or {},
                        "rerun_steps": args.get("rerun_steps"),
                        "used_skill_cases": args.get("used_skill_cases") or [],
                        "tool_calls_made": tool_call_count,
                        "skills_written": skills_written,
                        "mode": mode,
                        "forced_finish_from_plain_text": True,
                    }
            else:
                return {
                    "action": "stop",
                    "diagnosis": (
                        "agent finished without calling finish(), and forced finish() did not return "
                        f"a finish tool call. Last content: {raw[:300]}"
                    ),
                    "harness": {},
                    "rerun_steps": None,
                    "used_skill_cases": [],
                    "tool_calls_made": tool_call_count,
                    "skills_written": skills_written,
                    "mode": mode,
                }
            continue

        # Execute non-finish tool calls and append results
        for tc in tool_calls:
            if tc.function.name == "finish":
                continue
            result = _dispatch(tc.function.name, tc.function.arguments)
            if tc.function.name == "write_skill" and result.startswith("OK:"):
                try:
                    _ws_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    skills_written.append(_ws_args.get("filename", "unknown"))
                except Exception:
                    pass
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            tool_call_count += 1

    return {
        "action": "stop",
        "diagnosis": f"agent reached max_tool_calls={effective_max_tool_calls} without calling finish()",
        "harness": {},
        "rerun_steps": None,
        "used_skill_cases": [],
        "tool_calls_made": tool_call_count,
        "skills_written": skills_written,
        "mode": mode,
    }
