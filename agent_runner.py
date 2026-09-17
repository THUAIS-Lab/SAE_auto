"""
agent_runner.py — Phase 3 outer loop for one SAE feature.

Loads an existing trace.json (from the fixed pipeline), then iteratively:
  1. Checks if the score meets the threshold (early stop).
  2. Calls the LLM proposer to diagnose failure and propose a new harness config.
  3. Executes the pipeline with the proposed config (new timestamp = {initial_ts}_r{n}).
  4. Loads the new trace and repeats.

Usage:
  python agent_runner.py \\
    --layer-id 6 --feature-id 100 \\
    --initial-timestamp 20260420_175328 \\
    --sae-path /path/to/gemma-scope-2b-pt-res/layer_6/width_16k/average_l0_70 \\
    --model-checkpoint-path /path/to/gemma-2-2b \\
    --llm-base-url $LLM_BASE_URL --llm-model $LLM_MODEL \\
    --max-rounds 3 --score-threshold 4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - allows CLI help/tests without OpenAI installed
    OpenAI = Any  # type: ignore[misc,assignment]

from agent_proposer import propose_harness
from function import TokenUsageAccumulator, append_text_locked, build_feature_dir, build_round_dir, read_api_key
from trace_metrics import extract_trace_metrics
from skill_case_store import (
    append_skill_case,
    case_side_from_id_or_lookup,
    increment_case_usage,
    make_case_id,
)
from inference_client import DEFAULT_INFERENCE_SERVER_URL, DEFAULT_INFERENCE_TIMEOUT_SEC
from skill_maintenance import maybe_run_auto_skill_maintenance
from support_info.llm_api_info import (
    api_key_file as DEFAULT_API_KEY_FILE,
    base_url as DEFAULT_BASE_URL,
    model_name as DEFAULT_MODEL_NAME,
)

PYTHON = sys.executable
CODE_DIR = Path(__file__).parent

# Below this combined activation+boundary improvement, consider the gate "stuck"
_GATE1_STUCK_DELTA = 0.05

# Step-output filename suffixes (mirrors workflow_step_utils.STEP_OUTPUT_NAMES)
_STEP_OUTPUT_NAMES = {
    1: "initial-observation",
    2: "input-hypotheses",
    4: "input-experiment-scores",
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def _log(msg: str, log_file: Optional[Path] = None) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_file:
        append_text_locked(log_file, line + "\n")


def _collect_skill_tool_events(agent_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for entry in agent_log:
        proposal = entry.get("proposal") if isinstance(entry, dict) else None
        if not isinstance(proposal, dict):
            continue
        for raw_event in proposal.get("tool_events") or []:
            if not isinstance(raw_event, dict):
                continue
            name = str(raw_event.get("event") or "")
            if not (name.startswith("skill") or name.startswith("case")):
                continue
            event = dict(raw_event)
            event.setdefault("agent_log_round", entry.get("round"))
            events.append(event)
    return events


def _run_step(cmd: List[str], *, env: Dict[str, str], label: str, log_file: Optional[Path]) -> bool:
    _log(f"{label} START", log_file)
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=str(CODE_DIR), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert proc.stdout is not None
    if log_file:
        for line in proc.stdout:
            print(line, end="", flush=True)
            append_text_locked(log_file, line)
    else:
        for line in proc.stdout:
            print(line, end="", flush=True)
    proc.wait()
    elapsed = time.time() - t0
    if proc.returncode != 0:
        _log(f"{label} FAILED ({elapsed:.1f}s)", log_file)
        return False
    _log(f"{label} OK ({elapsed:.1f}s)", log_file)
    return True




_ALLOWED_INTERVENTION_SCOPES = {"last_token_only", "all_tokens", "max_activation_token"}
_INPUT_HARNESS_KEYS = ("extra_input_guidance", "bos_prompt_id")
_OUTPUT_HARNESS_KEYS = (
    "intervention_scope",
    "max_activation_scale",
    "last_token_scale",
    "custom_steering_prompts",
    "extra_output_guidance",
    "top_k",
    "skip_observation_and_design",
)
_CHAIN_HARNESS_KEYS = ("extra_chain_guidance",)


def _harness_targeted_input_or_output(harness: Dict[str, Any]) -> str:
    """Return 'input', 'output', 'chain', or 'unknown' based on which params were set."""
    if any(k in harness for k in _INPUT_HARNESS_KEYS):
        return "input"
    if any(k in harness for k in _CHAIN_HARNESS_KEYS):
        return "chain"
    if any(k in harness for k in _OUTPUT_HARNESS_KEYS):
        return "output"
    return "unknown"


def _normalize_intervention_scope(value: Any) -> str:
    text = str(value if value is not None else "max_activation_token").strip().rstrip(",")
    return text if text in _ALLOWED_INTERVENTION_SCOPES else "max_activation_token"


def _normalize_top_k(value: Any, default: int = 30) -> int:
    """Return a positive integer top-k, falling back for null or invalid LLM values."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


def _load_step4_gate1(
    layer_id: str,
    feature_id: str,
    timestamp: str,
    logs_root: Optional[Path] = None,
) -> Optional[tuple]:
    """Return (act_rate, bnd_rate) from the step 4 output file, or None if unreadable."""
    path = (
        build_round_dir(layer_id=layer_id, feature_id=feature_id, timestamp=timestamp, round_id="input", logs_root=logs_root)
        / f"layer{layer_id}-feature{feature_id}-step4-input-experiment-scores.json"
    )
    if not path.exists():
        return None
    try:
        out = json.loads(path.read_text(encoding="utf-8")).get("outputs", {})
        act = out.get("overall_score_non_zero_rate", out.get("overall_activation_rate"))
        bnd = out.get("overall_score_boundary_non_activation_rate", out.get("overall_boundary_non_activation_rate"))
        if act is None or bnd is None:
            return None
        return float(act), float(bnd)
    except Exception:
        return None


def _write_partial_trace(
    *,
    layer_id: str,
    feature_id: str,
    timestamp: str,
    log_file: Optional[Path],
    logs_root: Optional[Path] = None,
) -> bool:
    """Write a stub trace.json from steps 1-4, skipping the GPU-heavy steps 5-9.
    Lets the agent diagnose Gate 1 failure without wasting compute on output steps."""
    ts_dir = build_round_dir(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=timestamp,
        round_id="input",
        logs_root=logs_root,
    ).parent

    def _read_outputs(step_index: int) -> Dict[str, Any]:
        name = _STEP_OUTPUT_NAMES.get(step_index, "")
        phase = "input" if int(step_index) <= 4 else "output" if int(step_index) <= 7 else "chain"
        path = ts_dir / phase / f"layer{layer_id}-feature{feature_id}-step{step_index}-{name}.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("outputs", {})
        except Exception:
            return {}

    try:
        eval_data = _read_outputs(4)
        step2_out = _read_outputs(2)
        hypotheses = step2_out.get("hypotheses", step2_out.get("input_hypotheses", []))
        observation = _read_outputs(1)

        trace = {
            "meta": {
                "layer_id": layer_id,
                "feature_id": feature_id,
                "timestamp": timestamp,
                "gate1_early_stop": True,
                "token_cost": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
            "input_round": {
                "observation": observation,
                "hypotheses": hypotheses if isinstance(hypotheses, list) else [],
                "eval": eval_data,
            },
            "output_round": {"per_hypothesis": []},
            "chain": {
                "best_pair_idx": 0,
                "chain_explanation": "(gate1_early_stop: steps 5-9 skipped)",
                "pairs": [],
            },
        }
        trace_path = ts_dir / "trace.json"
        trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _log(f"[gate1_early_stop] Wrote partial trace → {trace_path}", log_file)
        return True
    except Exception as e:
        _log(f"[gate1_early_stop] Failed to write partial trace: {e}", log_file)
        return False


def _should_stop(
    trace: Dict[str, Any],
    score_threshold: int,
    input_activation_threshold: float,
    input_boundary_threshold: float,
    output_score_threshold: float,
) -> bool:
    m = extract_trace_metrics(trace)
    return (
        m["best_chain_score"] >= float(score_threshold)
        and m["input_activation_rate"] >= float(input_activation_threshold)
        and m["input_boundary_non_activation_rate"] >= float(input_boundary_threshold)
        and m["output_score"] >= float(output_score_threshold)
    )


def _rank_key(
    trace: Dict[str, Any],
    score_threshold: int,
    input_activation_threshold: float,
    input_boundary_threshold: float,
    output_score_threshold: float,
) -> tuple:
    """Return a comparison key: (gates_passed, chain_score, output_score, act+bnd).

    Higher is better. Used to select the best round across all attempts.
    Tiebreakers after gates and chain_score: output_score, then avg activation+boundary.
    """
    m = extract_trace_metrics(trace)
    gate1 = (m["input_activation_rate"] >= input_activation_threshold
             and m["input_boundary_non_activation_rate"] >= input_boundary_threshold)
    gate2 = m["output_score"] >= output_score_threshold
    gate3 = m["best_chain_score"] >= score_threshold
    gates_passed = sum([gate1, gate2, gate3])
    avg_input = (m["input_activation_rate"] + m["input_boundary_non_activation_rate"]) / 2
    return (gates_passed, m["best_chain_score"], m["output_score"], avg_input)


# Phase → step range mapping
_STEP_PHASE = {s: "input" for s in range(1, 5)}
_STEP_PHASE.update({s: "output" for s in range(5, 8)})
_STEP_PHASE.update({s: "chain" for s in range(8, 10)})


def _copy_phases(
    layer_id: str, feature_id: str,
    from_ts: str, to_ts: str,
    phases: List[str],
    logs_root: Optional[Path] = None,
) -> None:
    for phase in phases:
        src = build_round_dir(layer_id=layer_id, feature_id=feature_id, timestamp=from_ts, round_id=phase, logs_root=logs_root)
        dst = build_round_dir(layer_id=layer_id, feature_id=feature_id, timestamp=to_ts, round_id=phase, logs_root=logs_root)
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(src), str(dst))


def _earliest_rerun_step(harness: Dict[str, Any]) -> int:
    if not harness:
        return 1

    start_step: Optional[int] = None

    if harness.get("extra_input_guidance") or harness.get("bos_prompt_id"):
        start_step = 2

    step5_trigger = bool(harness.get("skip_observation_and_design", False))
    step5_trigger = step5_trigger or any(
        key in harness and harness.get(key) not in (None, "", [], {})
        for key in (
            "custom_steering_prompts",
            "intervention_scope",
            "max_activation_scale",
            "last_token_scale",
            "top_k",
        )
    )
    if step5_trigger:
        start_step = min(start_step, 5) if start_step is not None else 5

    if harness.get("extra_output_guidance"):
        start_step = min(start_step, 6) if start_step is not None else 6

    if harness.get("extra_chain_guidance"):
        start_step = min(start_step, 8) if start_step is not None else 8

    return start_step if start_step is not None else 1


def _resolve_rerun_steps(
    harness: Dict[str, Any],
    rerun_steps: Optional[List[int]],
) -> List[int]:
    if rerun_steps:
        steps = sorted(set(rerun_steps))
        # Auto-append step 9 only if any output-side step is included
        # (input-only iterations write partial trace instead)
        if 9 not in steps and any(s >= 5 for s in steps):
            steps.append(9)
        return steps
    start = _earliest_rerun_step(harness)
    return list(range(start, 10))


def _copy_step_rounds(
    layer_id: str,
    feature_id: str,
    from_ts: str,
    to_ts: str,
    *,
    before_step: int,
    logs_root: Optional[Path] = None,
) -> None:
    if before_step <= 1:
        return
    # Collect phase dirs that contain any step before before_step
    phases_needed = {_STEP_PHASE[s] for s in range(1, int(before_step)) if s in _STEP_PHASE}
    _copy_phases(layer_id, feature_id, from_ts, to_ts, list(phases_needed), logs_root=logs_root)


def _discover_latest_completed_round(feature_dir: Path, initial_timestamp: str) -> int:
    """
    Return the largest completed round index N such that
    logs/.../{initial_timestamp}_rN/trace.json exists.
    """
    latest = 0
    pattern = re.compile(rf"^{re.escape(initial_timestamp)}_r(\d+)$")
    if not feature_dir.exists():
        return latest
    for child in feature_dir.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if not m:
            continue
        if not (child / "trace.json").exists():
            continue
        latest = max(latest, int(m.group(1)))
    return latest


def _discover_next_incomplete_round(
    feature_dir: Path,
    initial_timestamp: str,
    after_round: int,
) -> Optional[int]:
    """Return the next round directory after after_round that lacks trace.json."""
    pattern = re.compile(rf"^{re.escape(initial_timestamp)}_r(\d+)$")
    if not feature_dir.exists():
        return None
    candidates: List[int] = []
    for child in feature_dir.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx <= after_round:
            continue
        if not (child / "trace.json").exists():
            candidates.append(idx)
    return min(candidates) if candidates else None


def _round_trace_path(feature_dir: Path, timestamp: str) -> Path:
    return feature_dir / timestamp / "trace.json"


def _pipeline_failure_path(feature_dir: Path, timestamp: str) -> Path:
    return feature_dir / timestamp / "pipeline_failure.json"


def _write_pipeline_failure(
    *,
    layer_id: str,
    feature_id: str,
    timestamp: str,
    failed_step: int,
    command: List[str],
    logs_root: Optional[Path] = None,
) -> Dict[str, Any]:
    feature_dir = build_feature_dir(layer_id=layer_id, feature_id=feature_id, logs_root=logs_root)
    failure = {
        "type": "pipeline_step_failed",
        "timestamp": timestamp,
        "failed_step": int(failed_step),
        "command": command,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = _pipeline_failure_path(feature_dir, timestamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return failure


def _read_pipeline_failure(feature_dir: Path, timestamp: str) -> Optional[Dict[str, Any]]:
    path = _pipeline_failure_path(feature_dir, timestamp)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _clear_pipeline_failure(feature_dir: Path, timestamp: str) -> None:
    path = _pipeline_failure_path(feature_dir, timestamp)
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def _proposal_round_index(timestamp: str, initial_timestamp: str) -> int:
    if timestamp == initial_timestamp:
        return 0
    m = re.match(rf"^{re.escape(initial_timestamp)}_r(\d+)$", timestamp)
    return int(m.group(1)) if m else -1


def _find_proposal_for_round(
    feature_dir: Path,
    initial_timestamp: str,
    round_idx: int,
) -> tuple[Optional[Dict[str, Any]], Optional[Path]]:
    candidates: List[Path] = []
    for child in feature_dir.iterdir() if feature_dir.exists() else []:
        if not child.is_dir():
            continue
        if child.name != initial_timestamp and not re.match(rf"^{re.escape(initial_timestamp)}_r\d+$", child.name):
            continue
        path = child / f"agent_proposal_r{round_idx}.json"
        if path.exists():
            candidates.append(path)
    if not candidates:
        return None, None
    candidates.sort(key=lambda p: _proposal_round_index(p.parent.name, initial_timestamp), reverse=True)
    path = candidates[0]
    try:
        return json.loads(path.read_text(encoding="utf-8")), path
    except Exception:
        return None, path


def _infer_failed_step_from_logs(
    *,
    layer_id: str,
    feature_id: str,
    initial_timestamp: str,
    round_timestamp: str,
) -> Optional[int]:
    pattern = re.compile(
        rf"L{re.escape(layer_id)}-F{re.escape(feature_id)}/{re.escape(round_timestamp)}/Step(\d+)\s+FAILED"
    )
    failed_step: Optional[int] = None
    for log_path in sorted(CODE_DIR.glob(f"main_exp_{initial_timestamp}_*.log"), key=lambda p: p.stat().st_mtime):
        try:
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                m = pattern.search(line)
                if m:
                    failed_step = int(m.group(1))
        except OSError:
            continue
    return failed_step


def _resume_steps_for_incomplete_round(
    *,
    layer_id: str,
    feature_id: str,
    initial_timestamp: str,
    round_timestamp: str,
    feature_dir: Path,
    proposal: Dict[str, Any],
) -> List[int]:
    failure = _read_pipeline_failure(feature_dir, round_timestamp) or {}
    failed_step = failure.get("failed_step")
    if failed_step is None:
        failed_step = _infer_failed_step_from_logs(
            layer_id=layer_id,
            feature_id=feature_id,
            initial_timestamp=initial_timestamp,
            round_timestamp=round_timestamp,
        )
    if failed_step is not None:
        return list(range(int(failed_step), 10))
    harness = proposal.get("harness") or {}
    return _resolve_rerun_steps(harness, proposal.get("rerun_steps"))


# ── Pipeline execution ────────────────────────────────────────────────────────


_INPUT_CASE_THRESHOLD = 0.20
_OUTPUT_CASE_THRESHOLD = 0.20
_CHAIN_CASE_THRESHOLD = 1.0  # score points

def _append_case_locked(cases_path: Path, case_text: str) -> None:
    append_text_locked(cases_path, "\n" + case_text + "\n")


def _harness_subset(harness: Dict[str, Any], keys: tuple[str, ...]) -> Dict[str, Any]:
    return {
        key: harness[key]
        for key in keys
        if key in harness and harness.get(key) not in (None, "", [], {})
    }



def _metric_for_side(metrics: Dict[str, Any], side: str) -> float:
    if side == "input":
        return min(
            float(metrics.get("input_activation_rate", 0.0)),
            float(metrics.get("input_boundary_non_activation_rate", 0.0)),
        )
    if side == "output":
        return float(metrics.get("output_score", 0.0))
    if side == "chain":
        return float(metrics.get("best_chain_score", 0.0))
    return 0.0


def _initial_best_metrics(all_round_traces: Dict[str, Dict[str, Any]], current_trace: Dict[str, Any]) -> Dict[str, Any]:
    if not all_round_traces:
        return extract_trace_metrics(current_trace)
    metrics_list = [extract_trace_metrics(trace) for trace in all_round_traces.values()]
    best_input = max(metrics_list, key=lambda m: _metric_for_side(m, "input"))
    best_output = max(metrics_list, key=lambda m: _metric_for_side(m, "output"))
    best_chain = max(metrics_list, key=lambda m: _metric_for_side(m, "chain"))
    return {
        "input_activation_rate": best_input["input_activation_rate"],
        "input_boundary_non_activation_rate": best_input["input_boundary_non_activation_rate"],
        "output_score": best_output["output_score"],
        "best_chain_score": best_chain["best_chain_score"],
    }


def _gate_pass_for_side(
    metrics: Dict[str, Any],
    side: str,
    *,
    score_threshold: int,
    input_activation_threshold: float,
    input_boundary_threshold: float,
    output_score_threshold: float,
) -> bool:
    if side == "input":
        return (
            float(metrics.get("input_activation_rate", 0.0)) >= float(input_activation_threshold)
            and float(metrics.get("input_boundary_non_activation_rate", 0.0)) >= float(input_boundary_threshold)
        )
    if side == "output":
        return float(metrics.get("output_score", 0.0)) >= float(output_score_threshold)
    if side == "chain":
        return float(metrics.get("best_chain_score", 0.0)) >= float(score_threshold)
    return False


def _side_threshold(side: str) -> float:
    if side == "input":
        return _INPUT_CASE_THRESHOLD
    if side == "output":
        return _OUTPUT_CASE_THRESHOLD
    if side == "chain":
        return _CHAIN_CASE_THRESHOLD
    return 1.0


def _side_margin(side: str) -> float:
    return 0.1 if side in ("input", "output") else 1


def _harness_for_side(harness: Dict[str, Any], side: str) -> Dict[str, Any]:
    if side == "input":
        return _harness_subset(harness, _INPUT_HARNESS_KEYS)
    if side == "output":
        return _harness_subset(harness, _OUTPUT_HARNESS_KEYS)
    if side == "chain":
        return _harness_subset(harness, _CHAIN_HARNESS_KEYS)
    return {}


def _candidate_sides(harness: Dict[str, Any]) -> List[str]:
    return [side for side in ("input", "output", "chain") if _harness_for_side(harness, side)]


def _meaningful_global_improvement(
    *,
    side: str,
    before_metrics: Dict[str, Any],
    after_metrics: Dict[str, Any],
    prior_best: float,
    score_threshold: int,
    input_activation_threshold: float,
    input_boundary_threshold: float,
    output_score_threshold: float,
) -> tuple[bool, str]:
    before_score = _metric_for_side(before_metrics, side)
    after_score = _metric_for_side(after_metrics, side)
    delta = after_score - before_score
    gate_flip = (
        not _gate_pass_for_side(
            before_metrics,
            side,
            score_threshold=score_threshold,
            input_activation_threshold=input_activation_threshold,
            input_boundary_threshold=input_boundary_threshold,
            output_score_threshold=output_score_threshold,
        )
        and _gate_pass_for_side(
            after_metrics,
            side,
            score_threshold=score_threshold,
            input_activation_threshold=input_activation_threshold,
            input_boundary_threshold=input_boundary_threshold,
            output_score_threshold=output_score_threshold,
        )
    )
    if delta < _side_threshold(side) and not gate_flip:
        return False, f"delta {delta:.3f} below threshold"
    if after_score <= prior_best + _side_margin(side):
        return False, f"after {after_score:.3f} did not beat prior best {prior_best:.3f}"
    return True, f"{side} {before_score:.3f}->{after_score:.3f}, prior_best={prior_best:.3f}"


def _metrics_json(metrics: Dict[str, Any]) -> str:
    return json.dumps(metrics, ensure_ascii=False, sort_keys=True)


def _append_feature_case(
    *,
    skills_dir: Path,
    side: str,
    record: Dict[str, Any],
    reason: str,
    prior_best: float,
    layer_id: str,
    feature_id: str,
    post_write: Optional[Callable[[], Any]] = None,
) -> str:
    harness = _harness_for_side(record.get("harness") or {}, side)
    case_id = make_case_id(
        side=side,
        layer_id=layer_id,
        feature_id=feature_id,
        round_idx=int(record.get("round_idx") or 0),
        trigger="feature_extract",
        harness=harness,
    )
    after_score = _metric_for_side(record.get("metrics_after") or {}, side)
    before_score = _metric_for_side(record.get("metrics_before") or {}, side)
    heading = (
        f"## [Case][{side}][case: {case_id}] "
        f"L{layer_id}-F{feature_id} r{int(record.get('round_idx') or 0)}: "
        f"{side} +{after_score - before_score:.2f}"
    )
    harness_json = json.dumps(harness, ensure_ascii=False, indent=2, sort_keys=True)
    body = (
        f"**When:** Feature L{layer_id}-F{feature_id}, round {int(record.get('round_idx') or 0)}, "
        f"mode `{record.get('mode', 'unknown')}`.\n\n"
        f"**Fix:** Apply this {side}-side harness change when the same trace pattern appears.\n\n"
        f"**Evidence:** {reason}. Before metrics `{_metrics_json(record.get('metrics_before') or {})}`; "
        f"after metrics `{_metrics_json(record.get('metrics_after') or {})}`.\n\n"
        f"**Prior best {side} score:** {prior_best:.3f}.\n\n"
        f"**Harness:**\n```json\n{harness_json}\n```\n\n"
        f"**Diagnosis:** {str(record.get('diagnosis') or '')[:1200]}"
    )
    meta = {
        "case_id": case_id,
        "side": side,
        "source_feature": {
            "layer_id": int(layer_id),
            "feature_id": int(feature_id),
            "round_idx": int(record.get("round_idx") or 0),
            "before_timestamp": record.get("before_timestamp"),
            "after_timestamp": record.get("after_timestamp"),
        },
        "write_trigger": "feature_extract",
        "metrics_before": record.get("metrics_before") or {},
        "metrics_after": record.get("metrics_after") or {},
        "prior_best_score": prior_best,
        "usage": {
            "use_count": 0,
            "effective_use_count": 0,
            "compressed": False,
            "compressed_at": None,
            "compressed_result": None,
        },
    }
    return append_skill_case(
        skills_dir,
        side=side,
        heading=heading,
        body=body,
        meta=meta,
        post_write=post_write,
    )


def _write_feature_end_cases(
    *,
    round_records: List[Dict[str, Any]],
    initial_metrics: Dict[str, Any],
    skills_dir: Path,
    layer_id: str,
    feature_id: str,
    score_threshold: int,
    input_activation_threshold: float,
    input_boundary_threshold: float,
    output_score_threshold: float,
    post_write: Optional[Callable[[], Any]] = None,
) -> List[Dict[str, Any]]:
    running_best = {side: _metric_for_side(initial_metrics, side) for side in ("input", "output", "chain")}
    written: List[Dict[str, Any]] = []
    for record in round_records:
        after_metrics = record.get("metrics_after") or {}
        before_metrics = record.get("metrics_before") or {}
        for side in _candidate_sides(record.get("harness") or {}):
            prior_best = running_best.get(side, 0.0)
            ok, reason = _meaningful_global_improvement(
                side=side,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                prior_best=prior_best,
                score_threshold=score_threshold,
                input_activation_threshold=input_activation_threshold,
                input_boundary_threshold=input_boundary_threshold,
                output_score_threshold=output_score_threshold,
            )
            if ok:
                case_id = _append_feature_case(
                    skills_dir=skills_dir,
                    side=side,
                    record=record,
                    reason=reason,
                    prior_best=prior_best,
                    layer_id=layer_id,
                    feature_id=feature_id,
                    post_write=post_write,
                )
                written.append({"case_id": case_id, "side": side, "reason": reason, "round_idx": record.get("round_idx")})
        for side in running_best:
            running_best[side] = max(running_best[side], _metric_for_side(after_metrics, side))
    return written


def _normalise_used_skill_cases(raw_cases: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_cases, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw_cases:
        if isinstance(item, str):
            out.append({"case_id": item})
        elif isinstance(item, dict) and item.get("case_id"):
            out.append(dict(item))
    return out


def _record_used_skill_cases(
    *,
    round_records: List[Dict[str, Any]],
    initial_metrics: Dict[str, Any],
    skills_dir: Path,
    layer_id: str,
    feature_id: str,
    score_threshold: int,
    input_activation_threshold: float,
    input_boundary_threshold: float,
    output_score_threshold: float,
    post_write: Optional[Callable[[], Any]] = None,
) -> List[Dict[str, Any]]:
    running_best = {side: _metric_for_side(initial_metrics, side) for side in ("input", "output", "chain")}
    events: List[Dict[str, Any]] = []
    effective_counted: set[str] = set()
    use_counted: set[str] = set()
    feature_key = f"L{layer_id}-F{feature_id}"
    for record in round_records:
        before_metrics = record.get("metrics_before") or {}
        after_metrics = record.get("metrics_after") or {}
        used_cases = _normalise_used_skill_cases((record.get("proposal") or {}).get("used_skill_cases"))
        seen_this_round: set[str] = set()
        for used in used_cases:
            case_id = str(used.get("case_id") or "").strip()
            if not case_id or case_id in seen_this_round:
                continue
            seen_this_round.add(case_id)
            side = case_side_from_id_or_lookup(skills_dir, case_id)
            effective = False
            reason = "case not found"
            dedupe_key = f"{feature_key}:{case_id}"
            count_this_use = False
            if side:
                count_this_use = dedupe_key not in use_counted
                if count_this_use:
                    use_counted.add(dedupe_key)
                    effective, reason = _meaningful_global_improvement(
                        side=side,
                        before_metrics=before_metrics,
                        after_metrics=after_metrics,
                        prior_best=running_best.get(side, 0.0),
                        score_threshold=score_threshold,
                        input_activation_threshold=input_activation_threshold,
                        input_boundary_threshold=input_boundary_threshold,
                        output_score_threshold=output_score_threshold,
                    )
                    if effective and dedupe_key in effective_counted:
                        effective = False
                        reason = "effective use already counted for this feature"
                    elif effective:
                        effective_counted.add(dedupe_key)
                else:
                    reason = "case use already counted for this feature"
            if side and count_this_use:
                increment_case_usage(
                    skills_dir,
                    case_id,
                    use_delta=1,
                    effective_delta=1 if effective else 0,
                    post_write=post_write,
                )
            event = {
                "event": "case_use",
                "case_id": case_id,
                "side": side,
                "layer_id": int(layer_id),
                "feature_id": int(feature_id),
                "round_idx": int(record.get("round_idx") or 0),
                "mode": record.get("mode"),
                "applied_experience": used.get("applied_experience", ""),
                "expected_effect": used.get("expected_effect", ""),
                "metrics_before": before_metrics,
                "metrics_after": after_metrics,
                "effective": bool(effective),
                "effect_reason": reason,
            }
            events.append(event)
        for side in running_best:
            running_best[side] = max(running_best[side], _metric_for_side(after_metrics, side))
    return events


def _build_step1_command(
    *,
    layer_id: str,
    feature_id: str,
    timestamp: str,
    logs_root: Path,
    model_id: str,
    sae_width: str,
    input_observation_source: str,
    bos_token_root: Optional[Path],
    bos_prompt_id: Optional[str],
) -> List[str]:
    cmd = [
        PYTHON,
        "step1_initial_observation.py",
        "--model-id", str(model_id),
        "--layer-id", str(layer_id),
        "--feature-id", str(feature_id),
        "--width", str(sae_width),
        "--timestamp", str(timestamp),
        "--logs-root", str(logs_root),
        "--observation-m", "10",
        "--observation-n", "5",
        "--input-observation-source", str(input_observation_source),
    ]
    if str(input_observation_source) == "bos_token":
        if bos_token_root is None:
            raise ValueError("bos_token_root is required for a BOS-token run")
        if not str(bos_prompt_id or "").strip():
            raise ValueError("bos_prompt_id is required for a BOS-token run")
        cmd.extend([
            "--bos-token-root", str(bos_token_root),
            "--bos-prompt-id", str(bos_prompt_id),
        ])
    return cmd


def _trace_bos_prompt_id(trace: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(trace, dict):
        return None
    observation = (trace.get("input_round") or {}).get("observation") or {}
    input_side = observation.get("input_side_observation") or observation
    meta = input_side.get("bos_token_scan_meta") or {}
    value = meta.get("prompt_id")
    return str(value).strip() if value else None


def _trace_input_source(trace: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(trace, dict):
        return None
    observation = (trace.get("input_round") or {}).get("observation") or {}
    value = observation.get("input_source")
    if not value:
        value = (observation.get("input_side_observation") or {}).get("source")
    return str(value).strip() if value else None


def _resolve_bos_prompt_choice(
    *,
    input_observation_source: str,
    bos_token_root: Optional[Path],
    layer_id: str,
    feature_id: str,
    current_prompt_id: Optional[str],
    harness: Dict[str, Any],
) -> tuple[Optional[str], bool]:
    requested = str(harness.get("bos_prompt_id") or "").strip()
    if input_observation_source != "bos_token":
        if requested:
            raise ValueError("bos_prompt_id is not allowed in a Neuronpedia run")
        return None, False
    if bos_token_root is None:
        raise ValueError("bos_token_root is required for a BOS-token run")
    selected = requested or str(current_prompt_id or "").strip()
    if not selected:
        raise ValueError("a BOS-token run requires a current or requested bos_prompt_id")
    if "/" in selected or "\\" in selected or selected in {".", ".."}:
        raise ValueError("bos_prompt_id must be a single directory name")
    prompt_path = (
        Path(bos_token_root)
        / f"layer-{int(layer_id)}"
        / f"feature-{int(feature_id)}"
        / "bos_token"
        / selected
        / "top_tokens.json"
    )
    if not prompt_path.exists():
        raise ValueError(f"selected BOS prompt has no top_tokens.json: {prompt_path}")
    return selected, selected != str(current_prompt_id or "").strip()


def _execute_pipeline(
    *,
    layer_id: str,
    feature_id: str,
    new_timestamp: str,
    prev_timestamp: str,
    harness: Dict[str, Any],
    sae_path: str,
    model_path: str,
    llm_base_url: str,
    llm_model: str,
    api_key: str,
    device: str,
    model_id: str,
    log_file: Optional[Path],
    rerun_steps: Optional[List[int]] = None,
    skip_gate1_early_stop: bool = False,  # True = run all steps (use for initial baseline pipeline)
    sae_width: str = "16k",
    inference_server_url: str = DEFAULT_INFERENCE_SERVER_URL,
    inference_timeout_sec: float = DEFAULT_INFERENCE_TIMEOUT_SEC,
    no_inference_server: bool = False,
    logs_root: Optional[Path] = None,
    input_observation_source: str = "neuronpedia",
    bos_token_root: Optional[Path] = None,
    bos_prompt_id: Optional[str] = None,
) -> bool:
    label = f"L{layer_id}-F{feature_id}/{new_timestamp}"
    steps_to_run = _resolve_rerun_steps(harness, rerun_steps)
    selected_bos_prompt_id, prompt_changed = _resolve_bos_prompt_choice(
        input_observation_source=input_observation_source,
        bos_token_root=bos_token_root,
        layer_id=layer_id,
        feature_id=feature_id,
        current_prompt_id=bos_prompt_id,
        harness=harness,
    )
    if prompt_changed:
        steps_to_run = sorted(set(steps_to_run).union({1, 2, 3, 4}))

    env = dict(os.environ)
    env.update({"LLM_BASE_URL": llm_base_url, "LLM_MODEL": llm_model, "LLM_API_KEY": api_key})

    # Copy ALL phases first (preserves step outputs not being re-run)
    _copy_phases(layer_id, feature_id, prev_timestamp, new_timestamp, ["input", "output", "chain"], logs_root=logs_root)
    _log(f"{label}: copied all phases from {prev_timestamp}, re-running steps {steps_to_run}", log_file)

    base_args = ["--layer-id", layer_id, "--feature-id", feature_id, "--timestamp", new_timestamp, "--logs-root", str(logs_root or "logs")]
    inference_args = ["--no-inference-server"] if no_inference_server else [
        "--inference-server-url", inference_server_url,
        "--inference-timeout-sec", str(inference_timeout_sec),
    ]
    raw_intervention_scope = harness.get("intervention_scope", "max_activation_token")
    intervention_scope = _normalize_intervention_scope(raw_intervention_scope)
    if str(raw_intervention_scope).strip() != intervention_scope:
        _log(
            f"{label}: normalized intervention_scope {raw_intervention_scope!r} -> {intervention_scope!r}",
            log_file,
        )
    steps: List[tuple[int, List[str]]] = [
        (
            1,
            _build_step1_command(
                layer_id=layer_id,
                feature_id=feature_id,
                timestamp=new_timestamp,
                logs_root=Path(logs_root or "logs"),
                model_id=model_id,
                sae_width=sae_width,
                input_observation_source=input_observation_source,
                bos_token_root=bos_token_root,
                bos_prompt_id=selected_bos_prompt_id,
            ),
        ),
        (
            2,
            [
                PYTHON, "step2_generate_input_hypotheses.py",
                *base_args,
                "--num-hypothesis", "3",
                "--llm-base-url", llm_base_url, "--llm-model", llm_model,
            ],
        ),
        (
            3,
            [
                PYTHON, "step3_design_input_experiments.py",
                *base_args,
                "--num-sentences-per-hypothesis", "5",
                "--llm-base-url", llm_base_url, "--llm-model", llm_model,
            ],
        ),
        (
            4,
            [
                PYTHON, "step4_score_input_experiments.py",
                *base_args,
                "--width", sae_width,
                "--sae-path", sae_path,
                "--model-checkpoint-path", model_path,
                "--device", device,
                *inference_args,
            ],
        ),
        (
            5,
            [
                PYTHON, "step5_run_intervention.py",
                *base_args,
                "--width", sae_width,
                "--sae-path", sae_path,
                "--model-checkpoint-path", model_path,
                "--device", device,
                *inference_args,
                "--top-k", str(_normalize_top_k(harness.get("top_k", 30))),
                "--intervention-scope", intervention_scope,
            ],
        ),
        (
            6,
            [
                PYTHON, "step6_generate_output_hypotheses.py",
                *base_args,
                "--llm-base-url", llm_base_url, "--llm-model", llm_model,
            ],
        ),
        (
            7,
            [
                PYTHON, "step7_score_output_hypotheses.py",
                *base_args,
                "--llm-base-url", llm_base_url, "--llm-model", llm_model,
            ],
        ),
        (
            8,
            [
                PYTHON, "step8_build_chain_hypotheses.py",
                *base_args,
                "--llm-base-url", llm_base_url, "--llm-model", llm_model,
            ],
        ),
        (
            9,
            [
                PYTHON, "step9_synthesize_chain_explanation.py",
                *base_args,
                "--llm-base-url", llm_base_url, "--llm-model", llm_model,
            ],
        ),
    ]

    extra_in = harness.get("extra_input_guidance")
    if extra_in:
        steps[1][1].extend(["--extra-input-guidance", str(extra_in)])

    max_activation_scale = harness.get("max_activation_scale")
    if max_activation_scale is not None:
        steps[4][1].extend(["--max-activation-scale", str(max_activation_scale)])
    last_token_scale = harness.get("last_token_scale")
    if last_token_scale is not None:
        steps[4][1].extend(["--last-token-scale", str(last_token_scale)])
    custom_prompts = harness.get("custom_steering_prompts")
    if custom_prompts and isinstance(custom_prompts, list):
        steps[4][1].extend(["--custom-steering-prompts", json.dumps(custom_prompts)])

    extra_out = harness.get("extra_output_guidance")
    if extra_out:
        steps[5][1].extend(["--extra-output-guidance", str(extra_out)])

    extra_chain = harness.get("extra_chain_guidance")
    if extra_chain:
        steps[7][1].extend(["--extra-chain-guidance", str(extra_chain)])

    for step_index, cmd in steps:
        if step_index not in steps_to_run:
            continue
        if not _run_step(cmd, env=env, label=f"{label}/Step{step_index}", log_file=log_file):
            _write_pipeline_failure(
                layer_id=layer_id,
                feature_id=feature_id,
                timestamp=new_timestamp,
                failed_step=step_index,
                command=cmd,
                logs_root=logs_root,
            )
            return False
        # Gate 1 early stop: skip expensive steps 5-7 when activation clearly fails.
        # Steps 8-9 (chain) still run with Gate 2 data inherited from the previous round,
        # combining the new Gate 1 hypotheses with the old output evidence.
        if (not skip_gate1_early_stop) and step_index == 4 and any(5 <= s <= 7 for s in steps_to_run):
            gate1 = _load_step4_gate1(layer_id, feature_id, new_timestamp, logs_root=logs_root)
            if gate1 is not None:
                act, bnd = gate1
                if act < 0.75:
                    _log(
                        f"{label}: Gate 1 activation {act:.3f} < 0.75 — skipping steps 5-7, "
                        f"steps 8-9 will chain new Gate 1 with inherited Gate 2 data",
                        log_file,
                    )
                    steps_to_run = [s for s in steps_to_run if s not in (5, 6, 7)]



    # If step 9 didn't run (input-only iteration), write partial trace from available step outputs.
    # This happens when rerun_steps is all within [1-4] and gate1_early_stop didn't fire.
    if 9 not in steps_to_run:
        trace_dir = build_round_dir(
            layer_id=layer_id, feature_id=feature_id, timestamp=new_timestamp, round_id="input", logs_root=logs_root,
        ).parent
        trace_path = trace_dir / "trace.json"
        if not trace_path.exists():
            ok = _write_partial_trace(
                layer_id=layer_id, feature_id=feature_id,
                timestamp=new_timestamp, log_file=log_file, logs_root=logs_root,
            )
            if ok:
                _clear_pipeline_failure(build_feature_dir(layer_id=layer_id, feature_id=feature_id, logs_root=logs_root), new_timestamp)
            return ok

    _clear_pipeline_failure(build_feature_dir(layer_id=layer_id, feature_id=feature_id, logs_root=logs_root), new_timestamp)
    return True

# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent_loop(
    *,
    layer_id: str,
    feature_id: str,
    initial_timestamp: str,
    sae_path: str,
    model_path: str,
    llm_base_url: str,
    llm_model: str,
    llm_api_key_file: Optional[str] = DEFAULT_API_KEY_FILE,
    device: str = "cuda",
    model_id: str = "gemma-2-2b",
    max_rounds: int = 5,
    score_threshold: int = 4,
    input_activation_threshold: float = 0.8,
    input_boundary_threshold: float = 0.8,
    output_score_threshold: float = 0.5,
    log_file: Optional[Path] = None,
    skills_dir: Optional[Path] = None,
    skill_learning_enabled: bool = True,
    skill_maintenance_mode: str = "off",
    skill_maintenance_archive_dir: Optional[Path] = None,
    sae_width: str = "16k",
    inference_server_url: str = DEFAULT_INFERENCE_SERVER_URL,
    inference_timeout_sec: float = DEFAULT_INFERENCE_TIMEOUT_SEC,
    no_inference_server: bool = False,
    logs_root: Optional[Path] = None,
    input_observation_source: str = "neuronpedia",
    bos_token_root: Optional[Path] = None,
    bos_prompt_id: Optional[str] = None,
) -> Dict[str, Any]:
    if skill_maintenance_mode not in {"off", "auto"}:
        raise ValueError("skill_maintenance_mode must be 'off' or 'auto'")
    if input_observation_source not in {"neuronpedia", "bos_token"}:
        raise ValueError("input_observation_source must be 'neuronpedia' or 'bos_token'")
    api_key = read_api_key(llm_api_key_file)
    client = OpenAI(base_url=llm_base_url, api_key=api_key)
    token_counter = TokenUsageAccumulator()
    pipeline_cost = TokenUsageAccumulator()

    resolved_maintenance_archive = (
        Path(skill_maintenance_archive_dir)
        if skill_maintenance_archive_dir is not None
        else (Path(logs_root) if logs_root is not None else CODE_DIR / "logs") / "skill_case_archive"
    )

    def _post_skill_write() -> Dict[str, Any]:
        if skill_maintenance_mode != "auto" or skills_dir is None:
            return {"status": "disabled"}
        result = maybe_run_auto_skill_maintenance(
            client,
            llm_model,
            skills_dir=skills_dir,
            archive_dir=resolved_maintenance_archive,
            maintenance_name=f"run-{initial_timestamp}",
            verbose=True,
        )
        if result.get("status") == "failed":
            _log(f"Skill maintenance failed (non-fatal): {result.get('error', 'unknown error')}", log_file)
        return result

    feature_dir = build_feature_dir(layer_id=layer_id, feature_id=feature_id, logs_root=logs_root)
    initial_trace_path = feature_dir / initial_timestamp / "trace.json"
    if not initial_trace_path.exists():
        raise FileNotFoundError(f"Initial trace not found: {initial_trace_path}")

    latest_completed_round = _discover_latest_completed_round(feature_dir, initial_timestamp)
    if latest_completed_round > 0:
        current_timestamp = f"{initial_timestamp}_r{latest_completed_round}"
        current_trace_path = feature_dir / current_timestamp / "trace.json"
        if not current_trace_path.exists():
            current_trace_path = initial_trace_path
            current_timestamp = initial_timestamp
            latest_completed_round = 0
    else:
        current_timestamp = initial_timestamp
        current_trace_path = initial_trace_path

    current_trace = json.loads(current_trace_path.read_text(encoding="utf-8"))
    trace_input_source = _trace_input_source(current_trace)
    if trace_input_source and trace_input_source != input_observation_source:
        raise ValueError(
            f"initial trace input source {trace_input_source!r} does not match configured "
            f"source {input_observation_source!r}"
        )
    current_bos_prompt_id = _trace_bos_prompt_id(current_trace) or (
        str(bos_prompt_id).strip() if bos_prompt_id else None
    )
    if input_observation_source == "bos_token":
        _resolve_bos_prompt_choice(
            input_observation_source=input_observation_source,
            bos_token_root=bos_token_root,
            layer_id=layer_id,
            feature_id=feature_id,
            current_prompt_id=current_bos_prompt_id,
            harness={},
        )
    pipeline_cost.add(current_trace.get("meta", {}).get("token_cost", {}))
    # Revert baseline: track the best-ever output so regression always reverts to it.
    # Updated when a new round achieves output above threshold.
    _initial_m = extract_trace_metrics(current_trace)
    revert_baseline_output: float = _initial_m.get("output_score", 0.0)
    revert_baseline_ts: str = current_timestamp
    revert_baseline_trace: Dict[str, Any] = current_trace
    prev_timestamps: List[str] = [initial_timestamp]
    for i in range(1, latest_completed_round):
        prev_timestamps.append(f"{initial_timestamp}_r{i}")
    if latest_completed_round == 0:
        prev_timestamps = []

    # Track all round traces so we can pick the best at the end
    all_round_traces: Dict[str, Dict[str, Any]] = {}
    # Load all completed round traces (initial + r1..rN)
    for ts_candidate in [initial_timestamp] + [
        f"{initial_timestamp}_r{i}" for i in range(1, latest_completed_round + 1)
    ]:
        _cand_path = feature_dir / ts_candidate / "trace.json"
        if _cand_path.exists():
            try:
                all_round_traces[ts_candidate] = json.loads(_cand_path.read_text(encoding="utf-8"))
            except Exception:
                pass
    if current_timestamp not in all_round_traces:
        all_round_traces[current_timestamp] = current_trace
    initial_prior_best_metrics = _initial_best_metrics(all_round_traces, current_trace)
    agent_log: List[Dict[str, Any]] = []
    round_records: List[Dict[str, Any]] = []
    prev_harness: Optional[Dict[str, Any]] = None  # tracks last proposed harness config
    prev_metrics: Optional[Dict[str, Any]] = None  # for stuck detection
    prev_diagnosis: Optional[str] = None  # last agent diagnosis, injected into next round
    rounds_started: int = latest_completed_round  # counts rounds entered (incl. early-stop rounds)
    gate1_stuck_count: int = 0  # consecutive Gate-1-targeted rounds with delta < threshold
    gate1_accepted: bool = False  # when True, skip Gate 1 retries and move to Gate 2 / chain
    loop_status = "complete"
    failure_info: Optional[Dict[str, Any]] = None

    _log(
        f"=== Agent loop: layer={layer_id} feature={feature_id} initial_ts={initial_timestamp} "
        f"max_rounds={max_rounds} chain_threshold={score_threshold} "
        f"input_thresholds=({input_activation_threshold:.2f},{input_boundary_threshold:.2f}) "
        f"output_threshold={output_score_threshold:.2f} ===",
        log_file,
    )
    if latest_completed_round > 0:
        _log(
            f"Resume detected: latest_completed_round={latest_completed_round}, "
            f"current_ts={current_timestamp}",
            log_file,
        )

    for round_idx in range(latest_completed_round + 1, max_rounds + 1):
        new_timestamp = f"{initial_timestamp}_r{round_idx}"
        resume_incomplete = (
            _discover_next_incomplete_round(feature_dir, initial_timestamp, round_idx - 1) == round_idx
        )
        resume_proposal_path: Optional[Path] = None
        proposal: Dict[str, Any]

        if resume_incomplete:
            proposal, resume_proposal_path = _find_proposal_for_round(feature_dir, initial_timestamp, round_idx)
            if proposal is None:
                _log(
                    f"Round {round_idx}: incomplete round exists but proposal is missing; "
                    "asking agent for a fresh proposal",
                    log_file,
                )
                resume_incomplete = False
            else:
                proposal_prev_timestamp = resume_proposal_path.parent.name if resume_proposal_path else current_timestamp
                proposal_trace_path = _round_trace_path(feature_dir, proposal_prev_timestamp)
                if proposal_trace_path.exists():
                    current_timestamp = proposal_prev_timestamp
                    current_trace = json.loads(proposal_trace_path.read_text(encoding="utf-8"))
                    if input_observation_source == "bos_token":
                        current_bos_prompt_id = _trace_bos_prompt_id(current_trace) or current_bos_prompt_id
                    all_round_traces[current_timestamp] = current_trace

        if not resume_incomplete and _should_stop(
            current_trace,
            score_threshold,
            input_activation_threshold,
            input_boundary_threshold,
            output_score_threshold,
        ):
            m = extract_trace_metrics(current_trace)
            _log(
                f"Round {round_idx}: joint thresholds met, early stop "
                f"(chain={int(m['best_chain_score'])}, input={m['input_activation_rate']:.3f}, "
                f"boundary={m['input_boundary_non_activation_rate']:.3f}, output={m['output_score']:.3f})",
                log_file,
            )
            break

        rounds_started += 1

        if resume_incomplete:
            harness = proposal.get("harness") or {}
            rerun_steps = _resume_steps_for_incomplete_round(
                layer_id=layer_id,
                feature_id=feature_id,
                initial_timestamp=initial_timestamp,
                round_timestamp=new_timestamp,
                feature_dir=feature_dir,
                proposal=proposal,
            )
            agent_log.append({
                "round": round_idx,
                "proposal": proposal,
                "resumed_incomplete_round": True,
                "proposal_path": str(resume_proposal_path) if resume_proposal_path else None,
                "rerun_steps": rerun_steps,
            })
            _log(
                f"Round {round_idx}: resuming incomplete pipeline at timestamp={new_timestamp}; "
                f"rerun_steps={rerun_steps}",
                log_file,
            )
            if proposal.get("action") == "stop":
                _log(f"Round {round_idx}: saved proposal was stop; stopping loop", log_file)
                break
        else:
            # If Gate 1 is accepted as best-effort, force mode to skip input retries
            _override_mode: Optional[str] = None
            if gate1_accepted:
                _cur_m = extract_trace_metrics(current_trace)
                _override_mode = "chain" if _cur_m["output_score"] >= output_score_threshold else "output_validate"
                _log(f"Round {round_idx}: gate1_accepted — forcing mode={_override_mode}", log_file)

            _log(f"Round {round_idx}/{max_rounds}: proposing new harness...", log_file)
            try:
                proposal = propose_harness(
                    layer_id=layer_id,
                    feature_id=feature_id,
                    current_timestamp=current_timestamp,
                    round_idx=round_idx,
                    prev_timestamps=list(prev_timestamps),
                    prev_harness=prev_harness,
                    prev_diagnosis=prev_diagnosis,
                    client=client,
                    model=llm_model,
                    token_counter=token_counter,
                    override_mode=_override_mode,
                    skills_dir=skills_dir,
                    skill_learning_enabled=skill_learning_enabled,
                    skill_maintenance_mode=skill_maintenance_mode,
                    skill_maintenance_archive_dir=resolved_maintenance_archive,
                    logs_root=logs_root,
                    input_source=input_observation_source,
                    bos_token_root=bos_token_root,
                    bos_prompt_id=current_bos_prompt_id,
                    sae_path=sae_path,
                    model_path=model_path,
                    device=device,
                    sae_width=sae_width,
                    inference_server_url=inference_server_url,
                    inference_timeout_sec=inference_timeout_sec,
                    no_inference_server=no_inference_server,
                )
            except Exception as exc:
                loop_status = "failed_proposer"
                failure_info = {
                    "type": "proposer_exception",
                    "round_idx": round_idx,
                    "current_timestamp": current_timestamp,
                    "message": str(exc),
                }
                agent_log.append({"round": round_idx, "error": failure_info})
                _log(f"Round {round_idx}: proposer failed: {exc}", log_file)
                break
            agent_log.append({"round": round_idx, "proposal": proposal})

            # Persist proposal alongside current trace
            proposal_path = feature_dir / current_timestamp / f"agent_proposal_r{round_idx}.json"
            proposal_path.parent.mkdir(parents=True, exist_ok=True)
            proposal_path.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            skills_written = proposal.get("skills_written") or []
            _log(
                f"Round {round_idx}: action={proposal.get('action')}  "
                f"diagnosis={str(proposal.get('diagnosis', ''))[:120]}"
                + (f"  skills_written={skills_written}" if skills_written else ""),
                log_file,
            )

            if proposal.get("action") == "stop":
                _log(f"Round {round_idx}: agent chose to stop", log_file)
                break

            harness = proposal.get("harness") or {}
            rerun_steps = proposal.get("rerun_steps")

        _log(f"Round {round_idx}: running pipeline → timestamp={new_timestamp}", log_file)

        ok = _execute_pipeline(
            layer_id=layer_id,
            feature_id=feature_id,
            new_timestamp=new_timestamp,
            prev_timestamp=current_timestamp,
            harness=harness,
            sae_path=sae_path,
            model_path=model_path,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            api_key=api_key,
            device=device,
            model_id=model_id,
            log_file=log_file,
            rerun_steps=rerun_steps,
            sae_width=sae_width,
            inference_server_url=inference_server_url,
            inference_timeout_sec=inference_timeout_sec,
            no_inference_server=no_inference_server,
            logs_root=logs_root,
            input_observation_source=input_observation_source,
            bos_token_root=bos_token_root,
            bos_prompt_id=current_bos_prompt_id,
        )
        if not ok:
            loop_status = "failed_pipeline"
            failure_info = _read_pipeline_failure(feature_dir, new_timestamp) or {
                "type": "pipeline_failed",
                "timestamp": new_timestamp,
            }
            failure_info["round_idx"] = round_idx
            _log(f"Round {round_idx}: pipeline failed, stopping loop", log_file)
            break

        new_trace_path = feature_dir / new_timestamp / "trace.json"
        if not new_trace_path.exists():
            loop_status = "failed_pipeline"
            failure_info = {
                "type": "trace_missing_after_pipeline",
                "timestamp": new_timestamp,
                "round_idx": round_idx,
            }
            _log(f"Round {round_idx}: trace.json missing after pipeline, stopping", log_file)
            break

        before_timestamp = current_timestamp
        prev_timestamps.append(current_timestamp)
        prev_metrics = extract_trace_metrics(current_trace)
        current_trace = json.loads(new_trace_path.read_text(encoding="utf-8"))
        if input_observation_source == "bos_token":
            current_bos_prompt_id = _trace_bos_prompt_id(current_trace) or current_bos_prompt_id
        pipeline_cost.add(current_trace.get("meta", {}).get("token_cost", {}))
        current_timestamp = new_timestamp
        all_round_traces[current_timestamp] = current_trace
        prev_harness = harness
        prev_diagnosis = proposal.get("diagnosis", "")
        new_m = extract_trace_metrics(current_trace)
        if prev_metrics and harness:
            round_records.append({
                "round_idx": round_idx,
                "before_timestamp": before_timestamp,
                "after_timestamp": new_timestamp,
                "metrics_before": dict(prev_metrics),
                "metrics_after": dict(new_m),
                "proposal": proposal,
                "harness": harness,
                "mode": proposal.get("mode", "unknown"),
                "diagnosis": proposal.get("diagnosis", ""),
                "rerun_steps": rerun_steps,
            })

        # ── Post-round analysis ──────────────────────────────────────────────────
        if prev_metrics and harness:
            targeted = _harness_targeted_input_or_output(harness)

            # Output regression protection: revert to best-ever baseline if output drops.
            # The baseline is updated whenever a round achieves output above threshold,
            # so we always protect the best output seen so far.
            if (
                targeted != "input"
                and revert_baseline_output >= output_score_threshold
                and new_m["output_score"] < output_score_threshold
            ):
                _log(
                    f"Round {round_idx}: output REGRESSION "
                    f"(baseline={revert_baseline_output:.3f} → {new_m['output_score']:.3f}), "
                    f"reverting to baseline trace ({revert_baseline_ts})",
                    log_file,
                )
                current_trace = revert_baseline_trace
                current_timestamp = revert_baseline_ts
                if input_observation_source == "bos_token":
                    current_bos_prompt_id = _trace_bos_prompt_id(current_trace) or current_bos_prompt_id
                prev_timestamps.clear()
                prev_harness = None
                prev_diagnosis = None
            elif new_m["output_score"] > revert_baseline_output:
                # Update baseline when this round achieves better output
                revert_baseline_output = new_m["output_score"]
                revert_baseline_ts = new_timestamp
                revert_baseline_trace = current_trace
                _log(
                    f"Round {round_idx}: output baseline updated to {revert_baseline_output:.3f} "
                    f"(ts={revert_baseline_ts})",
                    log_file,
                )

            # Stuck detection
            elif targeted == "input":
                delta = (
                    abs(new_m["input_activation_rate"] - prev_metrics["input_activation_rate"])
                    + abs(new_m["input_boundary_non_activation_rate"] - prev_metrics["input_boundary_non_activation_rate"])
                )
                if delta < _GATE1_STUCK_DELTA:
                    gate1_stuck_count += 1
                    if gate1_stuck_count >= 2 and not gate1_accepted:
                        _log(
                            f"Round {round_idx}: Gate 1 stuck for {gate1_stuck_count} consecutive rounds "
                            f"(delta={delta:.3f}). Accepting Gate 1 as best-effort and shifting focus.",
                            log_file,
                        )
                        gate1_accepted = True
                    else:
                        _log(
                            f"Round {round_idx}: Gate 1 slow (delta={delta:.3f} < {_GATE1_STUCK_DELTA}, "
                            f"stuck_count={gate1_stuck_count})",
                            log_file,
                        )
                else:
                    gate1_stuck_count = 0  # reset on meaningful movement

            elif targeted == "output":
                delta = abs(new_m["output_score"] - prev_metrics["output_score"])
                if delta < 0.05:
                    _log(
                        f"Round {round_idx}: stuck on Gate 2 (delta={delta:.3f} < 0.05), stopping early",
                        log_file,
                    )
                    break


    # ── Select the best round across all attempts ──────────────────────────────
    best_ts = max(
        all_round_traces,
        key=lambda ts: _rank_key(
            all_round_traces[ts],
            score_threshold, input_activation_threshold,
            input_boundary_threshold, output_score_threshold,
        ),
    )
    best_trace = all_round_traces[best_ts]
    best_rank = _rank_key(best_trace, score_threshold, input_activation_threshold,
                          input_boundary_threshold, output_score_threshold)
    last_rank = _rank_key(current_trace, score_threshold, input_activation_threshold,
                          input_boundary_threshold, output_score_threshold)

    if best_ts != current_timestamp:
        _log(
            f"Best round is {best_ts} (gates={best_rank[0]}, chain={best_rank[1]}, "
            f"out={best_rank[2]:.3f}) — overriding last round {current_timestamp} "
            f"(gates={last_rank[0]}, chain={last_rank[1]}, out={last_rank[2]:.3f})",
            log_file,
        )
        current_trace = best_trace
        current_timestamp = best_ts
    else:
        _log(
            f"Last round {current_timestamp} is already the best "
            f"(gates={best_rank[0]}, chain={best_rank[1]}, out={best_rank[2]:.3f})",
            log_file,
        )

    metrics = extract_trace_metrics(current_trace)
    best_score = int(metrics["best_chain_score"])

    # Write final token cost (pipeline + agent_loop + total) back into the trace
    final_token_cost = {
        "pipeline": pipeline_cost.as_dict(),
        "agent_loop": token_counter.as_dict(),
        "total": {
            "prompt_tokens": pipeline_cost.prompt_tokens + token_counter.prompt_tokens,
            "completion_tokens": pipeline_cost.completion_tokens + token_counter.completion_tokens,
            "total_tokens": pipeline_cost.total_tokens + token_counter.total_tokens,
        },
    }
    meta = current_trace.setdefault("meta", {})
    meta["token_cost"] = final_token_cost

    # Final gate metrics
    act = metrics["input_activation_rate"]
    bnd = metrics["input_boundary_non_activation_rate"]
    out = metrics["output_score"]
    chn = metrics["best_chain_score"]
    gate1 = act >= input_activation_threshold and bnd >= input_boundary_threshold
    gate2 = out >= output_score_threshold
    gate3 = chn >= score_threshold
    meta["final_metrics"] = {
        "input_activation_rate": round(act, 4),
        "input_boundary_non_activation_rate": round(bnd, 4),
        "output_score": round(out, 4),
        "chain_judge_score": int(chn),
    }
    meta["gate_pass"] = {"gate1": gate1, "gate2": gate2, "gate3": gate3}

    final_trace_path = feature_dir / current_timestamp / "trace.json"
    final_trace_path.write_text(json.dumps(current_trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Also write the final trace at the feature level for easy access (same level as round folders)
    feature_level_trace_path = feature_dir / f"{initial_timestamp}_final_trace.json"
    feature_level_trace_path.write_text(json.dumps(current_trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    skill_learning_summary: Dict[str, Any] = {"auto_cases_written": [], "case_usage_events": []}
    if skill_learning_enabled and skills_dir is not None:
        try:
            case_usage_events = _record_used_skill_cases(
                round_records=round_records,
                initial_metrics=initial_prior_best_metrics,
                skills_dir=skills_dir,
                layer_id=layer_id,
                feature_id=feature_id,
                score_threshold=score_threshold,
                input_activation_threshold=input_activation_threshold,
                input_boundary_threshold=input_boundary_threshold,
                output_score_threshold=output_score_threshold,
                post_write=_post_skill_write,
            )
            auto_cases_written = _write_feature_end_cases(
                round_records=round_records,
                initial_metrics=initial_prior_best_metrics,
                skills_dir=skills_dir,
                layer_id=layer_id,
                feature_id=feature_id,
                score_threshold=score_threshold,
                input_activation_threshold=input_activation_threshold,
                input_boundary_threshold=input_boundary_threshold,
                output_score_threshold=output_score_threshold,
                post_write=_post_skill_write,
            )
            skill_learning_summary = {
                "auto_cases_written": auto_cases_written,
                "case_usage_events": case_usage_events,
            }
            if auto_cases_written or case_usage_events:
                _log(
                    f"Skill learning: wrote {len(auto_cases_written)} feature-end case(s), "
                    f"recorded {len(case_usage_events)} case use event(s)",
                    log_file,
                )
        except Exception as exc:
            skill_learning_summary = {"error": str(exc), "auto_cases_written": [], "case_usage_events": []}
            _log(f"Skill learning finalization failed (non-fatal): {exc}", log_file)

    skill_tool_events = _collect_skill_tool_events(agent_log)

    summary = {
        "status": loop_status,
        "failure": failure_info,
        "layer_id": layer_id,
        "feature_id": feature_id,
        "initial_timestamp": initial_timestamp,
        "final_timestamp": current_timestamp,
        "input_source": input_observation_source,
        "bos_prompt_id": _trace_bos_prompt_id(current_trace) if input_observation_source == "bos_token" else None,
        "rounds_run": rounds_started,
        "agent_loop_token_cost": token_counter.as_dict(),
        "final_chain_score": best_score,
        "final_input_activation_rate": metrics["input_activation_rate"],
        "final_input_boundary_non_activation_rate": metrics["input_boundary_non_activation_rate"],
        "final_output_score": metrics["output_score"],
        "thresholds": {
            "chain_score_threshold": score_threshold,
            "input_activation_threshold": input_activation_threshold,
            "input_boundary_threshold": input_boundary_threshold,
            "output_score_threshold": output_score_threshold,
        },
        "final_chain_explanation": current_trace.get("chain", {}).get("chain_explanation", ""),
        "agent_log": agent_log,
        "round_skill_records": round_records,
        "skill_tool_events": skill_tool_events,
        "skill_learning": skill_learning_summary,
    }
    summary_path = feature_dir / initial_timestamp / "agent_loop_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _log(
        f"=== Done: final_ts={current_timestamp}  score={best_score}/5  "
        f"rounds={len(prev_timestamps)}  pipeline_tokens={pipeline_cost.total_tokens}  agent_tokens={token_counter.total_tokens} ===",
        log_file,
    )
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 3: agent loop for one SAE feature.")
    parser.add_argument("--layer-id", required=True)
    parser.add_argument("--feature-id", required=True)
    parser.add_argument("--initial-timestamp", required=True, help="Timestamp of the existing base trace.json")
    parser.add_argument("--model-id", default="gemma-2-2b")

    parser.add_argument("--sae-path", required=True)
    parser.add_argument("--model-checkpoint-path", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--inference-server-url", default=DEFAULT_INFERENCE_SERVER_URL)
    parser.add_argument("--inference-timeout-sec", type=float, default=DEFAULT_INFERENCE_TIMEOUT_SEC)
    parser.add_argument("--no-inference-server", action="store_true",
                        help="Use local model loading for steps 4/5 and BOS scans instead of the inference server")
    parser.add_argument(
        "--input-observation-source",
        choices=("neuronpedia", "bos_token"),
        default="neuronpedia",
    )
    parser.add_argument("--bos-token-root", type=Path, default=Path("initial_observation"))
    parser.add_argument("--bos-prompt-id", default="prompt-0001")

    parser.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--llm-api-key-file", default=DEFAULT_API_KEY_FILE)

    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--score-threshold", type=int, default=4)
    parser.add_argument("--input-activation-threshold", type=float, default=0.8)
    parser.add_argument("--input-boundary-threshold", type=float, default=0.8)
    parser.add_argument("--output-score-threshold", type=float, default=0.5)
    parser.add_argument("--skills-dir", type=Path, default=None, help="Skills directory to read/write (default: ./skills)")
    parser.add_argument("--logs-root", type=Path, default=Path("logs"), help="Root directory for feature logs (default: ./logs)")
    parser.add_argument("--disable-skill-learning", action="store_true", help="Read skills but do not expose write_skill or append auto cases")
    parser.add_argument("--skill-maintenance", choices=("off", "auto"), default="off")
    parser.add_argument("--skill-maintenance-archive-dir", type=Path, default=None)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    log_path = (
        build_feature_dir(layer_id=args.layer_id, feature_id=args.feature_id, logs_root=args.logs_root)
        / args.initial_timestamp
        / "agent_loop.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary = run_agent_loop(
        layer_id=str(args.layer_id),
        feature_id=str(args.feature_id),
        initial_timestamp=str(args.initial_timestamp),
        sae_path=str(args.sae_path),
        model_path=str(args.model_checkpoint_path),
        llm_base_url=str(args.llm_base_url),
        llm_model=str(args.llm_model),
        llm_api_key_file=args.llm_api_key_file,
        device=str(args.device),
        model_id=str(args.model_id),
        max_rounds=int(args.max_rounds),
        score_threshold=int(args.score_threshold),
        input_activation_threshold=float(args.input_activation_threshold),
        input_boundary_threshold=float(args.input_boundary_threshold),
        output_score_threshold=float(args.output_score_threshold),
        log_file=log_path,
        skills_dir=args.skills_dir,
        skill_learning_enabled=not bool(args.disable_skill_learning),
        skill_maintenance_mode=str(args.skill_maintenance),
        skill_maintenance_archive_dir=args.skill_maintenance_archive_dir,
        inference_server_url=str(args.inference_server_url),
        inference_timeout_sec=float(args.inference_timeout_sec),
        no_inference_server=bool(args.no_inference_server),
        logs_root=args.logs_root,
        input_observation_source=str(args.input_observation_source),
        bos_token_root=(
            args.bos_token_root
            if args.bos_token_root.is_absolute()
            else CODE_DIR / args.bos_token_root
        ),
        bos_prompt_id=str(args.bos_prompt_id),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
