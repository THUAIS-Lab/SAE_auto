#!/usr/bin/env python3
"""
run_main_experiment.py - Full pipeline + agent loop on Gemma-Scope 16k SAE.

Runs step1-9 (initial pipeline) followed by the agent loop (max_turns=5)
for each selected feature. Supports layer-level splitting for multi-GPU setups.

Single-GPU usage (all 5 layers):
  python run_main_experiment.py --timestamp 20260613_200000 --device cuda:0

Multi-GPU split example (if you have 5 GPUs):
  python run_main_experiment.py --timestamp TS --layers 0    --device cuda:0 &
  python run_main_experiment.py --timestamp TS --layers 6    --device cuda:1 &
  python run_main_experiment.py --timestamp TS --layers 12   --device cuda:2 &
  python run_main_experiment.py --timestamp TS --layers 18   --device cuda:3 &
  python run_main_experiment.py --timestamp TS --layers 24   --device cuda:4 &

Resume: re-run with the same --timestamp; completed features (trace.json exists) are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

PYTHON = sys.executable
CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))

from function import append_line_locked, append_text_locked
from inference_client import DEFAULT_INFERENCE_SERVER_URL, DEFAULT_INFERENCE_TIMEOUT_SEC
from prepare_bos_feature_manifest import load_feature_manifest
from workflow_step_utils import build_layer_sae_paths
from support_info.llm_api_info import (
    api_key_file as DEFAULT_API_KEY_FILE,
    base_url as DEFAULT_BASE_URL,
    model_name as DEFAULT_MODEL_NAME,
)

LAYER_IDS = [0, 6, 12, 18, 24]
N_FEATURES_PER_LAYER = 100
DEFAULT_SAE_ROOT = os.environ.get(
    "SAE_ROOT",
    "gemma-scope-2b-pt-res",
)
SAE_PATHS_16K = build_layer_sae_paths(layer_ids=LAYER_IDS, width="16k", sae_root=DEFAULT_SAE_ROOT)
DEFAULT_MODEL_PATH = os.environ.get(
    "SAE_MODEL_CHECKPOINT_PATH",
    "google/gemma-2-2b",
)
DEFAULT_MAX_ROUNDS = 10


def _configured_sae_paths(args: argparse.Namespace) -> Dict[int, str]:
    return getattr(args, "sae_paths_16k", SAE_PATHS_16K)


def _resolve_logs_root(logs_root: Path) -> Path:
    path = Path(logs_root)
    return path if path.is_absolute() else CODE_DIR / path


def _resolve_code_path(path: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else CODE_DIR / value


def _get_feature_ids(layer: int, n: int, seed_offset: int = 0) -> List[int]:
    """Deterministic feature selection: sample n features from 16k SAE space."""
    rng = random.Random(layer * 31337 + 42 + seed_offset)
    candidates = list(range(0, 16384, 30))
    return sorted(rng.sample(candidates, min(n, len(candidates))))


def _uniform_subsample(items: List[int], n: int) -> List[int]:
    if n >= len(items):
        return list(items)
    if n <= 0:
        return []
    if n == 1:
        return [items[len(items) // 2]]
    last = len(items) - 1
    return [items[round(i * last / (n - 1))] for i in range(n)]


def _dedupe_sorted_feature_ids(feature_ids: List[int]) -> List[int]:
    return sorted(dict.fromkeys(feature_ids))


def _parse_layer_feature_ids(values: Optional[List[str]]) -> Dict[int, List[int]]:
    parsed: Dict[int, List[int]] = {}
    if not values:
        return parsed
    for item in values:
        if ":" not in item:
            raise ValueError(f"expected LAYER:ID,ID,..., got {item!r}")
        layer_text, ids_text = item.split(":", 1)
        layer = int(layer_text)
        ids = [int(x) for x in ids_text.replace(";", ",").split(",") if x.strip()]
        parsed[layer] = _dedupe_sorted_feature_ids(ids)
    return parsed


def _select_feature_ids(args: argparse.Namespace, layer: int) -> List[int]:
    if args.layer_feature_id_map:
        return args.layer_feature_id_map[layer]
    if args.feature_ids:
        return _dedupe_sorted_feature_ids(args.feature_ids)
    base = _get_feature_ids(layer, args.n_features)
    if args.feature_subsample is not None:
        return _uniform_subsample(base, args.feature_subsample)
    return base


def _validate_bos_prompt_artifacts(
    *,
    bos_token_root: Path,
    prompt_id: str,
    layer_feature_ids: Dict[int, List[int]],
) -> None:
    problems: List[str] = []
    for layer_id, feature_ids in sorted(layer_feature_ids.items()):
        for feature_id in feature_ids:
            path = (
                Path(bos_token_root)
                / f"layer-{int(layer_id)}"
                / f"feature-{int(feature_id)}"
                / "bos_token"
                / str(prompt_id)
                / "top_tokens.json"
            )
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                problems.append(f"{path}: {exc}")
                continue
            expected = {
                "layer_id": int(layer_id),
                "feature_id": int(feature_id),
                "prompt_id": str(prompt_id),
            }
            mismatches = {
                key: payload.get(key)
                for key, value in expected.items()
                if payload.get(key) != value
            }
            if mismatches:
                problems.append(f"{path}: metadata mismatch {mismatches}, expected {expected}")
    if problems:
        preview = "\n".join(f"  - {problem}" for problem in problems[:20])
        suffix = "" if len(problems) <= 20 else f"\n  ... and {len(problems) - 20} more"
        raise ValueError(
            f"BOS prompt preflight failed for {len(problems)} feature(s):\n{preview}{suffix}"
        )


def _log(msg: str, log_file: Path) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    append_line_locked(log_file, line)


def _run_step(cmd: List[str], *, label: str, log_file: Path) -> bool:
    """Run a subprocess step, streaming output to log. Returns True on success."""
    _log(f"  -> {label}", log_file)
    proc = subprocess.Popen(
        cmd,
        cwd=CODE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        append_text_locked(log_file, line)
    rc = proc.wait()
    if rc != 0:
        _log(f"  FAILED {label} (exit {rc})", log_file)
        return False
    return True


def _run_initial_pipeline(
    layer: int,
    fid: int,
    ts: str,
    sae_path: str,
    model_path: str,
    device: str,
    llm_base_url: str,
    llm_model: str,
    llm_api_key_file: Optional[str],
    inference_server_url: str,
    inference_timeout_sec: float,
    no_inference_server: bool,
    logs_root: Path,
    log_file: Path,
    input_observation_source: str = "neuronpedia",
    bos_token_root: Optional[Path] = None,
    bos_prompt_id: Optional[str] = None,
) -> bool:
    """Run step1-9 for one feature. Returns True if trace.json produced."""
    base = ["--layer-id", str(layer), "--feature-id", str(fid), "--timestamp", ts, "--logs-root", str(logs_root)]
    llm = ["--llm-base-url", llm_base_url, "--llm-model", llm_model]
    if llm_api_key_file:
        llm += ["--llm-api-key-file", llm_api_key_file]
    inference_args = ["--no-inference-server"] if no_inference_server else [
        "--inference-server-url", inference_server_url,
        "--inference-timeout-sec", str(inference_timeout_sec),
    ]

    step1 = [
        PYTHON, "step1_initial_observation.py",
        "--model-id", "gemma-2-2b", "--layer-id", str(layer),
        "--feature-id", str(fid), "--width", "16k", "--timestamp", ts,
        "--logs-root", str(logs_root),
        "--observation-m", "10", "--observation-n", "5",
        "--input-observation-source", str(input_observation_source),
    ]
    if input_observation_source == "bos_token":
        if bos_token_root is None or not str(bos_prompt_id or "").strip():
            raise ValueError("BOS initial pipeline requires bos_token_root and bos_prompt_id")
        step1.extend(["--bos-token-root", str(bos_token_root), "--bos-prompt-id", str(bos_prompt_id)])

    steps = [
        ("step1", step1),
        ("step2", [PYTHON, "step2_generate_input_hypotheses.py",
                   *base, "--num-hypothesis", "3", *llm]),
        ("step3", [PYTHON, "step3_design_input_experiments.py",
                   *base, "--num-sentences-per-hypothesis", "5", *llm]),
        ("step4", [PYTHON, "step4_score_input_experiments.py",
                   *base, "--width", "16k", "--sae-path", sae_path,
                   "--model-checkpoint-path", model_path, "--device", device,
                   *inference_args]),
        ("step5", [PYTHON, "step5_run_intervention.py",
                   *base, "--width", "16k", "--sae-path", sae_path,
                   "--model-checkpoint-path", model_path, "--device", device,
                   "--top-k", "30", "--intervention-scope", "max_activation_token",
                   *inference_args]),
        ("step6", [PYTHON, "step6_generate_output_hypotheses.py", *base, *llm]),
        ("step7", [PYTHON, "step7_score_output_hypotheses.py", *base, *llm]),
        ("step8", [PYTHON, "step8_build_chain_hypotheses.py", *base, *llm]),
        ("step9", [PYTHON, "step9_synthesize_chain_explanation.py", *base, *llm]),
    ]

    label = f"L{layer}-F{fid}"
    for name, cmd in steps:
        if not _run_step(cmd, label=f"{label}/{name}", log_file=log_file):
            return False

    return _trace_path(layer, fid, ts, logs_root).exists()


def _run_initial_feature(args: argparse.Namespace, layer: int, fid: int, sae_path: str, log_file: Path) -> Tuple[str, bool]:
    label = f"L{layer}-F{fid}"
    _log(f"Init pipeline: {label}", log_file)
    ok = _run_initial_pipeline(
        layer,
        fid,
        args.timestamp,
        sae_path,
        args.model_path,
        args.device,
        args.llm_base_url,
        args.llm_model,
        args.llm_api_key_file,
        args.inference_server_url,
        args.inference_timeout_sec,
        args.no_inference_server,
        args.logs_root,
        log_file,
        args.input_observation_source,
        args.bos_token_root,
        args.bos_prompt_id,
    )
    return label, ok


def _run_initial_phase(
    args: argparse.Namespace,
    *,
    layers: List[int],
    log_file: Path,
) -> Tuple[List[Tuple[int, int]], int, int, int]:
    all_features: List[Tuple[int, int]] = []
    init_skip = 0
    init_ok = 0
    init_fail = 0
    jobs: List[Tuple[int, int, str]] = []

    for layer in layers:
        selected_ids = _select_feature_ids(args, layer)
        all_features.extend((layer, fid) for fid in selected_ids)
        sae_path = _configured_sae_paths(args).get(layer)
        if not sae_path or not Path(sae_path).exists():
            init_fail += len(selected_ids)
            _log(
                f"Layer {layer}: SAE path missing ({sae_path or 'not configured'}); "
                f"marking {len(selected_ids)} selected feature(s) as failed",
                log_file,
            )
            continue
        for fid in selected_ids:
            trace = _trace_path(layer, fid, args.timestamp, args.logs_root)
            if trace.exists() and not args.force:
                init_skip += 1
                continue
            jobs.append((layer, fid, sae_path))

    if args.feature_workers == 1:
        for layer, fid, sae_path in jobs:
            _, ok = _run_initial_feature(args, layer, fid, sae_path, log_file)
            if ok:
                init_ok += 1
            else:
                init_fail += 1
        return all_features, init_ok, init_skip, init_fail

    max_workers = min(args.feature_workers, len(jobs)) if jobs else 1
    if jobs:
        _log(f"Init phase: running {len(jobs)} feature(s) with feature_workers={max_workers}", log_file)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_label = {
            executor.submit(_run_initial_feature, args, layer, fid, sae_path, log_file): f"L{layer}-F{fid}"
            for layer, fid, sae_path in jobs
        }
        for future in as_completed(future_to_label):
            label = future_to_label[future]
            try:
                _, ok = future.result()
            except Exception as exc:
                init_fail += 1
                _log(f"  {label} init error: {exc}", log_file)
                continue
            if ok:
                init_ok += 1
            else:
                init_fail += 1

    return all_features, init_ok, init_skip, init_fail


def _trace_path(layer: int, fid: int, timestamp: str, logs_root: Path) -> Path:
    return _feature_dir(layer, fid, logs_root) / timestamp / "trace.json"


def _feature_dir(layer: int, fid: int, logs_root: Path) -> Path:
    return Path(logs_root) / f"layer-{layer}" / f"feature-{fid}"


def _agent_summary_path(layer: int, fid: int, timestamp: str, logs_root: Path) -> Path:
    return _feature_dir(layer, fid, logs_root) / timestamp / "agent_loop_summary.json"


def _agent_final_trace_path(layer: int, fid: int, timestamp: str, logs_root: Path) -> Path:
    return _feature_dir(layer, fid, logs_root) / f"{timestamp}_final_trace.json"


def _has_incomplete_agent_round(layer: int, fid: int, timestamp: str, logs_root: Path) -> bool:
    feature_dir = _feature_dir(layer, fid, logs_root)
    if not feature_dir.exists():
        return False
    pattern = re.compile(rf"^{re.escape(timestamp)}_r\d+$")
    for child in feature_dir.iterdir():
        if child.is_dir() and pattern.match(child.name) and not (child / "trace.json").exists():
            return True
    return False


def _read_agent_summary(layer: int, fid: int, timestamp: str, logs_root: Path) -> Optional[Dict[str, Any]]:
    path = _agent_summary_path(layer, fid, timestamp, logs_root)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _agent_loop_state(args: argparse.Namespace, layer: int, fid: int) -> Tuple[str, str]:
    if not _trace_path(layer, fid, args.timestamp, args.logs_root).exists():
        return "missing_trace", "no trace.json"
    if getattr(args, "force_agent", False):
        return "run", "--force-agent set"
    if _has_incomplete_agent_round(layer, fid, args.timestamp, args.logs_root):
        return "resume", "incomplete agent round exists"

    summary = _read_agent_summary(layer, fid, args.timestamp, args.logs_root)
    final_trace_exists = _agent_final_trace_path(layer, fid, args.timestamp, args.logs_root).exists()
    if summary is None:
        return "run", "no agent_loop_summary.json"

    status = summary.get("status")
    if final_trace_exists and status in (None, "complete"):
        return "skip", "agent_loop_summary.json and final trace exist"
    return "resume", f"agent summary status={status or 'unknown'}"


def _log_grouped_features(title: str, features: List[Tuple[int, int]], layers: List[int], log_file: Path) -> None:
    grouped: Dict[int, List[int]] = {layer: [] for layer in layers}
    for layer, fid in features:
        grouped.setdefault(layer, []).append(fid)
    _log(f"{title} ({len(features)}):", log_file)
    if not features:
        _log("  none", log_file)
        return
    for layer in layers:
        ids = grouped.get(layer, [])
        if ids:
            _log(f"  layer {layer}: {' '.join(map(str, ids))}", log_file)


def _dry_run(args: argparse.Namespace, layers: List[int], log_file: Path) -> None:
    selected: List[Tuple[int, int]] = []
    init_would_run: List[Tuple[int, int]] = []
    init_would_skip: List[Tuple[int, int]] = []
    agent_would_run: List[Tuple[int, int]] = []
    agent_would_resume: List[Tuple[int, int]] = []
    agent_would_skip: List[Tuple[int, int]] = []
    agent_missing_trace: List[Tuple[int, int]] = []

    for layer in layers:
        sae_path = _configured_sae_paths(args).get(layer)
        if not sae_path or not Path(sae_path).exists():
            _log(f"Layer {layer}: SAE path missing, would skip layer", log_file)
            continue
        for fid in _select_feature_ids(args, layer):
            selected.append((layer, fid))
            trace_exists = _trace_path(layer, fid, args.timestamp, args.logs_root).exists()
            if trace_exists and not args.force:
                init_would_skip.append((layer, fid))
            else:
                init_would_run.append((layer, fid))

            state, _reason = _agent_loop_state(args, layer, fid)
            if state == "missing_trace":
                agent_missing_trace.append((layer, fid))
            elif state == "skip":
                agent_would_skip.append((layer, fid))
            elif state == "resume":
                agent_would_resume.append((layer, fid))
            else:
                agent_would_run.append((layer, fid))

    _log("=== Dry Run: no experiment steps will be executed ===", log_file)
    _log_grouped_features("Selected features", selected, layers, log_file)
    _log_grouped_features("Init pipeline would run", init_would_run, layers, log_file)
    _log_grouped_features("Init pipeline would skip (trace.json exists)", init_would_skip, layers, log_file)
    if args.skip_agent:
        _log("Agent loop would be skipped because --skip-agent is set", log_file)
    else:
        _log_grouped_features("Agent loop would run new", agent_would_run, layers, log_file)
        _log_grouped_features("Agent loop would resume failed/incomplete", agent_would_resume, layers, log_file)
        _log_grouped_features("Agent loop would skip complete", agent_would_skip, layers, log_file)
        _log_grouped_features("Agent loop currently lacks trace.json", agent_missing_trace, layers, log_file)


def _run_one_agent_loop(args: argparse.Namespace, layer: int, fid: int, log_file: Path) -> Tuple[str, str]:
    from agent_runner import run_agent_loop

    label = f"L{layer}-F{fid}"
    state, reason = _agent_loop_state(args, layer, fid)
    if state == "missing_trace":
        return label, "SKIP (no trace.json)"
    if state == "skip":
        return label, "SKIP (agent complete)"

    _log(f"Agent loop: {label} ({reason})", log_file)
    sae_path = _configured_sae_paths(args)[layer]
    try:
        summary = run_agent_loop(
            layer_id=str(layer),
            feature_id=str(fid),
            initial_timestamp=args.timestamp,
            sae_path=sae_path,
            model_path=args.model_path,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
            llm_api_key_file=args.llm_api_key_file,
            device=args.device,
            model_id="gemma-2-2b",
            max_rounds=args.max_rounds,
            score_threshold=args.score_threshold,
            input_activation_threshold=args.input_activation_threshold,
            input_boundary_threshold=args.input_boundary_threshold,
            output_score_threshold=args.output_score_threshold,
            log_file=log_file,
            skills_dir=args.skills_dir,
            skill_learning_enabled=not bool(args.disable_skill_learning),
            skill_maintenance_mode=args.skill_maintenance,
            skill_maintenance_archive_dir=args.skill_maintenance_archive_dir,
            inference_server_url=args.inference_server_url,
            inference_timeout_sec=args.inference_timeout_sec,
            no_inference_server=args.no_inference_server,
            logs_root=args.logs_root,
            input_observation_source=args.input_observation_source,
            bos_token_root=args.bos_token_root,
            bos_prompt_id=args.bos_prompt_id,
        )
        score = summary.get("final_chain_score", 0)
        rounds = summary.get("rounds_run", 0)
        status = summary.get("status", "complete")
        if status != "complete":
            return label, f"{status} score={score}/5 rounds={rounds}"
        return label, f"OK score={score}/5 rounds={rounds}"
    except Exception as exc:
        _log(f"  {label} agent error: {exc}", log_file)
        return label, f"ERROR: {exc}"


def _run_agent_loop_all(
    args: argparse.Namespace,
    *,
    features: List[Tuple[int, int]],
    log_file: Path,
) -> Dict[str, Any]:
    """Run agent loop on all features that have a trace.json."""
    results: Dict[str, str] = {}

    if args.feature_workers == 1:
        for layer, fid in features:
            label, status = _run_one_agent_loop(args, layer, fid, log_file)
            results[label] = status
        return results

    _log(f"Agent phase: running {len(features)} feature(s) with feature_workers={args.feature_workers}", log_file)
    max_workers = min(args.feature_workers, len(features)) if features else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_label = {
            executor.submit(_run_one_agent_loop, args, layer, fid, log_file): f"L{layer}-F{fid}"
            for layer, fid in features
        }
        for future in as_completed(future_to_label):
            label = future_to_label[future]
            try:
                got_label, status = future.result()
            except Exception as exc:
                got_label, status = label, f"ERROR: {exc}"
                _log(f"  {label} agent error: {exc}", log_file)
            results[got_label] = status

    return {f"L{layer}-F{fid}": results.get(f"L{layer}-F{fid}", "SKIP (no trace.json)") for layer, fid in features}


def _agent_result_failed(status: str) -> bool:
    normalized = str(status).strip().lower()
    return (
        normalized.startswith("error")
        or normalized.startswith("failed")
        or normalized.startswith("skip (no trace")
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Full pipeline + agent loop, 16k SAE.")
    p.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"),
                   help="Run timestamp (default: now). Reuse to resume.")
    p.add_argument("--logs-root", type=Path, default=Path("logs"),
                   help="Root directory for feature logs (default: ./logs).")
    p.add_argument(
        "--input-observation-source",
        choices=("neuronpedia", "bos_token"),
        default="neuronpedia",
        help="Immutable input observation source for initial and retry pipelines (default: neuronpedia)",
    )
    p.add_argument(
        "--bos-token-root",
        type=Path,
        default=Path("initial_observation"),
        help="BOS observation root used only when --input-observation-source=bos_token",
    )
    p.add_argument(
        "--bos-prompt-id",
        default="prompt-0001",
        help="Initial feature-local BOS prompt id (default: prompt-0001)",
    )
    p.add_argument("--layers", nargs="*", type=int, default=None,
                   help="Layer IDs to process (default: all 5 layers)")
    p.add_argument("--n-features", type=int, default=N_FEATURES_PER_LAYER,
                   help="Features per layer (default: 100)")
    p.add_argument("--feature-ids", nargs="+", type=int, default=None,
                   help="Explicit feature IDs to process for every selected layer; overrides --n-features")
    p.add_argument("--layer-feature-ids", nargs="+", default=None, metavar="LAYER:ID,ID,...",
                   help="Explicit feature IDs per layer, e.g. 0:30,60 6:90,120")
    p.add_argument(
        "--feature-manifest",
        type=Path,
        default=None,
        help="JSON feature manifest from prepare_bos_feature_manifest.py",
    )
    p.add_argument("--feature-subsample", type=int, default=None,
                   help="Uniformly select this many IDs from each layer's --n-features list")
    p.add_argument("--feature-workers", type=int, default=1,
                   help="Number of feature pipelines to run concurrently in each phase (default: 1)")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument(
        "--sae-root",
        default=DEFAULT_SAE_ROOT,
        help="Local Gemma-Scope SAE root (default: SAE_ROOT or the current server path)",
    )
    p.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    p.add_argument("--score-threshold", type=int, default=4)
    p.add_argument("--input-activation-threshold", type=float, default=0.8)
    p.add_argument("--input-boundary-threshold", type=float, default=0.8)
    p.add_argument("--output-score-threshold", type=float, default=0.5)
    p.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--llm-model", default=DEFAULT_MODEL_NAME)
    p.add_argument("--llm-api-key-file", default=DEFAULT_API_KEY_FILE)
    p.add_argument("--inference-server-url", default=DEFAULT_INFERENCE_SERVER_URL)
    p.add_argument("--inference-timeout-sec", type=float, default=DEFAULT_INFERENCE_TIMEOUT_SEC)
    p.add_argument("--no-inference-server", action="store_true",
                   help="Use local model loading for steps 4/5 instead of the inference server")
    p.add_argument("--skills-dir", type=Path, default=None,
                   help="Skills directory for the agent to read/write (default: ./skills)")
    p.add_argument("--disable-skill-learning", action="store_true",
                   help="Read skills but do not expose write_skill or append auto cases")
    p.add_argument(
        "--skill-maintenance",
        choices=("off", "auto"),
        default="off",
        help="Check thresholds after each successful skill append (default: off)",
    )
    p.add_argument(
        "--skill-maintenance-archive-dir",
        type=Path,
        default=None,
        help="Archive directory for automatic skill maintenance",
    )
    p.add_argument(
        "--save-skills-checkpoint",
        action="store_true",
        help="Unsupported here; use run_self_evolution_train.py for skills checkpoints",
    )
    p.add_argument("--skip-agent", action="store_true",
                   help="Only run pipeline init (step1-9), skip agent loop")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if trace.json already exists")
    p.add_argument("--force-agent", action="store_true",
                   help="Re-run agent loop even if agent_loop_summary.json already looks complete")
    p.add_argument("--dry-run", action="store_true",
                   help="Print selected features and resume status without running steps")
    args = p.parse_args()

    if str(args.sae_root) == str(DEFAULT_SAE_ROOT):
        # Keep the module-level map injectable for tests and programmatic users.
        # An explicit --sae-root always takes precedence.
        args.sae_paths_16k = dict(SAE_PATHS_16K)
    else:
        args.sae_paths_16k = build_layer_sae_paths(
            layer_ids=LAYER_IDS,
            width="16k",
            sae_root=args.sae_root,
        )

    args.logs_root = _resolve_logs_root(args.logs_root)
    args.bos_token_root = _resolve_code_path(args.bos_token_root)
    if args.skill_maintenance_archive_dir is None:
        args.skill_maintenance_archive_dir = args.logs_root / "skill_case_archive"

    if args.save_skills_checkpoint:
        p.error("--save-skills-checkpoint requires run_self_evolution_train.py")

    if args.feature_workers < 1:
        p.error("--feature-workers must be >= 1")
    selection_modes = [
        bool(args.feature_ids),
        bool(args.layer_feature_ids),
        args.feature_manifest is not None,
        args.feature_subsample is not None,
    ]
    if sum(selection_modes) > 1:
        p.error(
            "Use only one of --feature-ids, --layer-feature-ids, "
            "--feature-manifest, or --feature-subsample"
        )
    if args.feature_subsample is not None and args.feature_subsample < 1:
        p.error("--feature-subsample must be >= 1")

    try:
        if args.feature_manifest is not None:
            args.feature_manifest = _resolve_code_path(args.feature_manifest)
            args.layer_feature_id_map = load_feature_manifest(args.feature_manifest)
        else:
            args.layer_feature_id_map = _parse_layer_feature_ids(args.layer_feature_ids)
    except ValueError as exc:
        p.error(str(exc))
    except (OSError, json.JSONDecodeError) as exc:
        p.error(f"could not load --feature-manifest: {exc}")

    all_explicit_feature_ids = list(args.feature_ids or [])
    for ids in args.layer_feature_id_map.values():
        all_explicit_feature_ids.extend(ids)
    bad_feature_ids = [fid for fid in all_explicit_feature_ids if fid < 0 or fid >= 16384]
    if bad_feature_ids:
        p.error(f"feature IDs must be in [0, 16383], got: {bad_feature_ids}")

    if args.feature_workers > 1 and args.no_inference_server:
        p.error("--feature-workers > 1 requires the inference server. Remove --no-inference-server or set --feature-workers 1.")
    if args.input_observation_source == "bos_token" and not str(args.bos_prompt_id).strip():
        p.error("--bos-prompt-id must be non-empty for a BOS-token run")

    layers = args.layers or (sorted(args.layer_feature_id_map) if args.feature_manifest else LAYER_IDS)
    if args.layer_feature_id_map:
        missing_layers = [layer for layer in layers if layer not in args.layer_feature_id_map]
        if missing_layers:
            source_name = "--feature-manifest" if args.feature_manifest else "--layer-feature-ids"
            p.error(f"{source_name} missing selected layer(s): {missing_layers}")
        unknown_layers = [layer for layer in args.layer_feature_id_map if layer not in LAYER_IDS]
        if unknown_layers:
            source_name = "--feature-manifest" if args.feature_manifest else "--layer-feature-ids"
            p.error(f"unknown layer(s) in {source_name}: {unknown_layers}")
    if args.input_observation_source == "bos_token":
        try:
            _validate_bos_prompt_artifacts(
                bos_token_root=args.bos_token_root,
                prompt_id=args.bos_prompt_id,
                layer_feature_ids={layer: _select_feature_ids(args, layer) for layer in layers},
            )
        except ValueError as exc:
            p.error(str(exc))
    tag = f"gpu{args.device.replace(':', '')}"
    log_file = CODE_DIR / f"main_exp_{args.timestamp}_{tag}.log"

    _log("=== Main Experiment: 16k SAE ===", log_file)
    _log(f"timestamp={args.timestamp}  device={args.device}  layers={layers}  logs_root={args.logs_root}", log_file)
    _log(
        f"n_features={args.n_features}  max_rounds={args.max_rounds}  "
        f"model={args.llm_model}  feature_workers={args.feature_workers}  "
        f"skills_dir={args.skills_dir or (CODE_DIR / 'skills')}  "
        f"input_source={args.input_observation_source}  "
        f"bos_token_root={args.bos_token_root if args.input_observation_source == 'bos_token' else '(unused)'}  "
        f"skill_learning={not bool(args.disable_skill_learning)}  "
        f"skill_maintenance={args.skill_maintenance}",
        log_file,
    )
    if args.feature_ids or args.layer_feature_id_map or args.feature_subsample is not None:
        selected_by_layer = {layer: _select_feature_ids(args, layer) for layer in layers}
        _log(f"feature_ids_by_layer={selected_by_layer}", log_file)

    if args.dry_run:
        _dry_run(args, layers, log_file)
        return

    all_features, init_ok, init_skip, init_fail = _run_initial_phase(args, layers=layers, log_file=log_file)
    _log(f"Init done: {init_ok} OK, {init_skip} skipped, {init_fail} failed", log_file)

    if args.skip_agent:
        _log("--skip-agent set, stopping after init.", log_file)
        if init_fail:
            raise SystemExit(1)
        return

    _log(f"=== Agent Loop (max_rounds={args.max_rounds}) ===", log_file)
    agent_results = _run_agent_loop_all(args, features=all_features, log_file=log_file)

    _log("\n=== Summary ===", log_file)
    for key, status in agent_results.items():
        _log(f"  {key}: {status}", log_file)
    ok = sum(1 for s in agent_results.values() if s.startswith("OK"))
    _log(f"{ok}/{len(agent_results)} agent runs succeeded", log_file)
    failed_agents = {key: value for key, value in agent_results.items() if _agent_result_failed(value)}
    if init_fail or failed_agents:
        _log(
            f"Experiment finished with failures: init_fail={init_fail}, "
            f"agent_fail={len(failed_agents)}",
            log_file,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
