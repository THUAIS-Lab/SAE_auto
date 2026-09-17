#!/usr/bin/env python3
"""Utilities for the self-evolution experiment runners."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from run_main_experiment import CODE_DIR, LAYER_IDS, _get_feature_ids
from trace_metrics import extract_trace_metrics

DEFAULT_OUTPUT_ROOT = CODE_DIR / "analysis_output" / "self_evolution"
DEFAULT_TRAIN_PER_LAYER = 20
DEFAULT_VAL_PER_LAYER = 10
DEFAULT_CHECKPOINT_EVERY = 10
DEFAULT_VAL_SEED_OFFSET = 1000

THRESHOLDS = {
    "input_activation_threshold": 0.8,
    "input_boundary_threshold": 0.8,
    "output_score_threshold": 0.5,
    "chain_score_threshold": 4.0,
}


def timestamp_id(prefix: str = "selfev") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(
    *,
    layers: Sequence[int] = LAYER_IDS,
    train_per_layer: int = DEFAULT_TRAIN_PER_LAYER,
    val_per_layer: int = DEFAULT_VAL_PER_LAYER,
    train_seed_offset: int = 0,
    val_seed_offset: int = DEFAULT_VAL_SEED_OFFSET,
) -> Dict[str, Any]:
    train: Dict[str, List[int]] = {}
    val: Dict[str, List[int]] = {}
    for layer in layers:
        train_ids = list(_get_feature_ids(int(layer), int(train_per_layer), int(train_seed_offset)))
        train_set = set(train_ids)
        val_ids: List[int] = []
        seed = int(val_seed_offset)
        guard = 0
        while len(val_ids) < int(val_per_layer):
            guard += 1
            if guard > 10000:
                raise RuntimeError(f"Could not build validation split for layer {layer}")
            candidates = _get_feature_ids(int(layer), max(int(train_per_layer), int(val_per_layer)), seed)
            for fid in candidates:
                if fid in train_set or fid in val_ids:
                    continue
                val_ids.append(int(fid))
                if len(val_ids) >= int(val_per_layer):
                    break
            seed += 1
        train[str(layer)] = train_ids
        val[str(layer)] = val_ids

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "layers": [int(layer) for layer in layers],
        "train_per_layer": int(train_per_layer),
        "val_per_layer": int(val_per_layer),
        "train_seed_offset": int(train_seed_offset),
        "val_seed_offset": int(val_seed_offset),
        "train_features": train,
        "validation_features": val,
    }


def manifest_layers(manifest: Mapping[str, Any]) -> List[int]:
    return [int(layer) for layer in manifest.get("layers", LAYER_IDS)]


def feature_batches(manifest: Mapping[str, Any], split: str) -> List[Dict[int, int]]:
    key = "train_features" if split == "train" else "validation_features"
    features = manifest[key]
    layers = manifest_layers(manifest)
    per_layer = min(len(features[str(layer)]) for layer in layers)
    batches: List[Dict[int, int]] = []
    for idx in range(per_layer):
        batches.append({int(layer): int(features[str(layer)][idx]) for layer in layers})
    return batches


def _feature_id_values(value: Any) -> List[int]:
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]
    return [int(value)]


def layer_feature_arg(batch: Mapping[int, Any]) -> List[str]:
    args: List[str] = []
    for layer in sorted(batch):
        feature_ids = ",".join(str(fid) for fid in _feature_id_values(batch[layer]))
        args.append(f"{int(layer)}:{feature_ids}")
    return args


def checkpoint_names(
    total_features: int,
    every: int = DEFAULT_CHECKPOINT_EVERY,
    *,
    offset: int = 0,
) -> List[str]:
    if every <= 0:
        raise ValueError("checkpoint cadence must be positive")
    if offset < 0:
        raise ValueError("checkpoint offset must be non-negative")
    names = [f"ckpt_{int(offset):03d}"]
    for processed in range(every, int(total_features) + 1, every):
        names.append(f"ckpt_{int(offset) + processed:03d}")
    final_name = f"ckpt_{int(offset) + int(total_features):03d}"
    if names[-1] != final_name:
        names.append(final_name)
    return names


def checkpoint_train_features(name: str) -> int:
    try:
        return int(str(name).split("_", 1)[1])
    except Exception:
        return 0


def make_main_experiment_cmd(
    *,
    timestamp: str,
    batch: Mapping[int, Any],
    feature_workers: int = 5,
    skip_agent: bool = False,
    logs_root: Optional[Path] = None,
    skills_dir: Optional[Path] = None,
    disable_skill_learning: bool = False,
    skill_maintenance_mode: Optional[str] = None,
    skill_maintenance_archive_dir: Optional[Path] = None,
    force: bool = False,
    force_agent: bool = False,
    extra_args: Sequence[str] = (),
) -> List[str]:
    layers = [str(int(layer)) for layer in sorted(batch)]
    cmd = [
        sys.executable,
        "run_main_experiment.py",
        "--timestamp",
        str(timestamp),
        "--layers",
        *layers,
        "--layer-feature-ids",
        *layer_feature_arg(batch),
        "--feature-workers",
        str(int(feature_workers)),
    ]
    if skip_agent:
        cmd.append("--skip-agent")
    if logs_root is not None:
        cmd.extend(["--logs-root", str(logs_root)])
    if skills_dir is not None:
        cmd.extend(["--skills-dir", str(skills_dir)])
    if disable_skill_learning:
        cmd.append("--disable-skill-learning")
    if skill_maintenance_mode is not None:
        cmd.extend(["--skill-maintenance", str(skill_maintenance_mode)])
    if skill_maintenance_archive_dir is not None:
        cmd.extend(["--skill-maintenance-archive-dir", str(skill_maintenance_archive_dir)])
    if force:
        cmd.append("--force")
    if force_agent:
        cmd.append("--force-agent")
    cmd.extend(str(arg) for arg in extra_args)
    return cmd


def copy_skills_checkpoint(source_dir: Path, dest_dir: Path, *, overwrite: bool = True) -> Dict[str, Any]:
    if not source_dir.exists():
        raise FileNotFoundError(f"skills source does not exist: {source_dir}")
    if dest_dir.exists() and overwrite:
        shutil.rmtree(dest_dir)
    if not dest_dir.exists():
        shutil.copytree(
            source_dir,
            dest_dir,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".maintenance.lock"),
        )
    return {
        "path": str(dest_dir),
        "hash": hash_tree(dest_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def hash_tree(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return ""
    for item in sorted(
        p for p in path.rglob("*") if p.is_file() and p.name != ".maintenance.lock"
    ):
        rel = item.relative_to(path).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def feature_dir(layer: int, feature_id: int, *, logs_root: Optional[Path] = None) -> Path:
    root = logs_root if logs_root is not None else CODE_DIR / "logs"
    return root / f"layer-{int(layer)}" / f"feature-{int(feature_id)}"


def trace_path(layer: int, feature_id: int, timestamp: str, *, logs_root: Optional[Path] = None) -> Path:
    return feature_dir(layer, feature_id, logs_root=logs_root) / str(timestamp) / "trace.json"


def summary_path(layer: int, feature_id: int, timestamp: str, *, logs_root: Optional[Path] = None) -> Path:
    return feature_dir(layer, feature_id, logs_root=logs_root) / str(timestamp) / "agent_loop_summary.json"


def final_trace_path(layer: int, feature_id: int, timestamp: str, *, logs_root: Optional[Path] = None) -> Path:
    return feature_dir(layer, feature_id, logs_root=logs_root) / f"{timestamp}_final_trace.json"


def copy_base_trace_for_eval(
    *,
    layer: int,
    feature_id: int,
    base_timestamp: str,
    eval_timestamp: str,
    force: bool = False,
    logs_root: Optional[Path] = None,
) -> Path:
    src = feature_dir(layer, feature_id, logs_root=logs_root) / str(base_timestamp)
    dst = feature_dir(layer, feature_id, logs_root=logs_root) / str(eval_timestamp)
    if not src.exists():
        raise FileNotFoundError(f"base trace directory missing: {src}")
    if dst.exists():
        if not force:
            return dst
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("agent_loop_summary.json", "agent_proposal_r*.json"))
    return dst


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect_feature_result(
    *,
    layer: int,
    feature_id: int,
    timestamp: str,
    checkpoint: str,
    checkpoint_path: Optional[Path] = None,
    logs_root: Optional[Path] = None,
    thresholds: Mapping[str, float] = THRESHOLDS,
) -> Dict[str, Any]:
    ftrace = final_trace_path(layer, feature_id, timestamp, logs_root=logs_root)
    if not ftrace.exists():
        ftrace = trace_path(layer, feature_id, timestamp, logs_root=logs_root)
    trace = _load_json_if_exists(ftrace)
    metrics = extract_trace_metrics(trace) if trace else {
        "input_activation_rate": 0.0,
        "input_boundary_non_activation_rate": 0.0,
        "output_score": 0.0,
        "best_chain_score": 0.0,
    }
    summary = _load_json_if_exists(summary_path(layer, feature_id, timestamp, logs_root=logs_root))
    gate1 = (
        metrics["input_activation_rate"] >= float(thresholds["input_activation_threshold"])
        and metrics["input_boundary_non_activation_rate"] >= float(thresholds["input_boundary_threshold"])
    )
    gate2 = metrics["output_score"] >= float(thresholds["output_score_threshold"])
    gate3 = metrics["best_chain_score"] >= float(thresholds["chain_score_threshold"])
    agent_cost = summary.get("agent_loop_token_cost") or summary.get("agent_token_cost") or {}
    total_cost = (trace.get("meta", {}).get("token_cost", {}).get("total", {}) if trace else {})
    return {
        "checkpoint": checkpoint,
        "checkpoint_train_features": checkpoint_train_features(checkpoint),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else "",
        "timestamp": str(timestamp),
        "layer_id": int(layer),
        "feature_id": int(feature_id),
        "status": summary.get("status") or ("complete" if trace else "missing_trace"),
        "rounds_run": int(summary.get("rounds_run") or 0),
        "input_activation_rate": float(metrics["input_activation_rate"]),
        "input_boundary_non_activation_rate": float(metrics["input_boundary_non_activation_rate"]),
        "output_score": float(metrics["output_score"]),
        "chain_judge_score": float(metrics["best_chain_score"]),
        "gate1_pass": bool(gate1),
        "gate2_pass": bool(gate2),
        "gate3_pass": bool(gate3),
        "all_gates_pass": bool(gate1 and gate2 and gate3),
        "agent_total_tokens": int(agent_cost.get("total_tokens") or 0),
        "total_tokens": int(total_cost.get("total_tokens") or 0),
        "trace_path": str(ftrace),
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["checkpoint"]), []).append(row)
    out: List[Dict[str, Any]] = []
    for checkpoint, items in grouped.items():
        n = len(items)
        def avg(key: str) -> float:
            return float(mean(float(item.get(key, 0.0)) for item in items)) if items else 0.0
        def rate(key: str) -> float:
            return float(mean(1.0 if item.get(key) else 0.0 for item in items)) if items else 0.0
        out.append({
            "checkpoint": checkpoint,
            "checkpoint_train_features": checkpoint_train_features(checkpoint),
            "n_features": n,
            "mean_input_activation_rate": avg("input_activation_rate"),
            "mean_input_boundary_non_activation_rate": avg("input_boundary_non_activation_rate"),
            "mean_output_score": avg("output_score"),
            "mean_chain_judge_score": avg("chain_judge_score"),
            "gate1_pass_rate": rate("gate1_pass"),
            "gate2_pass_rate": rate("gate2_pass"),
            "gate3_pass_rate": rate("gate3_pass"),
            "all_gates_pass_rate": rate("all_gates_pass"),
            "mean_rounds_run": avg("rounds_run"),
            "mean_agent_total_tokens": avg("agent_total_tokens"),
            "mean_total_tokens": avg("total_tokens"),
        })
    return sorted(out, key=lambda row: int(row["checkpoint_train_features"]))


def plot_curves(summary_rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if not summary_rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(summary_rows, key=lambda row: int(row["checkpoint_train_features"]))
    x = [int(row["checkpoint_train_features"]) for row in rows]
    series = [
        ("all_gates_pass_rate", "All Gates Pass Rate"),
        ("mean_chain_judge_score", "Mean Chain Score"),
        ("mean_output_score", "Mean Output Score"),
        ("mean_input_activation_rate", "Mean Input Activation"),
        ("mean_input_boundary_non_activation_rate", "Mean Input Boundary"),
    ]
    fig, axes = plt.subplots(len(series), 1, figsize=(8, 12), sharex=True)
    for ax, (key, title) in zip(axes, series):
        y = [float(row.get(key, 0.0)) for row in rows]
        ax.plot(x, y, marker="o", linewidth=2)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if key.endswith("rate") or "input" in key or key == "mean_output_score":
            ax.set_ylim(-0.02, 1.02)
    axes[-1].set_xlabel("Training features accumulated in skills checkpoint")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
