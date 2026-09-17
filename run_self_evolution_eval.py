#!/usr/bin/env python3
"""Evaluate validation features against saved skills checkpoints."""
from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from run_main_experiment import CODE_DIR
from self_evolution_utils import (
    DEFAULT_OUTPUT_ROOT,
    collect_feature_result,
    copy_base_trace_for_eval,
    feature_batches,
    feature_dir,
    hash_tree,
    make_main_experiment_cmd,
    plot_curves,
    read_json,
    summarize_results,
    write_csv,
    write_jsonl,
)

SIDE_ROUND_KEYS = ("input", "output", "chain", "unknown")
MODE_TO_SIDE = {
    "input": "input",
    "input_validate": "input",
    "output": "output",
    "output_validate": "output",
    "chain": "chain",
}


def _latest_experiment(output_root: Path) -> str:
    candidates = [p for p in output_root.iterdir() if p.is_dir()] if output_root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"no experiments found under {output_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime).name


def _parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluate validation features using saved self-evolution skills checkpoints."
    )
    parser.add_argument("--experiment-id", default=None, help="Experiment id under analysis_output/self_evolution")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--checkpoints-index", type=Path, default=None)
    parser.add_argument("--eval-base-timestamp", default=None)
    parser.add_argument("--eval-timestamp-prefix", default=None)
    parser.add_argument("--logs-root", type=Path, default=None)
    parser.add_argument("--feature-workers", type=int, default=5)
    parser.add_argument(
        "--eval-batch-mode",
        choices=("all", "per-index"),
        default="all",
        help=(
            "How to pass validation features to run_main_experiment.py. "
            "'all' sends every selected validation feature in one call per checkpoint; "
            "'per-index' preserves the older 5-feature batches."
        ),
    )
    parser.add_argument("--checkpoints", nargs="*", default=None, help="Optional checkpoint names to evaluate")
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None, help="Optional smoke-test validation batch limit")
    parser.add_argument("--force-base", action="store_true", help="Force re-run validation base traces")
    parser.add_argument("--force-copy-base", action="store_true", help="Reset eval timestamp dirs before each checkpoint run")
    parser.add_argument("--force-agent", action="store_true", help="Forward --force-agent to run_main_experiment.py")
    parser.add_argument(
        "--allow-shared-logs-root",
        action="store_true",
        help=(
            "Allow evaluating multiple checkpoints under one logs root. "
            "Not recommended: use one isolated --logs-root per checkpoint and aggregate afterwards."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    if args.experiment_id is None:
        args.experiment_id = _latest_experiment(args.output_root)
    if args.eval_base_timestamp is None:
        args.eval_base_timestamp = f"{args.experiment_id}_eval_base"
    if args.eval_timestamp_prefix is None:
        args.eval_timestamp_prefix = f"{args.experiment_id}_eval"
    return args, extra


def _select_checkpoints(index: Dict[str, Any], names: Sequence[str] | None, max_count: int | None) -> List[Dict[str, Any]]:
    checkpoints = list(index.get("checkpoints", []))
    checkpoints.sort(key=lambda item: int(item.get("processed_features", 0)))
    if names:
        wanted = set(names)
        checkpoints = [item for item in checkpoints if item.get("name") in wanted]
    if max_count is not None:
        checkpoints = checkpoints[: max(0, int(max_count))]
    return checkpoints


def _feature_values(value: Any) -> List[int]:
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]
    return [int(value)]


def _iter_batch_features(batch: Mapping[int, Any]) -> List[tuple[int, int]]:
    features: List[tuple[int, int]] = []
    for layer in sorted(batch):
        for fid in _feature_values(batch[layer]):
            features.append((int(layer), int(fid)))
    return features


def _combine_batches(batches: Sequence[Mapping[int, int]]) -> List[Dict[int, List[int]]]:
    if not batches:
        return []
    combined: Dict[int, List[int]] = {}
    for batch in batches:
        for layer, fid in batch.items():
            combined.setdefault(int(layer), []).append(int(fid))
    return [combined]


def _reset_eval_artifacts(layer: int, feature_id: int, eval_timestamp: str, logs_root: Path | None = None) -> None:
    fdir = feature_dir(layer, feature_id, logs_root=logs_root)
    if not fdir.exists():
        return
    for child in fdir.iterdir():
        if child.is_dir() and (child.name == eval_timestamp or child.name.startswith(f"{eval_timestamp}_r")):
            shutil.rmtree(child)
        elif child.is_file() and child.name == f"{eval_timestamp}_final_trace.json":
            child.unlink()


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = read_json(path)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _side_from_mode(mode: Any) -> str:
    return MODE_TO_SIDE.get(str(mode or "").strip(), "unknown")


def _agent_side_round_counts(
    layer: int,
    feature_id: int,
    timestamp: str,
    *,
    logs_root: Path | None = None,
) -> Dict[str, Any]:
    summary_file = feature_dir(layer, feature_id, logs_root=logs_root) / str(timestamp) / "agent_loop_summary.json"
    summary = _load_json_if_exists(summary_file)
    raw_records = summary.get("round_skill_records")
    records = raw_records if isinstance(raw_records, list) else []
    counts = {side: 0 for side in SIDE_ROUND_KEYS}
    for record in records:
        if not isinstance(record, dict):
            counts["unknown"] += 1
            continue
        counts[_side_from_mode(record.get("mode"))] += 1
    out: Dict[str, Any] = {f"agent_{side}_rounds": counts[side] for side in SIDE_ROUND_KEYS}
    out["agent_counted_rounds"] = sum(counts.values())
    out["agent_side_rounds_source"] = "round_skill_records" if isinstance(raw_records, list) else "none"
    return out


def _collect_feature_result_with_side_rounds(
    *,
    layer: int,
    feature_id: int,
    timestamp: str,
    checkpoint: str,
    checkpoint_path: Path,
    logs_root: Path | None = None,
) -> Dict[str, Any]:
    row = collect_feature_result(
        layer=layer,
        feature_id=feature_id,
        timestamp=timestamp,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        logs_root=logs_root,
    )
    row.update(_agent_side_round_counts(layer, feature_id, timestamp, logs_root=logs_root))
    return row


def _summarize_results_with_side_rounds(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    summary_rows = summarize_results(rows)
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["checkpoint"]), []).append(row)

    for summary in summary_rows:
        items = grouped.get(str(summary["checkpoint"]), [])

        def avg(key: str) -> float:
            return sum(float(item.get(key, 0.0) or 0.0) for item in items) / len(items) if items else 0.0

        summary["mean_agent_input_rounds_per_feature"] = avg("agent_input_rounds")
        summary["mean_agent_output_rounds_per_feature"] = avg("agent_output_rounds")
        summary["mean_agent_chain_rounds_per_feature"] = avg("agent_chain_rounds")
        summary["mean_agent_unknown_rounds_per_feature"] = avg("agent_unknown_rounds")
        summary["mean_agent_counted_rounds_per_feature"] = avg("agent_counted_rounds")
        summary["agent_side_rounds_feature_coverage"] = (
            sum(1 for item in items if item.get("agent_side_rounds_source") != "none") / len(items)
            if items
            else 0.0
        )
    return summary_rows


def _side_rounds_path(curves_path: Path) -> Path:
    suffix = curves_path.suffix or ".png"
    base = curves_path.with_suffix("")
    return base.with_name(f"{base.name}_side_rounds").with_suffix(suffix)


def _plot_side_rounds(summary_rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if not summary_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(summary_rows, key=lambda row: int(row["checkpoint_train_features"]))
    x = [int(row["checkpoint_train_features"]) for row in rows]
    series = [
        ("mean_agent_input_rounds_per_feature", "Input"),
        ("mean_agent_output_rounds_per_feature", "Output"),
        ("mean_agent_chain_rounds_per_feature", "Chain"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, label in series:
        y = [float(row.get(key, 0.0) or 0.0) for row in rows]
        ax.plot(x, y, marker="o", linewidth=2, label=label)
    ax.set_title("Mean Agent Rounds Per Feature by Side")
    ax.set_xlabel("Training features accumulated in skills checkpoint")
    ax.set_ylabel("Mean rounds per feature")
    ax.set_ylim(bottom=-0.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _run_or_print(cmd: List[str], *, dry_run: bool, label: str) -> int:
    print(label, flush=True)
    print("  " + shlex.join(cmd), flush=True)
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=CODE_DIR)
    return int(proc.returncode)


def main() -> int:
    args, extra_args = _parse_args()
    exp_dir = args.output_root / args.experiment_id
    manifest_path = args.manifest or (exp_dir / "manifest.json")
    index_path = args.checkpoints_index or (exp_dir / "checkpoints" / "index.json")
    results_path = exp_dir / "eval_results.jsonl"
    summary_path = exp_dir / "eval_summary.csv"
    curves_path = exp_dir / "curves.png"
    side_rounds_path = _side_rounds_path(curves_path)

    manifest = read_json(manifest_path)
    index = read_json(index_path)
    val_batches = feature_batches(manifest, "validation")
    if args.max_batches is not None:
        val_batches = val_batches[: max(0, int(args.max_batches))]
    run_batches = _combine_batches(val_batches) if args.eval_batch_mode == "all" else list(val_batches)
    checkpoints = _select_checkpoints(index, args.checkpoints, args.max_checkpoints)

    if len(checkpoints) > 1 and not args.allow_shared_logs_root:
        compact_names = " ".join(str(item.get("name", "")) for item in checkpoints)
        print("ERROR: refusing to evaluate multiple checkpoints in one logs root by default.")
        print("Reason: checkpoint eval logs must be isolated to avoid trace/agent-loop leakage across skills checkpoints.")
        print("Run this script once per checkpoint with a checkpoint-specific --logs-root and --checkpoints NAME.")
        print("Then run aggregate_self_evolution_eval.py to collect the isolated results.")
        print(f"Requested checkpoints: {compact_names}")
        print("Override with --allow-shared-logs-root only for debugging.")
        return 2

    if args.logs_root is None:
        print("WARNING: --logs-root was not provided; eval will use the default shared logs directory.")
        print("For checkpoint comparison, prefer a checkpoint-specific logs root, e.g. logs-<experiment>-eval-clean-ckpt000.")

    print(f"experiment_id={args.experiment_id}")
    print(f"experiment_dir={exp_dir}")
    print(f"manifest={manifest_path}")
    print(f"checkpoints_index={index_path}")
    print(f"eval_base_timestamp={args.eval_base_timestamp}")
    print(f"eval_timestamp_prefix={args.eval_timestamp_prefix}")
    print(f"logs_root={args.logs_root or (CODE_DIR / 'logs')}")
    print(f"eval_batch_mode={args.eval_batch_mode} run_batches={len(run_batches)}")
    print("validation_features")
    for layer in manifest["layers"]:
        print(f"  layer {layer}: {' '.join(map(str, manifest['validation_features'][str(layer)]))}")
    print("checkpoints=" + " ".join(item.get("name", "") for item in checkpoints))

    for batch_idx, batch in enumerate(run_batches, start=1):
        cmd = make_main_experiment_cmd(
            timestamp=args.eval_base_timestamp,
            batch=batch,
            feature_workers=args.feature_workers,
            skip_agent=True,
            logs_root=args.logs_root,
            force=args.force_base,
            extra_args=extra_args,
        )
        rc = _run_or_print(
            cmd,
            dry_run=args.dry_run,
            label=f"base batch {batch_idx}/{len(run_batches)} ({len(_iter_batch_features(batch))} features): {batch}",
        )
        if rc != 0:
            return rc

    all_rows: List[Dict[str, Any]] = []
    for checkpoint in checkpoints:
        name = str(checkpoint["name"])
        ckpt_path = Path(checkpoint["path"])
        if not ckpt_path.exists():
            print(f"ERROR: checkpoint path missing for {name}: {ckpt_path}")
            return 2
        eval_timestamp = f"{args.eval_timestamp_prefix}_{name}"
        before_hash = hash_tree(ckpt_path)
        print(f"checkpoint {name}: {ckpt_path} hash={before_hash}", flush=True)
        print(f"  skills_dir={ckpt_path}", flush=True)
        print(f"  logs_root={args.logs_root or (CODE_DIR / 'logs')}", flush=True)

        for batch_idx, batch in enumerate(run_batches, start=1):
            if args.dry_run:
                print(f"DRY copy base -> {eval_timestamp}: {batch}")
            else:
                for layer, fid in _iter_batch_features(batch):
                    if args.force_copy_base:
                        _reset_eval_artifacts(layer, fid, eval_timestamp, logs_root=args.logs_root)
                    copy_base_trace_for_eval(
                        layer=layer,
                        feature_id=fid,
                        base_timestamp=args.eval_base_timestamp,
                        eval_timestamp=eval_timestamp,
                        force=args.force_copy_base,
                        logs_root=args.logs_root,
                    )
            cmd = make_main_experiment_cmd(
                timestamp=eval_timestamp,
                batch=batch,
                feature_workers=args.feature_workers,
                skills_dir=ckpt_path,
                disable_skill_learning=True,
                force_agent=args.force_agent,
                logs_root=args.logs_root,
                extra_args=extra_args,
            )
            rc = _run_or_print(
                cmd,
                dry_run=args.dry_run,
                label=f"eval {name} batch {batch_idx}/{len(run_batches)} ({len(_iter_batch_features(batch))} features): {batch}",
            )
            if rc != 0:
                return rc

        if args.dry_run:
            continue

        after_hash = hash_tree(ckpt_path)
        if before_hash != after_hash:
            print(f"ERROR: checkpoint mutated during eval: {name}")
            print(f"  before={before_hash}")
            print(f"  after ={after_hash}")
            return 3

        for batch in val_batches:
            for layer, fid in batch.items():
                all_rows.append(
                    _collect_feature_result_with_side_rounds(
                        layer=layer,
                        feature_id=fid,
                        timestamp=eval_timestamp,
                        checkpoint=name,
                        checkpoint_path=ckpt_path,
                        logs_root=args.logs_root,
                    )
                )
        summary_rows = _summarize_results_with_side_rounds(all_rows)
        write_jsonl(results_path, all_rows)
        write_csv(summary_path, summary_rows)
        plot_curves(summary_rows, curves_path)
        _plot_side_rounds(summary_rows, side_rounds_path)
        print(f"wrote {results_path}")
        print(f"wrote {summary_path}")
        print(f"wrote {curves_path}")
        print(f"wrote {side_rounds_path}")

    if args.dry_run:
        return 0

    summary_rows = _summarize_results_with_side_rounds(all_rows)
    write_jsonl(results_path, all_rows)
    write_csv(summary_path, summary_rows)
    plot_curves(summary_rows, curves_path)
    _plot_side_rounds(summary_rows, side_rounds_path)
    print(f"done: rows={len(all_rows)} summary_rows={len(summary_rows)}")
    print(f"wrote {side_rounds_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
