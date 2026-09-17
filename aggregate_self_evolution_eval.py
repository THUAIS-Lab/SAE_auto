#!/usr/bin/env python3
"""Aggregate self-evolution eval results from isolated per-checkpoint logs roots."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from run_main_experiment import CODE_DIR
from self_evolution_utils import (
    DEFAULT_OUTPUT_ROOT,
    collect_feature_result,
    feature_batches,
    feature_dir,
    final_trace_path,
    read_json,
    summarize_results,
    trace_path,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate existing eval traces for checkpoints that were run under "
            "separate logs roots. This script does not run agents or call APIs."
        )
    )
    parser.add_argument("--experiment-id", default=None, help="Experiment id under analysis_output/self_evolution")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--checkpoints-index", type=Path, default=None)
    parser.add_argument("--checkpoints", nargs="*", default=None, help="Optional checkpoint names to aggregate")
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--eval-timestamp-prefix", default=None)
    parser.add_argument(
        "--logs-root-template",
        default=None,
        help=(
            "Optional template for checkpoint logs roots. Available fields: "
            "{experiment_id}, {experiment_base}, {checkpoint}, {checkpoint_compact}. "
            "Example: logs-selfev_001-eval-clean-{checkpoint_compact}"
        ),
    )
    parser.add_argument(
        "--logs-root-map",
        action="append",
        default=[],
        help="Manual mapping in the form ckpt_100=logs-selfev_001-eval-clean-ckpt100. Can be repeated.",
    )
    parser.add_argument("--output-suffix", default="all_ckpts")
    parser.add_argument("--results-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--curves-path", type=Path, default=None, help="Base path used to derive split plot names")
    parser.add_argument("--pass-rates-path", type=Path, default=None)
    parser.add_argument("--scores-path", type=Path, default=None)
    parser.add_argument("--side-rounds-path", type=Path, default=None)
    parser.add_argument("--require-complete", action="store_true", help="Return non-zero if any feature is not complete")
    parser.add_argument("--dry-run", action="store_true", help="Read logs and print the plan, but do not write outputs")
    args = parser.parse_args()
    if args.experiment_id is None:
        args.experiment_id = _latest_experiment(args.output_root)
    if args.eval_timestamp_prefix is None:
        args.eval_timestamp_prefix = f"{args.experiment_id}_eval"
    return args


def _select_checkpoints(index: Mapping[str, Any], names: Sequence[str] | None, max_count: int | None) -> List[Dict[str, Any]]:
    checkpoints = [dict(item) for item in index.get("checkpoints", [])]
    checkpoints.sort(key=lambda item: int(item.get("processed_features", 0)))
    if names:
        wanted = set(names)
        checkpoints = [item for item in checkpoints if item.get("name") in wanted]
    if max_count is not None:
        checkpoints = checkpoints[: max(0, int(max_count))]
    return checkpoints


def _parse_logs_root_map(items: Iterable[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--logs-root-map must be NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"--logs-root-map must be NAME=PATH, got: {item}")
        out[name] = Path(path)
    return out


def _under_code_dir(path: Path) -> Path:
    return path if path.is_absolute() else CODE_DIR / path


def _checkpoint_path(raw_path: str, index_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return index_path.parent / path


def _template_fields(experiment_id: str, checkpoint: str) -> Dict[str, str]:
    experiment_base = experiment_id[:-6] if experiment_id.endswith("_clean") else experiment_id
    return {
        "experiment_id": experiment_id,
        "experiment_base": experiment_base,
        "checkpoint": checkpoint,
        "checkpoint_compact": checkpoint.replace("_", ""),
    }


def _candidate_logs_roots(args: argparse.Namespace, checkpoint: str) -> List[Path]:
    fields = _template_fields(args.experiment_id, checkpoint)
    candidates: List[Path] = []

    if args.logs_root_template:
        candidates.append(Path(args.logs_root_template.format(**fields)))
        return [_under_code_dir(path) for path in candidates]

    experiment_id = fields["experiment_id"]
    experiment_base = fields["experiment_base"]
    compact = fields["checkpoint_compact"]
    candidates.extend(
        [
            Path(f"logs-{experiment_id}-eval-{compact}"),
            Path(f"logs-{experiment_id}-eval-clean-{compact}"),
            Path(f"logs-{experiment_base}-eval-clean-{compact}"),
            Path(f"logs-{experiment_base}-eval-{compact}"),
        ]
    )
    candidates.extend(CODE_DIR.glob(f"logs-*{compact}"))
    candidates.extend(CODE_DIR.glob(f"logs-*{checkpoint}"))

    resolved: List[Path] = []
    seen = set()
    for path in candidates:
        full = _under_code_dir(path)
        key = str(full.resolve()) if full.exists() else str(full)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(full)
    return resolved


def _has_eval_evidence(logs_root: Path, timestamp: str, features: Sequence[tuple[int, int]]) -> bool:
    for layer, fid in features:
        if final_trace_path(layer, fid, timestamp, logs_root=logs_root).exists():
            return True
        if trace_path(layer, fid, timestamp, logs_root=logs_root).exists():
            return True
        if (feature_dir(layer, fid, logs_root=logs_root) / timestamp / "agent_loop_summary.json").exists():
            return True
    return False


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
    logs_root: Path,
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
    logs_root: Path,
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


def _resolve_logs_root(
    *,
    args: argparse.Namespace,
    checkpoint: str,
    logs_root_map: Mapping[str, Path],
    timestamp: str,
    features: Sequence[tuple[int, int]],
) -> Path:
    if checkpoint in logs_root_map:
        root = _under_code_dir(logs_root_map[checkpoint])
        if not root.exists():
            raise FileNotFoundError(f"logs root missing for {checkpoint}: {root}")
        return root

    existing = [path for path in _candidate_logs_roots(args, checkpoint) if path.exists()]
    with_evidence = [path for path in existing if _has_eval_evidence(path, timestamp, features)]
    if len(with_evidence) == 1:
        return with_evidence[0]
    if len(with_evidence) > 1:
        raise RuntimeError(f"multiple logs roots match {checkpoint}: {' '.join(str(p) for p in with_evidence)}")
    if len(existing) == 1:
        return existing[0]
    if not existing:
        raise FileNotFoundError(
            f"could not find logs root for {checkpoint}; pass --logs-root-template or --logs-root-map"
        )
    raise RuntimeError(f"multiple logs roots exist for {checkpoint}: {' '.join(str(p) for p in existing)}")


def _validation_features(manifest: Mapping[str, Any]) -> List[tuple[int, int]]:
    features: List[tuple[int, int]] = []
    for batch in feature_batches(manifest, "validation"):
        for layer, fid in batch.items():
            features.append((int(layer), int(fid)))
    return features


def _default_output_paths(exp_dir: Path, suffix: str) -> tuple[Path, Path, Path]:
    suffix = suffix.strip("_")
    return (
        exp_dir / f"eval_results_{suffix}.jsonl",
        exp_dir / f"eval_summary_{suffix}.csv",
        exp_dir / f"curves_{suffix}.png",
    )


def _split_curve_paths(curves_path: Path) -> tuple[Path, Path]:
    suffix = curves_path.suffix or ".png"
    base = curves_path.with_suffix("")
    pass_rates_path = base.with_name(f"{base.name}_pass_rates").with_suffix(suffix)
    scores_path = base.with_name(f"{base.name}_scores").with_suffix(suffix)
    return pass_rates_path, scores_path


def _side_rounds_path(curves_path: Path) -> Path:
    suffix = curves_path.suffix or ".png"
    base = curves_path.with_suffix("")
    return base.with_name(f"{base.name}_side_rounds").with_suffix(suffix)


# Conceptual [lo, hi] bounds for each plotted metric. The y-axis is clamped to
# these so axes never show impossible regions (e.g. negative pass rates) while
# still trimming the empty band a fixed [0, 1] range leaves when all the data
# sits near the top of the scale.
_SERIES_BOUNDS: Dict[str, tuple[float, float]] = {
    "all_gates_pass_rate": (0.0, 1.0),
    "gate1_pass_rate": (0.0, 1.0),
    "gate2_pass_rate": (0.0, 1.0),
    "gate3_pass_rate": (0.0, 1.0),
    "mean_input_activation_rate": (0.0, 1.0),
    "mean_input_boundary_non_activation_rate": (0.0, 1.0),
    "mean_output_score": (-1.0, 1.0),
    "mean_chain_judge_score": (0.0, 5.0),
}


def _adaptive_ylim(
    values: Sequence[float], bound_lo: float, bound_hi: float
) -> tuple[float, float]:
    """Pick y-axis limits that hug the data, clamped to the series' bounds.

    Adds ~10% padding (at least 0.03) around the data range, then clamps into
    [bound_lo, bound_hi] with a sliver of headroom above the cap so points at
    the maximum don't sit flat on the top border.
    """
    if not values:
        return bound_lo, bound_hi
    vmin = min(values)
    vmax = max(values)
    span = vmax - vmin
    pad = max(span * 0.1, 0.03)
    headroom = 0.02 * (bound_hi - bound_lo)
    lo = max(bound_lo, vmin - pad)
    hi = min(bound_hi + headroom, vmax + pad)
    if hi <= lo:
        # Degenerate (constant series, single point, or data outside bounds):
        # fall back to the full conceptual range so the line stays visible.
        lo, hi = bound_lo, bound_hi
    return lo, hi


def _plot_series(
    summary_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    series: Sequence[tuple[str, str]],
) -> None:
    if not summary_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(summary_rows, key=lambda row: int(row["checkpoint_train_features"]))
    x = [int(row["checkpoint_train_features"]) for row in rows]
    y_per_series = [[float(row.get(key, 0.0)) for row in rows] for key, _ in series]

    # Series that share the same conceptual bounds share one y-scale too, so
    # comparable subplots (e.g. the four gate pass-rates) stay comparable while
    # still trimming the empty lower band a fixed [0, 1] range leaves behind.
    shared_ylim: Dict[tuple[float, float], tuple[float, float]] = {}
    for (key, _), ys in zip(series, y_per_series):
        bound = _SERIES_BOUNDS.get(key)
        if bound is None:
            continue
        lo, hi = _adaptive_ylim(ys, *bound)
        if bound in shared_ylim:
            prev_lo, prev_hi = shared_ylim[bound]
            shared_ylim[bound] = (min(prev_lo, lo), max(prev_hi, hi))
        else:
            shared_ylim[bound] = (lo, hi)

    fig, axes = plt.subplots(len(series), 1, figsize=(8, max(4, 2.4 * len(series))), sharex=True)
    if len(series) == 1:
        axes = [axes]
    for ax, (key, title), ys in zip(axes, series, y_per_series):
        ax.plot(x, ys, marker="o", linewidth=2)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        bound = _SERIES_BOUNDS.get(key)
        if bound is not None:
            ax.set_ylim(*shared_ylim[bound])
    axes[-1].set_xlabel("Training features accumulated in skills checkpoint")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_split_curves(summary_rows: Sequence[Mapping[str, Any]], pass_rates_path: Path, scores_path: Path) -> None:
    pass_rate_series = [
        ("all_gates_pass_rate", "All Gates Pass Rate"),
        ("gate1_pass_rate", "Gate 1 Pass Rate"),
        ("gate2_pass_rate", "Gate 2 Pass Rate"),
        ("gate3_pass_rate", "Gate 3 Pass Rate"),
    ]
    score_series = [
        ("mean_chain_judge_score", "Gate 3: Mean Chain Score"),
        ("mean_output_score", "Gate 2: Mean Output Score"),
        ("mean_input_activation_rate", "Gate 1: Mean Input Activation"),
        ("mean_input_boundary_non_activation_rate", "Gate 1: Mean Input Boundary"),
    ]
    _plot_series(summary_rows, pass_rates_path, pass_rate_series)
    _plot_series(summary_rows, scores_path, score_series)


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


def main() -> int:
    args = _parse_args()
    exp_dir = args.output_root / args.experiment_id
    manifest_path = args.manifest or (exp_dir / "manifest.json")
    index_path = args.checkpoints_index or (exp_dir / "checkpoints" / "index.json")
    default_results_path, default_summary_path, default_curves_path = _default_output_paths(
        exp_dir, args.output_suffix
    )
    results_path = args.results_path or default_results_path
    summary_path = args.summary_path or default_summary_path
    curves_path = args.curves_path or default_curves_path
    default_pass_rates_path, default_scores_path = _split_curve_paths(curves_path)
    pass_rates_path = args.pass_rates_path or default_pass_rates_path
    scores_path = args.scores_path or default_scores_path
    side_rounds_path = args.side_rounds_path or _side_rounds_path(curves_path)

    manifest = read_json(manifest_path)
    index = read_json(index_path)
    checkpoints = _select_checkpoints(index, args.checkpoints, args.max_checkpoints)
    features = _validation_features(manifest)
    logs_root_map = _parse_logs_root_map(args.logs_root_map)

    print(f"experiment_id={args.experiment_id}")
    print(f"experiment_dir={exp_dir}")
    print(f"manifest={manifest_path}")
    print(f"checkpoints_index={index_path}")
    print(f"eval_timestamp_prefix={args.eval_timestamp_prefix}")
    print(f"validation_features={len(features)}")
    print("checkpoints=" + " ".join(str(item.get("name", "")) for item in checkpoints))

    all_rows: List[Dict[str, Any]] = []
    incomplete: List[Dict[str, Any]] = []

    for checkpoint in checkpoints:
        name = str(checkpoint["name"])
        timestamp = f"{args.eval_timestamp_prefix}_{name}"
        logs_root = _resolve_logs_root(
            args=args,
            checkpoint=name,
            logs_root_map=logs_root_map,
            timestamp=timestamp,
            features=features,
        )
        ckpt_path = _checkpoint_path(str(checkpoint.get("path", "")), index_path)

        rows: List[Dict[str, Any]] = []
        for layer, fid in features:
            row = _collect_feature_result_with_side_rounds(
                layer=layer,
                feature_id=fid,
                timestamp=timestamp,
                checkpoint=name,
                checkpoint_path=ckpt_path,
                logs_root=logs_root,
            )
            row["logs_root"] = str(logs_root)
            rows.append(row)
            all_rows.append(row)
            if row["status"] != "complete":
                incomplete.append(row)

        status_counts = Counter(str(row["status"]) for row in rows)
        counts_text = ", ".join(f"{status}={count}" for status, count in sorted(status_counts.items()))
        print(f"{name}: logs_root={logs_root}  {counts_text}")

    summary_rows = _summarize_results_with_side_rounds(all_rows)
    print(f"rows={len(all_rows)} summary_rows={len(summary_rows)} incomplete={len(incomplete)}")

    if incomplete:
        print("incomplete_features")
        for row in incomplete:
            print(
                f"  {row['checkpoint']} L{row['layer_id']}-F{row['feature_id']} "
                f"status={row['status']} trace={row['trace_path']}"
            )

    print(f"results_path={results_path}")
    print(f"summary_path={summary_path}")
    print(f"pass_rates_path={pass_rates_path}")
    print(f"scores_path={scores_path}")
    print(f"side_rounds_path={side_rounds_path}")

    if args.require_complete and incomplete:
        print("ERROR: incomplete features found; not writing outputs because --require-complete was set")
        return 4

    if args.dry_run:
        print("dry-run: no files written")
        return 0

    write_jsonl(results_path, all_rows)
    write_csv(summary_path, summary_rows)
    _plot_split_curves(summary_rows, pass_rates_path, scores_path)
    _plot_side_rounds(summary_rows, side_rounds_path)
    print("wrote outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
