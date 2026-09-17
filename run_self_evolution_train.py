#!/usr/bin/env python3
"""Run the training/self-evolution phase and checkpoint skills over time."""
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

from run_main_experiment import CODE_DIR, LAYER_IDS
from self_evolution_utils import (
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TRAIN_PER_LAYER,
    DEFAULT_VAL_PER_LAYER,
    build_manifest,
    checkpoint_names,
    copy_skills_checkpoint,
    feature_batches,
    hash_tree,
    make_main_experiment_cmd,
    timestamp_id,
    write_json,
    read_json,
)
from skill_maintenance import run_skill_maintenance

FEATURE_WORKER_BATCH_SIZE = 5


def _parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Run the self-evolution training phase and save skills checkpoints."
    )
    parser.add_argument("--experiment-id", default=None, help="Experiment id under analysis_output/self_evolution")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--logs-root", type=Path, default=None)
    parser.add_argument("--skills-dir", type=Path, default=CODE_DIR / "skills")
    parser.add_argument("--timestamp", default=None, help="Timestamp used for all train feature runs")
    parser.add_argument("--layers", nargs="*", type=int, default=LAYER_IDS)
    parser.add_argument("--train-per-layer", type=int, default=DEFAULT_TRAIN_PER_LAYER)
    parser.add_argument("--val-per-layer", type=int, default=DEFAULT_VAL_PER_LAYER)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument(
        "--checkpoint-offset",
        type=int,
        default=0,
        help="Previously processed feature count used in checkpoint names (for example 120)",
    )
    parser.add_argument("--feature-workers", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=None, help="Optional smoke-test limit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run completed batches and overwrite checkpoints")
    parser.add_argument(
        "--skip-skill-compact",
        action="store_true",
        help="Skip checkpoint-time skill promotion, compaction, and deduplication",
    )
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    blocked_main_flags = (
        "--skill-maintenance",
        "--skill-maintenance-archive-dir",
        "--save-skills-checkpoint",
    )
    blocked = next(
        (
            item
            for item in extra
            if any(item == flag or item.startswith(flag + "=") for flag in blocked_main_flags)
        ),
        None,
    )
    if blocked is not None:
        parser.error(
            f"{blocked} cannot be passed through self-evolution; "
            "run_main skill maintenance is forced off to avoid duplicate checkpoint maintenance"
        )
    if args.experiment_id is None:
        args.experiment_id = timestamp_id()
    if args.timestamp is None:
        args.timestamp = f"{args.experiment_id}_train"
    if args.logs_root is None:
        args.logs_root = CODE_DIR / f"logs-{args.experiment_id}-train"
    if args.checkpoint_offset < 0:
        parser.error("--checkpoint-offset must be non-negative")
    return args, extra


def _load_or_create_manifest(args: argparse.Namespace, manifest_path: Path) -> Dict[str, Any]:
    if args.manifest is not None:
        return read_json(args.manifest)
    if manifest_path.exists():
        return read_json(manifest_path)
    return build_manifest(
        layers=args.layers,
        train_per_layer=args.train_per_layer,
        val_per_layer=args.val_per_layer,
    )


def _upsert_checkpoint(index: Dict[str, Any], checkpoint: Dict[str, Any]) -> None:
    checkpoints = index.setdefault("checkpoints", [])
    checkpoints[:] = [item for item in checkpoints if item.get("name") != checkpoint.get("name")]
    checkpoints.append(checkpoint)
    checkpoints.sort(key=lambda item: int(item.get("processed_features", 0)))


def _run_skill_maintenance(args: argparse.Namespace, *, checkpoint_name: str) -> Dict[str, Any]:
    pre_hash = hash_tree(args.skills_dir)
    try:
        from openai import OpenAI

        from function import read_api_key
        from support_info.llm_api_info import (
            api_key_file as DEFAULT_API_KEY_FILE,
            base_url as DEFAULT_BASE_URL,
            model_name as DEFAULT_MODEL_NAME,
        )

        api_key = read_api_key(DEFAULT_API_KEY_FILE)
        client = OpenAI(base_url=DEFAULT_BASE_URL, api_key=api_key)
        archive_dir = args.output_root / args.experiment_id / "skill_case_archive"
        result = run_skill_maintenance(
            client,
            DEFAULT_MODEL_NAME,
            skills_dir=args.skills_dir,
            archive_dir=archive_dir,
            maintenance_name=checkpoint_name,
            verbose=True,
        )
        result["checkpoint"] = checkpoint_name
        return result
    except Exception as exc:
        return {
            "status": "failed",
            "checkpoint": checkpoint_name,
            "pre_hash": pre_hash,
            "post_hash": hash_tree(args.skills_dir),
            "error": str(exc),
        }


def _batches_per_launch(feature_workers: int) -> int:
    return max(1, int(feature_workers) // FEATURE_WORKER_BATCH_SIZE)


def _batch_groups(
    train_batches: List[Dict[int, int]],
    *,
    batches_per_launch: int,
    features_per_batch: int,
    checkpoint_every: int,
) -> List[List[Tuple[int, Dict[int, int]]]]:
    groups: List[List[Tuple[int, Dict[int, int]]]] = []
    idx = 0
    checkpoint_batch_span = 0
    if features_per_batch > 0 and checkpoint_every > 0 and checkpoint_every % features_per_batch == 0:
        checkpoint_batch_span = int(checkpoint_every) // int(features_per_batch)
    while idx < len(train_batches):
        take = max(1, int(batches_per_launch))
        if checkpoint_batch_span > 0:
            take = min(take, checkpoint_batch_span - (idx % checkpoint_batch_span))
        end = min(len(train_batches), idx + take)
        groups.append([(batch_idx + 1, train_batches[batch_idx]) for batch_idx in range(idx, end)])
        idx = end
    return groups


def _merge_batch_group(group: List[Tuple[int, Dict[int, int]]]) -> Dict[int, List[int]]:
    merged: Dict[int, List[int]] = {}
    for _, batch in group:
        for layer, feature_id in batch.items():
            merged.setdefault(int(layer), []).append(int(feature_id))
    return merged


def _group_feature_count(group: List[Tuple[int, Dict[int, int]]]) -> int:
    return sum(len(batch) for _, batch in group)


def _group_indices(group: List[Tuple[int, Dict[int, int]]]) -> List[int]:
    return [idx for idx, _ in group]


def _completed_features_in_scope(completed: set[int], total_batches: int, features_per_batch: int) -> int:
    scoped = [idx for idx in completed if 1 <= int(idx) <= int(total_batches)]
    return len(scoped) * int(features_per_batch)


def _snapshot(
    *,
    args: argparse.Namespace,
    index: Dict[str, Any],
    index_path: Path,
    name: str,
    processed_features: int,
) -> None:
    dest = index_path.parent / name
    existing = dest.exists() and not args.force
    maintenance: Dict[str, Any] = {"status": "skipped_existing_checkpoint" if existing else "skipped_by_flag"}
    if (not existing) and not args.skip_skill_compact:
        maintenance = _run_skill_maintenance(args, checkpoint_name=name)
    if existing:
        prior = next((item for item in index.get("checkpoints", []) if item.get("name") == name), None)
        meta = prior if prior is not None else {
            "name": name,
            "processed_features": int(processed_features),
            "path": str(dest),
            "hash": hash_tree(dest),
            "created_at": "existing",
            "skill_maintenance": maintenance,
        }
    else:
        copied = copy_skills_checkpoint(args.skills_dir, dest, overwrite=args.force)
        meta = {
            "name": name,
            "processed_features": int(processed_features),
            **copied,
            "skill_maintenance": maintenance,
        }
    _upsert_checkpoint(index, meta)
    write_json(index_path, index)
    print(f"checkpoint {name}: {meta['hash']} -> {dest}", flush=True)


def main() -> int:
    args, extra_args = _parse_args()
    exp_dir = args.output_root / args.experiment_id
    manifest_path = exp_dir / "manifest.json"
    progress_path = exp_dir / "train_progress.json"
    index_path = exp_dir / "checkpoints" / "index.json"

    manifest = _load_or_create_manifest(args, manifest_path)
    train_batches = feature_batches(manifest, "train")
    if args.max_batches is not None:
        train_batches = train_batches[: max(0, int(args.max_batches))]

    features_per_batch = len(manifest["layers"])
    checkpoint_offset = int(args.checkpoint_offset)
    batches_per_launch = _batches_per_launch(args.feature_workers)
    batch_groups = _batch_groups(
        train_batches,
        batches_per_launch=batches_per_launch,
        features_per_batch=features_per_batch,
        checkpoint_every=int(args.checkpoint_every),
    )

    total_features = int(manifest["train_per_layer"]) * len(manifest["layers"])
    planned_checkpoints = checkpoint_names(
        total_features,
        args.checkpoint_every,
        offset=checkpoint_offset,
    )

    print(f"experiment_id={args.experiment_id}")
    print(f"experiment_dir={exp_dir}")
    print(f"train_timestamp={args.timestamp}")
    print(f"logs_root={args.logs_root}")
    print(f"skills_dir={args.skills_dir}")
    print(f"feature_workers={args.feature_workers}")
    print(f"checkpoint_offset={checkpoint_offset}")
    print(
        f"batches_per_launch={batches_per_launch} "
        f"(feature_workers // {FEATURE_WORKER_BATCH_SIZE}; remainder ignored)"
    )
    print(f"planned_checkpoints={planned_checkpoints}")

    for split_key in ("train_features", "validation_features"):
        print(split_key)
        for layer in manifest["layers"]:
            print(f"  layer {layer}: {' '.join(map(str, manifest[split_key][str(layer)]))}")

    if args.dry_run:
        for group_idx, group in enumerate(batch_groups, start=1):
            merged_batch = _merge_batch_group(group)
            cmd = make_main_experiment_cmd(
                timestamp=args.timestamp,
                batch=merged_batch,
                feature_workers=args.feature_workers,
                logs_root=args.logs_root,
                skills_dir=args.skills_dir,
                skill_maintenance_mode="off",
                force=args.force,
                force_agent=args.force,
                extra_args=extra_args,
            )
            if len(group) < batches_per_launch:
                label = "partial final" if group_idx == len(batch_groups) else "checkpoint boundary"
                print(
                    f"DRY {label} batch group {group_idx}/{len(batch_groups)}: "
                    f"requested {batches_per_launch} batch(es), got {len(group)}; "
                    f"running {_group_feature_count(group)} feature(s)"
                )
            print(f"DRY batch group {group_idx}/{len(batch_groups)} batches={_group_indices(group)}: {merged_batch}")
            print("  " + shlex.join(cmd))
        return 0

    exp_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)

    progress: Dict[str, Any] = read_json(progress_path) if progress_path.exists() else {
        "experiment_id": args.experiment_id,
        "timestamp": args.timestamp,
        "logs_root": str(args.logs_root),
        "completed_batches": [],
    }
    progress["logs_root"] = str(args.logs_root)
    completed = set() if args.force else set(int(x) for x in progress.get("completed_batches", []))
    index: Dict[str, Any] = read_json(index_path) if index_path.exists() else {
        "experiment_id": args.experiment_id,
        "source_skills_dir": str(args.skills_dir),
        "checkpoints": [],
    }

    _snapshot(
        args=args,
        index=index,
        index_path=index_path,
        name=f"ckpt_{checkpoint_offset:03d}",
        processed_features=checkpoint_offset,
    )

    for group_idx, group in enumerate(batch_groups, start=1):
        pending_group = [
            (batch_idx, batch)
            for batch_idx, batch in group
            if not (args.resume and not args.force and batch_idx in completed)
        ]
        if not pending_group:
            print(f"skip completed batch group {group_idx}/{len(batch_groups)} batches={_group_indices(group)}", flush=True)
        else:
            if len(group) < batches_per_launch:
                label = "partial final" if group_idx == len(batch_groups) else "checkpoint boundary"
                print(
                    f"{label} batch group {group_idx}/{len(batch_groups)}: "
                    f"requested {batches_per_launch} batch(es), got {len(group)}; "
                    f"running {_group_feature_count(pending_group)} pending feature(s)",
                    flush=True,
                )
            elif len(pending_group) < len(group):
                print(
                    f"resume partial batch group {group_idx}/{len(batch_groups)}: "
                    f"running pending batches={_group_indices(pending_group)}",
                    flush=True,
                )
            merged_batch = _merge_batch_group(pending_group)
            cmd = make_main_experiment_cmd(
                timestamp=args.timestamp,
                batch=merged_batch,
                feature_workers=args.feature_workers,
                logs_root=args.logs_root,
                skills_dir=args.skills_dir,
                skill_maintenance_mode="off",
                force=args.force,
                force_agent=args.force,
                extra_args=extra_args,
            )
            print(
                f"run batch group {group_idx}/{len(batch_groups)} "
                f"batches={_group_indices(pending_group)}: {merged_batch}",
                flush=True,
            )
            print("  " + shlex.join(cmd), flush=True)
            proc = subprocess.run(cmd, cwd=CODE_DIR)
            if proc.returncode != 0:
                progress["last_failed_batch"] = _group_indices(pending_group)[0]
                progress["last_failed_batch_group"] = _group_indices(pending_group)
                write_json(progress_path, progress)
                return int(proc.returncode)
            for batch_idx, _ in pending_group:
                completed.add(batch_idx)
            progress["completed_batches"] = sorted(completed)
            progress["last_completed_batch"] = max(_group_indices(pending_group))
            write_json(progress_path, progress)

        processed = _completed_features_in_scope(completed, len(train_batches), features_per_batch)
        if processed > 0 and processed % int(args.checkpoint_every) == 0:
            total_processed = checkpoint_offset + processed
            _snapshot(
                args=args,
                index=index,
                index_path=index_path,
                name=f"ckpt_{total_processed:03d}",
                processed_features=total_processed,
            )

    final_new_features = len(train_batches) * len(manifest["layers"])
    if final_new_features > 0 and final_new_features % int(args.checkpoint_every) != 0:
        final_processed = checkpoint_offset + final_new_features
        _snapshot(
            args=args,
            index=index,
            index_path=index_path,
            name=f"ckpt_{final_processed:03d}",
            processed_features=final_processed,
        )

    progress["status"] = "complete" if len(train_batches) == int(manifest["train_per_layer"]) else "partial"
    completed_new_features = _completed_features_in_scope(completed, len(train_batches), features_per_batch)
    progress["checkpoint_offset"] = checkpoint_offset
    progress["completed_new_features"] = completed_new_features
    progress["completed_features"] = checkpoint_offset + completed_new_features
    write_json(progress_path, progress)
    print(f"done: {progress['status']} completed_features={progress['completed_features']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
