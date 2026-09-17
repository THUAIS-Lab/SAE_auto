"""Shared skill promotion, compaction, and threshold-based deduplication."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List

from skill_lock import exclusive_skill_lock


CASE_DEDUP_THRESHOLDS = {
    "input": 210_000,
    "output": 170_000,
    "chain": 130_000,
}
GENERAL_SKILL_DEDUP_THRESHOLD_CHARS = 20_000
SKILL_DEDUP_TARGET_RATIO = 0.85
SKILL_SIDES = ("input", "output", "chain")


def _hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    base = Path(root)
    if not base.exists():
        return ""
    for path in sorted(p for p in base.rglob("*") if p.is_file() and p.name != ".maintenance.lock"):
        digest.update(path.relative_to(base).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_chars(path: Path) -> int:
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8", errors="replace"))


def oversized_skill_files(skills_dir: Path) -> List[Dict[str, Any]]:
    """Return independently thresholded skill files that currently exceed limits."""
    root = Path(skills_dir)
    oversized: List[Dict[str, Any]] = []
    for side in SKILL_SIDES:
        for kind, filename, threshold in (
            ("cases", f"skill_{side}_cases.md", CASE_DEDUP_THRESHOLDS[side]),
            ("general", f"skill_{side}.md", GENERAL_SKILL_DEDUP_THRESHOLD_CHARS),
        ):
            chars = _file_chars(root / filename)
            if chars > threshold:
                oversized.append(
                    {
                        "side": side,
                        "kind": kind,
                        "file": filename,
                        "chars": chars,
                        "threshold_chars": threshold,
                    }
                )
    return oversized


def _perform_skill_maintenance(
    client: Any,
    model: str,
    *,
    skills_dir: Path,
    archive_dir: Path,
    maintenance_name: str,
    verbose: bool,
) -> Dict[str, Any]:
    from skill_compact import compact_skill_files, deduplicate_skill_file, promote_effective_cases

    pre_hash = _hash_tree(skills_dir)
    if verbose:
        print(f"skill maintenance {maintenance_name}: promoting effective cases", flush=True)
    promoted = promote_effective_cases(client, model, skills_dir=skills_dir, verbose=verbose)
    if verbose:
        print(f"skill maintenance {maintenance_name}: compacting cases", flush=True)
    compacted = compact_skill_files(
        client,
        model,
        skills_dir=skills_dir,
        archive_dir=archive_dir,
        verbose=verbose,
    )
    deduplication: List[Dict[str, Any]] = []
    for side in SKILL_SIDES:
        deduplication.append(
            deduplicate_skill_file(
                client,
                model,
                skills_dir=skills_dir,
                side=side,
                kind="cases",
                threshold_chars=CASE_DEDUP_THRESHOLDS[side],
                target_ratio=SKILL_DEDUP_TARGET_RATIO,
                archive_dir=archive_dir,
                checkpoint_name=maintenance_name,
                verbose=verbose,
            )
        )
        deduplication.append(
            deduplicate_skill_file(
                client,
                model,
                skills_dir=skills_dir,
                side=side,
                kind="general",
                threshold_chars=GENERAL_SKILL_DEDUP_THRESHOLD_CHARS,
                target_ratio=SKILL_DEDUP_TARGET_RATIO,
                archive_dir=archive_dir,
                checkpoint_name=maintenance_name,
                verbose=verbose,
            )
        )
    return {
        "status": "ok",
        "maintenance_name": maintenance_name,
        "pre_hash": pre_hash,
        "post_hash": _hash_tree(skills_dir),
        "archive_dir": str(archive_dir),
        "deduplication_config": {
            "case_threshold_chars": CASE_DEDUP_THRESHOLDS,
            "general_threshold_chars": GENERAL_SKILL_DEDUP_THRESHOLD_CHARS,
            "target_ratio": SKILL_DEDUP_TARGET_RATIO,
        },
        "promoted_cases": promoted,
        "compacted_files": compacted,
        "deduplication": deduplication,
    }


def run_skill_maintenance(
    client: Any,
    model: str,
    *,
    skills_dir: Path,
    archive_dir: Path,
    maintenance_name: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full maintenance sequence under the global exclusive lock."""
    with exclusive_skill_lock(skills_dir, blocking=True) as acquired:
        if not acquired:  # blocking acquisition should never return False
            raise RuntimeError("failed to acquire the skills maintenance lock")
        return _perform_skill_maintenance(
            client,
            model,
            skills_dir=Path(skills_dir),
            archive_dir=Path(archive_dir),
            maintenance_name=maintenance_name,
            verbose=verbose,
        )


def maybe_run_auto_skill_maintenance(
    client: Any,
    model: str,
    *,
    skills_dir: Path,
    archive_dir: Path,
    maintenance_name: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Check after a write and serialize maintenance only when a file looks oversized."""
    try:
        initial_oversized = oversized_skill_files(skills_dir)
        if not initial_oversized:
            return {
                "status": "not_needed",
                "maintenance_name": maintenance_name,
                "oversized_files": [],
            }
        with exclusive_skill_lock(skills_dir, blocking=True) as acquired:
            if not acquired:  # blocking acquisition should never return False
                raise RuntimeError("failed to acquire the skills maintenance lock")
            pre_hash = _hash_tree(skills_dir)
            oversized = oversized_skill_files(skills_dir)
            if not oversized:
                return {
                    "status": "not_needed",
                    "maintenance_name": maintenance_name,
                    "pre_hash": pre_hash,
                    "post_hash": pre_hash,
                    "oversized_files": [],
                    "initial_oversized_files": initial_oversized,
                }
            if verbose:
                labels = ", ".join(f"{item['file']}={item['chars']}" for item in oversized)
                print(f"skill maintenance {maintenance_name}: threshold exceeded ({labels})", flush=True)
            result = _perform_skill_maintenance(
                client,
                model,
                skills_dir=Path(skills_dir),
                archive_dir=Path(archive_dir),
                maintenance_name=maintenance_name,
                verbose=verbose,
            )
            result["triggered_by"] = "post_write_threshold"
            result["oversized_files"] = oversized
            result["initial_oversized_files"] = initial_oversized
            return result
    except Exception as exc:
        return {
            "status": "failed",
            "maintenance_name": maintenance_name,
            "error": str(exc),
        }
