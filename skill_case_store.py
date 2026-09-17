"""Structured helpers for skill case markdown files.

The markdown case files remain the source of truth.  This module adds
machine-readable metadata blocks for auditing.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from function import append_text_locked
from skill_lock import exclusive_skill_lock, shared_skill_lock

CASE_FILES = {
    "input": "skill_input_cases.md",
    "output": "skill_output_cases.md",
    "chain": "skill_chain_cases.md",
}
CASE_SIDES = tuple(CASE_FILES)

_HEADING_RE = re.compile(r"(?m)^## .*$")
_META_RE = re.compile(r"<!--\s*skill_case_meta\s*(\{.*?\})\s*-->", re.DOTALL)
_STATS_TABLE_RE = re.compile(
    r"^\s*\|(?:\s*read_count\s*\|)?\s*use_count\s*\|\s*effective_use_count\s*\|\s*compressed\s*\|\n"
    r"\|[-:| ]+\|\n"
    r"\|[^\n]*\|\n*",
    re.MULTILINE,
)


@dataclass
class SkillCase:
    case_id: str
    side: str
    path: Path
    heading: str
    body: str
    section: str
    meta: Dict[str, Any]
    start: int
    end: int

    @property
    def text(self) -> str:
        return f"{self.heading}\n{self.body}".strip()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _slug(text: str, max_len: int = 64) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return (text or "case")[:max_len].strip("-") or "case"


def _legacy_case_id(side: str, heading: str) -> str:
    digest = hashlib.sha1(heading.encode("utf-8", errors="replace")).hexdigest()[:8]
    clean = re.sub(r"^##\s*", "", heading).replace("[Auto]", "")
    return f"{side}-{_slug(clean)}-{digest}"


def make_case_id(
    *,
    side: str,
    layer_id: str,
    feature_id: str,
    round_idx: int,
    trigger: str,
    harness: Optional[Dict[str, Any]] = None,
) -> str:
    payload = json.dumps(harness or {}, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"{side}-L{layer_id}-F{feature_id}-r{int(round_idx)}-{_slug(trigger, 24)}-{digest}"


def _extract_meta(body: str) -> Dict[str, Any]:
    match = _META_RE.search(body)
    if not match:
        return {}
    try:
        meta = json.loads(match.group(1))
        return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


def _normalize_meta(
    meta: Dict[str, Any],
    *,
    side: str,
    heading: str,
    path: Path,
) -> Dict[str, Any]:
    out = dict(meta or {})
    out["case_id"] = str(out.get("case_id") or _legacy_case_id(side, heading))
    out["side"] = str(out.get("side") or side)
    out["case_file"] = str(out.get("case_file") or path.name)
    out["heading"] = str(out.get("heading") or heading)
    usage = dict(out.get("usage") or {})
    usage.pop("read_count", None)
    usage["use_count"] = _safe_int(usage.get("use_count"))
    usage["effective_use_count"] = _safe_int(usage.get("effective_use_count"))
    usage["compressed"] = bool(usage.get("compressed", False))
    usage.setdefault("compressed_at", None)
    usage.setdefault("compressed_result", None)
    out["usage"] = usage
    out.setdefault("created_at", _now())
    return out


def _parse_cases_from_text(path: Path, side: str, text: str) -> List[SkillCase]:
    matches = list(_HEADING_RE.finditer(text))
    cases: List[SkillCase] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if not section:
            continue
        first_newline = section.find("\n")
        if first_newline == -1:
            heading = section.strip()
            body = ""
        else:
            heading = section[:first_newline].strip()
            body = section[first_newline + 1 :].strip()
        meta = _normalize_meta(_extract_meta(body), side=side, heading=heading, path=path)
        cases.append(
            SkillCase(
                case_id=str(meta["case_id"]),
                side=side,
                path=path,
                heading=heading,
                body=body,
                section=section,
                meta=meta,
                start=start,
                end=end,
            )
        )
    return cases


def iter_skill_cases(skills_dir: Path, side: Optional[str] = None) -> List[SkillCase]:
    sides = [side] if side else list(CASE_FILES)
    cases: List[SkillCase] = []
    with shared_skill_lock(skills_dir):
        for current_side in sides:
            if current_side not in CASE_FILES:
                continue
            path = Path(skills_dir) / CASE_FILES[current_side]
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            cases.extend(_parse_cases_from_text(path, current_side, text))
    return cases


def find_skill_case(skills_dir: Path, case_id: str) -> Optional[SkillCase]:
    wanted = str(case_id)
    for case in iter_skill_cases(skills_dir):
        if case.case_id == wanted:
            return case
    return None


def search_skill_cases(
    skills_dir: Path,
    *,
    side: Optional[str] = None,
    query: str = "",
    limit: int = 12,
) -> List[Dict[str, Any]]:
    terms = [term.lower() for term in re.split(r"\s+", str(query).strip()) if term]
    out: List[Dict[str, Any]] = []
    for case in iter_skill_cases(skills_dir, side=side):
        haystack = f"{case.heading}\n{case.body}".lower()
        if terms and not all(term in haystack for term in terms):
            continue
        usage = case.meta.get("usage") or {}
        snippet = _META_RE.sub("", case.body)
        snippet = _STATS_TABLE_RE.sub("", snippet).strip()
        snippet = re.sub(r"\s+", " ", snippet)[:360]
        out.append(
            {
                "case_id": case.case_id,
                "side": case.side,
                "heading": case.heading,
                "use_count": _safe_int(usage.get("use_count")),
                "effective_use_count": _safe_int(usage.get("effective_use_count")),
                "compressed": bool(usage.get("compressed", False)),
                "snippet": snippet,
            }
        )
        if len(out) >= int(limit):
            break
    return out


def _meta_comment(meta: Dict[str, Any]) -> str:
    return "<!-- skill_case_meta\n" + json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n-->"


def _stats_table(meta: Dict[str, Any]) -> str:
    usage = meta.get("usage") or {}
    compressed = "yes" if usage.get("compressed") else "no"
    return (
        "| use_count | effective_use_count | compressed |\n"
        "|---:|---:|:---|\n"
        f"| {_safe_int(usage.get('use_count'))} | "
        f"{_safe_int(usage.get('effective_use_count'))} | {compressed} |"
    )


def _clean_body(body: str) -> str:
    cleaned = _META_RE.sub("", body, count=1).strip()
    cleaned = _STATS_TABLE_RE.sub("", cleaned, count=1).strip()
    return cleaned


def render_case_section(heading: str, body: str, meta: Dict[str, Any]) -> str:
    return (
        f"{heading.strip()}\n\n"
        f"{_meta_comment(meta)}\n\n"
        f"{_stats_table(meta)}\n\n"
        f"{_clean_body(body).rstrip()}\n"
    )


def _write_text_locked(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            f.truncate()
            f.write(text)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _update_case(
    skills_dir: Path,
    case_id: str,
    mutator: Callable[[SkillCase], SkillCase],
) -> Optional[SkillCase]:
    wanted = str(case_id)
    with shared_skill_lock(skills_dir):
        for side, filename in CASE_FILES.items():
            path = Path(skills_dir) / filename
            if not path.exists():
                continue
            with path.open("a+", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.seek(0)
                    text = f.read()
                    cases = _parse_cases_from_text(path, side, text)
                    matched = next((case for case in cases if case.case_id == wanted), None)
                    if matched is None:
                        continue
                    updated = mutator(matched)
                    rendered = render_case_section(updated.heading, updated.body, updated.meta).strip()
                    new_text = text[: matched.start].rstrip() + "\n\n" + rendered + "\n\n" + text[matched.end :].lstrip()
                    f.seek(0)
                    f.truncate()
                    f.write(new_text.rstrip() + "\n")
                    f.flush()
                    return updated
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
    return None


def increment_case_usage(
    skills_dir: Path,
    case_id: str,
    *,
    use_delta: int = 0,
    effective_delta: int = 0,
    post_write: Optional[Callable[[], Any]] = None,
) -> Optional[SkillCase]:
    def _mutate(case: SkillCase) -> SkillCase:
        meta = _normalize_meta(case.meta, side=case.side, heading=case.heading, path=case.path)
        usage = meta.setdefault("usage", {})
        usage["use_count"] = _safe_int(usage.get("use_count")) + int(use_delta)
        usage["effective_use_count"] = _safe_int(usage.get("effective_use_count")) + int(effective_delta)
        meta["updated_at"] = _now()
        return SkillCase(
            case_id=case.case_id,
            side=case.side,
            path=case.path,
            heading=case.heading,
            body=case.body,
            section=case.section,
            meta=meta,
            start=case.start,
            end=case.end,
        )

    updated = _update_case(skills_dir, case_id, _mutate)
    if updated is not None and post_write is not None:
        post_write()
    return updated


def mark_case_compressed(
    skills_dir: Path,
    case_id: str,
    *,
    result: str,
    target: Optional[str] = None,
) -> Optional[SkillCase]:
    def _mutate(case: SkillCase) -> SkillCase:
        meta = _normalize_meta(case.meta, side=case.side, heading=case.heading, path=case.path)
        usage = meta.setdefault("usage", {})
        usage["compressed"] = True
        usage["compressed_at"] = _now()
        usage["compressed_result"] = result
        usage["compressed_target"] = target
        meta["updated_at"] = _now()
        return SkillCase(
            case_id=case.case_id,
            side=case.side,
            path=case.path,
            heading=case.heading,
            body=case.body,
            section=case.section,
            meta=meta,
            start=case.start,
            end=case.end,
        )

    return _update_case(skills_dir, case_id, _mutate)


def read_skill_case(
    skills_dir: Path,
    case_id: str,
) -> Optional[str]:
    case = find_skill_case(skills_dir, case_id)
    if case is None:
        return None
    return case.text


def append_skill_case(
    skills_dir: Path,
    *,
    side: str,
    heading: str,
    body: str,
    meta: Dict[str, Any],
    post_write: Optional[Callable[[], Any]] = None,
) -> str:
    if side not in CASE_FILES:
        raise ValueError(f"unknown skill side: {side}")
    path = Path(skills_dir) / CASE_FILES[side]
    with shared_skill_lock(skills_dir):
        normalized = _normalize_meta(meta, side=side, heading=heading, path=path)
        case_id = str(normalized["case_id"])
        if find_skill_case(skills_dir, case_id) is not None:
            return case_id
        section = render_case_section(heading, body, normalized).rstrip() + "\n"
        prefix = "\n\n" if path.exists() and path.read_text(encoding="utf-8", errors="replace").strip() else ""
        append_text_locked(path, prefix + section)
    if post_write is not None:
        post_write()
    return case_id


def append_agent_case(
    skills_dir: Path,
    *,
    side: str,
    content: str,
    source_feature: Dict[str, Any],
    post_write: Optional[Callable[[], Any]] = None,
) -> str:
    text = str(content or "").strip()
    match = _HEADING_RE.search(text)
    if match:
        line_end = text.find("\n", match.start())
        if line_end == -1:
            heading = text[match.start() :].strip()
            body = ""
        else:
            heading = text[match.start() : line_end].strip()
            body = text[line_end + 1 :].strip()
    else:
        heading = f"## [Case][{side}] Agent-discovered case"
        body = text
    case_id = make_case_id(
        side=side,
        layer_id=str(source_feature.get("layer_id", "unknown")),
        feature_id=str(source_feature.get("feature_id", "unknown")),
        round_idx=int(source_feature.get("round_idx") or 0),
        trigger="agent_decision",
        harness={"heading": heading, "body": body[:500]},
    )
    if "[case:" not in heading:
        heading = f"{heading} [case: {case_id}]"
    meta = {
        "case_id": case_id,
        "side": side,
        "source_feature": source_feature,
        "write_trigger": "agent_decision",
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


def case_side_from_id_or_lookup(skills_dir: Path, case_id: str) -> Optional[str]:
    case = find_skill_case(skills_dir, case_id)
    return case.side if case else None


def _default_archive_dir() -> Path:
    return Path(__file__).parent / "analysis_output" / "self_evolution" / "_manual" / "skill_case_archive"


def compact_compressed_case_bodies(
    skills_dir: Path,
    archive_dir: Optional[Path] = None,
    *,
    side: Optional[str] = None,
) -> List[str]:
    """Remove compressed cases from active files while archiving full original text."""
    archive_root = Path(archive_dir) if archive_dir is not None else _default_archive_dir()
    changed: List[str] = []
    if side is not None and side not in CASE_FILES:
        raise ValueError(f"unknown skill side: {side}")
    selected = [(side, CASE_FILES[side])] if side is not None else list(CASE_FILES.items())
    with exclusive_skill_lock(skills_dir):
        for current_side, filename in selected:
            path = Path(skills_dir) / filename
            if not path.exists():
                continue
            archive_path = archive_root / filename.replace(".md", "_archive.md")
            with path.open("a+", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.seek(0)
                    text = f.read()
                    cases = _parse_cases_from_text(path, current_side, text)
                    if not cases:
                        continue
                    archive_text = archive_path.read_text(encoding="utf-8", errors="replace") if archive_path.exists() else ""
                    archived_ids = {
                        archived.case_id
                        for archived in _parse_cases_from_text(archive_path, current_side, archive_text)
                    }
                    archive_additions: List[str] = []
                    new_parts: List[str] = []
                    cursor = 0
                    file_changed = False
                    for case in cases:
                        new_parts.append(text[cursor : case.start])
                        usage = case.meta.get("usage") or {}
                        if usage.get("compressed"):
                            if case.case_id not in archived_ids:
                                archive_additions.append(case.section.strip())
                                archived_ids.add(case.case_id)
                            file_changed = True
                        else:
                            new_parts.append(case.section.strip())
                        cursor = case.end
                    new_parts.append(text[cursor:])
                    if archive_additions:
                        archive_path.parent.mkdir(parents=True, exist_ok=True)
                        prefix = "\n\n" if archive_text.strip() else ""
                        append_text_locked(archive_path, prefix + "\n\n".join(archive_additions).rstrip() + "\n")
                    if file_changed:
                        f.seek(0)
                        f.truncate()
                        new_text = "\n\n".join(part.strip() for part in new_parts if part.strip())
                        f.write((new_text + "\n") if new_text else "")
                        f.flush()
                        changed.append(filename)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
    return changed

