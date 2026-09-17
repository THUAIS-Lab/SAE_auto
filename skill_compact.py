#!/usr/bin/env python3
"""
skill_compact.py — Utilities for keeping the skills/ knowledge base healthy.

1. promote_effective_cases() and compact_skill_files()
   Distil effective cases into general skills, then archive cases that have
   already been incorporated.

2. deduplicate_skill_file()
   When one cases/general file exceeds its configured threshold, first remove
   clear duplicates. If it remains oversized, use a broader similarity rule
   and require the result to fit within 85% of the threshold.

3. learn_from_batch()
   Reads agent_loop_summary.json + agent_proposal_rN.json files for a completed
   batch run. Calls an LLM to synthesise cross-feature failure patterns into new
   sections appended to the appropriate cases files.

Usage (standalone):
  # Compact existing cases files only
  python skill_compact.py --compact-only

  # Learn from batch + compact
  python skill_compact.py --initial-timestamp 20260502_094104 [--layer-ids 0 6 12]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - debug helpers may run without openai installed
    OpenAI = Any  # type: ignore[misc,assignment]

from function import append_text_locked, read_api_key
from skill_case_store import CASE_FILES, compact_compressed_case_bodies, iter_skill_cases, mark_case_compressed
from support_info.llm_api_info import (
    api_key_file as DEFAULT_API_KEY_FILE,
    base_url as DEFAULT_BASE_URL,
    model_name as DEFAULT_MODEL_NAME,
)

CODE_DIR = Path(__file__).parent
SKILLS_DIR = CODE_DIR / "skills"
LOGS_DIR = CODE_DIR / "logs"

_CASES_FILES = ["skill_input_cases.md", "skill_output_cases.md", "skill_chain_cases.md"]
_CASE_SIDES = ("input", "output", "chain")
_DEDUP_KINDS = ("cases", "general")
_DEDUP_CASE_SNIPPET_CHARS = 1800
_DEDUP_TARGET_RATIO = 0.85
_EMPTY_GENERAL_SENTINEL = "[EMPTY GENERAL SKILL FILE: no general experience has been recorded in this skill file yet.]"
_PROMOTION_SYSTEM_PROMPT = (
    "Return only valid JSON. You maintain a general skill guide for an SAE feature interpretation agent. "
    "Synthesize reusable rules from multiple concrete cases. Do not overfit to feature IDs. "
    # "If the current general skill file is empty, do not claim an experience already exists."
)
_PROMOTION_EXISTING_CONTEXT_CHARS = 8000
_PROMOTION_PRIMARY_CASE_CHARS = 8000
_PROMOTION_REFERENCE_LIMIT = 20
_PROMOTION_REFERENCE_SNIPPET_CHARS = 700
_CASE_META_BLOCK_RE = re.compile(r"<!--\s*skill_case_meta\s*\{.*?\}\s*-->", re.DOTALL)
_CASE_STATS_TABLE_RE = re.compile(
    r"\s*\|(?:\s*read_count\s*\|)?\s*use_count\s*\|\s*effective_use_count\s*\|\s*compressed\s*\|\n"
    r"\|[-:| ]+\|\n"
    r"\|[^\n]*\|\n*",
    re.MULTILINE,
)


def _llm_call(client: OpenAI, model: str, system: str, user: str, max_tokens: int = 10000) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_json_object(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


def _llm_call_json(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 10000,
    max_retries: int = 2,
    retry_backoff_seconds: float = 1.5,
) -> tuple[Dict[str, Any], str]:
    """Retry an LLM request when the returned JSON object cannot be parsed."""
    retry_user = user
    raw = ""
    for attempt in range(max(0, int(max_retries)) + 1):
        raw = _llm_call(client, model, system, retry_user, max_tokens=max_tokens)
        try:
            return _parse_json_object(raw), raw
        except (TypeError, ValueError, OverflowError) as exc:
            if attempt >= max(0, int(max_retries)):
                raise ValueError(
                    f"LLM response was still invalid after {attempt + 1} attempt(s): {exc}"
                ) from exc
            retry_user = (
                f"{user}\n\nYour previous response could not be parsed: {exc}.\n"
                f"Previous response:\n{raw}\n\nReturn only one valid JSON object."
            )
            if retry_backoff_seconds > 0:
                time.sleep(float(retry_backoff_seconds))
    raise AssertionError("unreachable")


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _case_usage(case: Any) -> Dict[str, Any]:
    usage = case.meta.get("usage") or {}
    return usage if isinstance(usage, dict) else {}


def _case_priority(case: Any) -> tuple[int, int]:
    usage = _case_usage(case)
    return (
        _safe_int(usage.get("effective_use_count")),
        _safe_int(usage.get("use_count")),
    )


def _strip_case_text(text: str) -> str:
    text = _CASE_META_BLOCK_RE.sub("", str(text or ""))
    text = _CASE_STATS_TABLE_RE.sub("\n", text)
    return text.strip()


def _trim(text: str, limit: int) -> str:
    text = str(text or "")
    if int(limit) <= 0 or len(text) <= int(limit):
        return text
    return text[: int(limit)].rstrip() + f"\n[... truncated at {int(limit)} chars, {len(text)} total]"


def _case_metadata_for_prompt(case: Any) -> Dict[str, Any]:
    meta = case.meta or {}
    keep = {}
    for key in ("source_feature", "metrics_before", "metrics_after", "prior_best_score", "write_trigger"):
        if key in meta:
            keep[key] = meta[key]
    return keep


def _format_primary_case(case: Any, *, char_limit: int) -> str:
    usage = _case_usage(case)
    metadata = _case_metadata_for_prompt(case)
    body = _trim(_strip_case_text(case.text), char_limit)
    return (
        f"### Primary case: {case.case_id}\n"
        f"side: {case.side}\n"
        f"heading: {case.heading}\n"
        f"usage: {json.dumps(usage, ensure_ascii=False, sort_keys=True)}\n"
        f"metadata: {json.dumps(metadata, ensure_ascii=False, sort_keys=True)}\n\n"
        f"{body}"
    )


def _format_reference_case(case: Any, *, snippet_chars: int) -> str:
    usage = _case_usage(case)
    snippet = _strip_case_text(case.text).replace("\n", " ")
    snippet = re.sub(r"\s+", " ", snippet).strip()
    snippet = _trim(snippet, snippet_chars)
    return (
        f"- case_id: {case.case_id}\n"
        f"  heading: {case.heading}\n"
        f"  usage: {json.dumps(usage, ensure_ascii=False, sort_keys=True)}\n"
        f"  snippet: {snippet}"
    )


def build_side_promotion_context(
    *,
    skills_dir: Path = SKILLS_DIR,
    side: str,
    threshold: int = 2,
    existing_context_chars: int = _PROMOTION_EXISTING_CONTEXT_CHARS,
    primary_case_char_limit: int = _PROMOTION_PRIMARY_CASE_CHARS,
    reference_limit: int = _PROMOTION_REFERENCE_LIMIT,
    reference_snippet_chars: int = _PROMOTION_REFERENCE_SNIPPET_CHARS,
    primary_limit: Optional[int] = None,
    force_compaction: bool = False,
) -> Dict[str, Any]:
    """Build the exact per-side LLM context used for promotion."""
    if side not in _CASE_SIDES:
        raise ValueError(f"unknown side: {side}")
    skills_dir = Path(skills_dir)
    skill_file = skills_dir / f"skill_{side}.md"
    existing = skill_file.read_text(encoding="utf-8") if skill_file.exists() else ""
    existing_is_empty = not existing.strip()
    existing_for_prompt = _trim(existing, existing_context_chars) if existing.strip() else _EMPTY_GENERAL_SENTINEL

    active_cases = [case for case in iter_skill_cases(skills_dir, side=side) if not _case_usage(case).get("compressed")]
    primary_cases = [case for case in active_cases if _safe_int(_case_usage(case).get("effective_use_count")) >= int(threshold)]
    if primary_limit is not None and int(primary_limit) > 0:
        primary_cases = primary_cases[: int(primary_limit)]
    primary_ids = {case.case_id for case in primary_cases}
    reference_cases = [case for case in active_cases if case.case_id not in primary_ids]
    reference_cases.sort(key=_case_priority, reverse=True)
    reference_cases = reference_cases[: max(0, int(reference_limit))]

    primary_text = "\n\n".join(
        _format_primary_case(case, char_limit=primary_case_char_limit) for case in primary_cases
    ) or "(none)"
    reference_text = "\n".join(
        _format_reference_case(case, snippet_chars=reference_snippet_chars) for case in reference_cases
    ) or "(none)"
    primary_note = (
        "There are primary effective cases. Distill rules from them when they support a reusable operational pattern."
        if primary_cases
        else (
            "IMPORTANT: There are NO primary effective cases for this side. "
            "All available cases are weak reference cases. You may still append a tentative general rule if the reference cases show a clear reusable pattern. "
            "covered_case_ids must be empty, and any reference cases used as support must be listed in supporting_case_ids."
        )
    )

    force_note = (
        "## Hard Compaction Mode\n"
        "The active cases file has exceeded a hard context-size limit. You must return action=append when any cases are available. "
        "Prefer compressing primary cases first. If no primary cases exist, compress the strongest reference cases as supporting evidence. "
        "It is acceptable to write a narrow but operational rule, or a concise refinement of an existing rule, as long as it preserves reusable guidance and names the cases it covers. "
        "Do not return skip_no_generalization merely because the pattern resembles existing guidance; instead add a compact refinement or boundary condition that lets these cases be archived.\n\n"
        if force_compaction
        else ""
    )

    prompt = (
        f"You are updating the general skill file: {skill_file.name}\n\n"
        f"{force_note}"
        f"## Current general skill file\n"
        f"general_skill_empty: {str(existing_is_empty).lower()}\n"
        f"{existing_for_prompt}\n\n"
        f"## Primary effective cases\n"
        f"primary_case_count: {len(primary_cases)}\n"
        f"These cases have effective_use_count >= {int(threshold)} and are candidates to be distilled into "
        f"general skill rules. Primary cases in covered_case_ids and reference cases in supporting_case_ids "
        f"will be archived and removed from the active cases file after your output uses them.\n"
        f"{primary_note}\n\n"
        f"{primary_text}\n\n"
        f"## Reference cases\n"
        f"reference_case_count: {len(reference_cases)}\n"
        f"These cases have not reached the effective-use threshold or are otherwise not primary. "
        f"Use them only as weak context: they may help you see patterns, but they should not be treated as proof. "
        f"If a reference case materially supports an appended section, include it in supporting_case_ids so it can be archived.\n\n"
        f"{reference_text}\n\n"
        f"## Task\n"
        f"Write general skill sections for {skill_file.name}.\n\n"
        f"Requirements:\n"
        f"1. Summarize reusable operational rules, not individual feature stories.\n"
        f"2. Each section must say when to apply, what to change, why it works, and evidence from covered/supporting cases.\n"
        f"3. Avoid duplicate rules already present in the current general skill file.\n"
        f"4. If the general skill file is empty, do not claim an experience already exists. If primary cases exist, normally append any clear reusable rule from them. If no primary cases exist, append only when reference cases show a clear reusable pattern. In hard compaction mode, do not skip when cases are available; write a concise operational refinement instead.\n"
        f"5. Use covered_case_ids only for primary case IDs actually distilled into the section; if there are no primary cases, covered_case_ids must be [].\n"
        f"6. Use supporting_case_ids only for reference case IDs used as weak support. These supporting cases will also be archived and removed from active cases.\n"
        f"7. Return only JSON with this schema:\n"
        f"{{\n"
        f"  \"action\": \"append\" | \"skip_no_generalization\",\n"
        f"  \"sections\": [\n"
        f"    {{\n"
        f"      \"title\": \"...\",\n"
        f"      \"content\": \"markdown section content\",\n"
        f"      \"covered_case_ids\": [\"primary case ids actually distilled\"],\n"
        f"      \"supporting_case_ids\": [\"reference case ids used only as support\"],\n"
        f"      \"reason\": \"why this is general\"\n"
        f"    }}\n"
        f"  ],\n"
        f"  \"reason\": \"overall reason\"\n"
        f"}}\n"
    )
    return {
        "side": side,
        "skill_file": str(skill_file),
        "existing_chars": len(existing),
        "existing_stripped_chars": len(existing.strip()),
        "general_skill_empty": existing_is_empty,
        "primary_case_ids": [case.case_id for case in primary_cases],
        "reference_case_ids": [case.case_id for case in reference_cases],
        "active_case_count": len(active_cases),
        "primary_case_count": len(primary_cases),
        "reference_case_count": len(reference_cases),
        "force_compaction": bool(force_compaction),
        "system": _PROMOTION_SYSTEM_PROMPT,
        "prompt": prompt,
    }


def _normalise_case_ids(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        if isinstance(item, str):
            case_id = item.strip()
        elif isinstance(item, dict):
            case_id = str(item.get("case_id") or "").strip()
        else:
            case_id = ""
        if case_id and case_id not in out:
            out.append(case_id)
    return out


def _section_markdown(section: Dict[str, Any]) -> str:
    title = str(section.get("title") or "Distilled Skill Experience").strip()
    content = str(section.get("content") or "").strip()
    if not content:
        return ""
    if not content.lstrip().startswith("##"):
        content = f"## {title}\n\n{content}"
    return content.strip()


def promote_effective_cases(
    client: OpenAI,
    model: str,
    skills_dir: Path = SKILLS_DIR,
    *,
    threshold: int = 2,
    reference_limit: int = _PROMOTION_REFERENCE_LIMIT,
    verbose: bool = True,
    force_compaction: bool = False,
) -> List[Dict[str, Any]]:
    """Distill repeatedly effective cases into framework skill files.

    Promotion is intentionally per side, not per case: input/output/chain each
    get at most one LLM call. Primary cases (effective_use_count >= threshold)
    provide the evidence to compress; lower-confidence cases are included only
    as reference context.
    """
    results: List[Dict[str, Any]] = []
    for side in _CASE_SIDES:
        context = build_side_promotion_context(
            skills_dir=skills_dir,
            side=side,
            threshold=threshold,
            reference_limit=reference_limit,
            force_compaction=force_compaction,
        )
        primary_ids = set(context["primary_case_ids"])
        reference_ids = set(context["reference_case_ids"])
        if not primary_ids and not reference_ids:
            if verbose:
                print(f"  promote {side}: no cases to summarize")
            continue
        if not primary_ids and verbose:
            print(f"  promote {side}: no primary effective cases; calling LLM with reference-only context")

        skill_file = Path(context["skill_file"])
        raw = ""
        try:
            data, raw = _llm_call_json(
                client,
                model,
                system=context["system"],
                user=context["prompt"],
            )
        except Exception as exc:
            result = {
                "side": side,
                "status": "failed",
                "target": skill_file.name,
                "primary_case_ids": sorted(primary_ids),
                "reference_case_ids": context["reference_case_ids"],
                "error": str(exc),
                "raw_llm_response": raw,
                "force_compaction": bool(force_compaction),
            }
            results.append(result)
            if verbose:
                print(f"  promote {side}: failed: {exc}")
            continue

        action = str(data.get("action") or "skip_no_generalization")
        sections = data.get("sections") if isinstance(data.get("sections"), list) else []
        accepted_sections: List[Dict[str, Any]] = []
        append_blocks: List[str] = []
        covered_ids: List[str] = []
        supporting_ids: List[str] = []
        ignored_case_ids: List[str] = []

        if action == "append":
            for section in sections:
                if not isinstance(section, dict):
                    continue
                markdown = _section_markdown(section)
                raw_covered = _normalise_case_ids(section.get("covered_case_ids"))
                raw_supporting = _normalise_case_ids(section.get("supporting_case_ids"))
                section_covered = [cid for cid in raw_covered if cid in primary_ids]
                section_supporting = [cid for cid in raw_supporting if cid in reference_ids]
                ignored_case_ids.extend(cid for cid in raw_covered if cid not in primary_ids)
                ignored_case_ids.extend(cid for cid in raw_supporting if cid not in reference_ids)
                if not markdown:
                    continue
                if primary_ids and not section_covered:
                    continue
                if not primary_ids and not section_supporting:
                    continue
                for cid in section_covered:
                    if cid not in covered_ids:
                        covered_ids.append(cid)
                for cid in section_supporting:
                    if cid not in supporting_ids:
                        supporting_ids.append(cid)
                accepted_sections.append({
                    "title": str(section.get("title") or "").strip(),
                    "covered_case_ids": section_covered,
                    "supporting_case_ids": section_supporting,
                    "reason": str(section.get("reason") or ""),
                })
                source_label = "distilled_from_cases" if section_covered else "distilled_from_reference_cases"
                source_ids = section_covered if section_covered else section_supporting
                append_blocks.append(
                    f"<!-- {source_label}: " + ", ".join(source_ids) + " -->\n" + markdown
                )

        if append_blocks:
            existing = skill_file.read_text(encoding="utf-8") if skill_file.exists() else ""
            prefix = "\n\n" if existing.strip() else ""
            append_text_locked(skill_file, prefix + "\n\n".join(append_blocks).rstrip() + "\n")
            for case_id in covered_ids:
                mark_case_compressed(skills_dir, case_id, result="appended_batch_covered", target=skill_file.name)
            for case_id in supporting_ids:
                mark_case_compressed(skills_dir, case_id, result="appended_batch_supporting", target=skill_file.name)
            status = "appended" if covered_ids else "appended_reference_only"
        else:
            status = "skip_no_generalization" if action != "append" else "skipped_no_valid_sections"

        result = {
            "side": side,
            "status": status,
            "action": action,
            "target": skill_file.name,
            "primary_case_ids": sorted(primary_ids),
            "reference_case_ids": context["reference_case_ids"],
            "covered_case_ids": covered_ids,
            "supporting_case_ids": supporting_ids,
            "compressed_case_ids": covered_ids + supporting_ids,
            "ignored_case_ids": sorted(set(ignored_case_ids)),
            "sections": accepted_sections,
            "reason": str(data.get("reason") or ""),
            "raw_llm_response": raw,
            "force_compaction": bool(force_compaction),
        }
        results.append(result)
        if verbose:
            print(
                f"  promote {side}: {status} -> {skill_file.name} "
                f"(primary={len(primary_ids)}, covered={len(covered_ids)}, "
                f"supporting={len(supporting_ids)}, refs={len(context['reference_case_ids'])})"
            )
    return results


# ── Threshold-triggered deduplication ────────────────────────────────────────

def _skill_path(skills_dir: Path, side: str, kind: str) -> Path:
    if side not in _CASE_SIDES:
        raise ValueError(f"unknown skill side: {side}")
    if kind not in _DEDUP_KINDS:
        raise ValueError(f"unknown skill kind: {kind}")
    filename = CASE_FILES[side] if kind == "cases" else f"skill_{side}.md"
    return Path(skills_dir) / filename


def _file_chars(path: Path) -> int:
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8", errors="replace"))


def _format_cases_for_dedup(skills_dir: Path, side: str) -> str:
    blocks: List[str] = []
    for case in iter_skill_cases(skills_dir, side=side):
        usage = _case_usage(case)
        body = _trim(_strip_case_text(case.text), _DEDUP_CASE_SNIPPET_CHARS)
        blocks.append(
            f"### case_id: {case.case_id}\n"
            f"section_chars: {len(case.section)}\n"
            f"effective_use_count: {_safe_int(usage.get('effective_use_count'))}\n"
            f"use_count: {_safe_int(usage.get('use_count'))}\n\n"
            f"{body}"
        )
    return "\n\n".join(blocks)


def _case_dedup_prompt(
    *,
    skills_dir: Path,
    side: str,
    stage: str,
    current_chars: int,
    threshold_chars: int,
    target_chars: int,
) -> str:
    if stage == "strict":
        policy = (
            "Use a strict similarity standard. Archive only cases that describe substantially the same "
            "failure pattern and the same reusable fix. Keep the clearest or best-supported representative "
            "from every duplicate group. Do not archive unique boundary conditions or distinct fixes."
        )
        size_requirement = (
            f"Reduce the file below the threshold of {threshold_chars} characters if clear duplicates permit it."
        )
    else:
        policy = (
            "Use a broader similarity standard. Cases may be merged when they teach the same reusable "
            "operational fix even if their surface feature, wording, or evidence differs. Preserve at least "
            "one representative for each genuinely distinct lesson."
        )
        size_requirement = (
            f"You must choose enough redundant cases that the remaining active file is at most "
            f"{target_chars} characters (85% of the {threshold_chars}-character threshold)."
        )
    active_case_count = len(iter_skill_cases(skills_dir, side=side))
    return (
        f"Deduplicate the active {side} case file.\n\n"
        f"stage: {stage.upper()}\n"
        f"active_case_count: {active_case_count}\n"
        f"current_chars: {current_chars}\n"
        f"threshold_chars: {threshold_chars}\n"
        f"target_chars: {target_chars}\n\n"
        f"{policy}\n{size_requirement}\n\n"
        "Return only JSON with this schema:\n"
        '{"archive_case_ids":["case ids to remove from the active file"],"reason":"short rationale"}\n'
        "Never invent a case ID and never select every active case. Your response is invalid if "
        "archive_case_ids contains all active case IDs. For every similarity group, keep at least one "
        "clear, well-supported representative in the active file; therefore at least one representative "
        "case ID must be absent from archive_case_ids. Selected originals will be archived.\n\n"
        f"## Active cases\n{_format_cases_for_dedup(skills_dir, side)}"
    )


def _case_dedup_all_cases_correction_prompt(original_prompt: str) -> str:
    return (
        f"{original_prompt}\n\n"
        "## Required correction\n"
        "Your previous response selected every active case for archival. That response is invalid and "
        "was not applied. Re-evaluate the similarity groups and return a corrected JSON response. "
        "You MUST leave at least one clear, well-supported representative active for every distinct "
        "lesson. Consequently archive_case_ids MUST be a strict subset of the active case IDs."
    )


def _project_case_chars_after_archive(path: Path, cases: List[Any], archive_ids: set[str]) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    parts: List[str] = []
    cursor = 0
    for case in cases:
        parts.append(text[cursor : case.start])
        if case.case_id not in archive_ids:
            parts.append(case.section.strip())
        cursor = case.end
    parts.append(text[cursor:])
    projected = "\n\n".join(part.strip() for part in parts if part.strip())
    return len(projected + "\n") if projected else 0


def _apply_case_dedup_stage(
    *,
    skills_dir: Path,
    side: str,
    archive_dir: Path,
    stage: str,
    data: Dict[str, Any],
    required_max_chars: Optional[int] = None,
) -> Dict[str, Any]:
    cases = iter_skill_cases(skills_dir, side=side)
    active_by_id = {case.case_id: case for case in cases}
    requested = _normalise_case_ids(data.get("archive_case_ids"))
    selected = [case_id for case_id in requested if case_id in active_by_id]
    ignored = [case_id for case_id in requested if case_id not in active_by_id]
    path = _skill_path(skills_dir, side, "cases")
    before_chars = _file_chars(path)

    if not selected:
        return {
            "stage": stage,
            "status": "no_valid_duplicates",
            "before_chars": before_chars,
            "after_chars": before_chars,
            "archived_case_ids": [],
            "ignored_case_ids": ignored,
            "reason": str(data.get("reason") or ""),
        }
    if len(selected) >= len(cases):
        return {
            "stage": stage,
            "status": "rejected_all_cases",
            "before_chars": before_chars,
            "after_chars": before_chars,
            "archived_case_ids": [],
            "ignored_case_ids": ignored,
            "reason": "At least one active case must remain.",
        }

    projected_chars = _project_case_chars_after_archive(path, cases, set(selected))
    if projected_chars >= before_chars:
        return {
            "stage": stage,
            "status": "rejected_no_reduction",
            "before_chars": before_chars,
            "after_chars": before_chars,
            "projected_chars": projected_chars,
            "archived_case_ids": [],
            "ignored_case_ids": ignored,
            "reason": str(data.get("reason") or ""),
        }
    if required_max_chars is not None and projected_chars > int(required_max_chars):
        return {
            "stage": stage,
            "status": "rejected_target_not_met",
            "before_chars": before_chars,
            "after_chars": before_chars,
            "projected_chars": projected_chars,
            "required_max_chars": int(required_max_chars),
            "archived_case_ids": [],
            "ignored_case_ids": ignored,
            "reason": str(data.get("reason") or ""),
        }

    target = f"dedup:{stage}"
    archived: List[str] = []
    for case_id in selected:
        if mark_case_compressed(
            skills_dir,
            case_id,
            result=f"deduplicated_{stage}",
            target=target,
        ) is not None:
            archived.append(case_id)
    changed_files = compact_compressed_case_bodies(skills_dir, archive_dir=archive_dir, side=side)
    return {
        "stage": stage,
        "status": "applied" if archived else "no_change",
        "before_chars": before_chars,
        "after_chars": _file_chars(path),
        "projected_chars": projected_chars,
        "archived_case_ids": archived,
        "ignored_case_ids": ignored,
        "changed_files": changed_files,
        "reason": str(data.get("reason") or ""),
    }


def _general_dedup_prompt(
    *,
    side: str,
    stage: str,
    content: str,
    threshold_chars: int,
    target_chars: int,
) -> str:
    if stage == "strict":
        policy = (
            "Use a strict similarity standard. Merge or remove only sections that give substantially the "
            "same operational guidance. Preserve every unique rule, trigger, boundary condition, and fix."
        )
        size_requirement = (
            f"Reduce the file below the threshold of {threshold_chars} characters if clear overlap permits it."
        )
    else:
        policy = (
            "Use a broader similarity standard. Merge sections that share the same reusable intervention, "
            "even when their examples or phrasing differ. Shorten repeated evidence while preserving distinct rules."
        )
        size_requirement = (
            f"The returned content must be at most {target_chars} characters "
            f"(85% of the {threshold_chars}-character threshold)."
        )
    return (
        f"Deduplicate the general {side} skill file.\n\n"
        f"stage: {stage.upper()}\n"
        f"current_chars: {len(content)}\n"
        f"threshold_chars: {threshold_chars}\n"
        f"target_chars: {target_chars}\n\n"
        f"{policy}\n{size_requirement}\n"
        "Return the complete revised Markdown, not a patch. Return only JSON with this schema:\n"
        '{"content":"complete revised Markdown","reason":"short rationale"}\n\n'
        f"## Current file\n{content}"
    )


def _archive_general_skill(
    *,
    archive_dir: Path,
    checkpoint_name: str,
    path: Path,
    stage: str,
    content: str,
) -> Path:
    archive_root = Path(archive_dir) / "general_skills"
    archive_root.mkdir(parents=True, exist_ok=True)
    checkpoint = re.sub(r"[^A-Za-z0-9_.-]+", "-", checkpoint_name or "checkpoint")
    base = archive_root / f"{checkpoint}-{path.stem}-{stage}.md"
    archive_path = base
    suffix = 2
    while archive_path.exists():
        archive_path = base.with_name(f"{base.stem}-{suffix}{base.suffix}")
        suffix += 1
    archive_path.write_text(content, encoding="utf-8")
    return archive_path


def _replace_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.dedup.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _apply_general_dedup_stage(
    *,
    path: Path,
    archive_dir: Path,
    checkpoint_name: str,
    stage: str,
    data: Dict[str, Any],
    required_max_chars: Optional[int] = None,
) -> Dict[str, Any]:
    before = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    content = data.get("content")
    revised = content.strip() + "\n" if isinstance(content, str) and content.strip() else ""
    if not revised:
        return {
            "stage": stage,
            "status": "rejected_empty_output",
            "before_chars": len(before),
            "after_chars": len(before),
            "reason": str(data.get("reason") or ""),
        }
    if len(revised) >= len(before):
        return {
            "stage": stage,
            "status": "rejected_no_reduction",
            "before_chars": len(before),
            "after_chars": len(before),
            "proposed_chars": len(revised),
            "reason": str(data.get("reason") or ""),
        }
    if required_max_chars is not None and len(revised) > int(required_max_chars):
        return {
            "stage": stage,
            "status": "rejected_target_not_met",
            "before_chars": len(before),
            "after_chars": len(before),
            "proposed_chars": len(revised),
            "required_max_chars": int(required_max_chars),
            "reason": str(data.get("reason") or ""),
        }

    archive_path = _archive_general_skill(
        archive_dir=archive_dir,
        checkpoint_name=checkpoint_name,
        path=path,
        stage=stage,
        content=before,
    )
    _replace_text(path, revised)
    return {
        "stage": stage,
        "status": "applied",
        "before_chars": len(before),
        "after_chars": len(revised),
        "archive_path": str(archive_path),
        "reason": str(data.get("reason") or ""),
    }


def deduplicate_skill_file(
    client: OpenAI,
    model: str,
    *,
    skills_dir: Path,
    side: str,
    kind: str,
    threshold_chars: int,
    archive_dir: Path,
    checkpoint_name: str,
    target_ratio: float = _DEDUP_TARGET_RATIO,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Deduplicate one independently thresholded cases or general skill file."""
    path = _skill_path(skills_dir, side, kind)
    threshold = int(threshold_chars)
    if threshold <= 0:
        raise ValueError("threshold_chars must be positive")
    ratio = float(target_ratio)
    if not 0 < ratio < 1:
        raise ValueError("target_ratio must be between 0 and 1")
    initial_chars = _file_chars(path)
    target_chars = int(threshold * ratio)
    result: Dict[str, Any] = {
        "side": side,
        "kind": kind,
        "file": path.name,
        "triggered": initial_chars > threshold,
        "initial_chars": initial_chars,
        "threshold_chars": threshold,
        "target_chars": target_chars,
        "target_ratio": ratio,
        "stages": [],
    }
    if initial_chars <= threshold:
        result.update({
            "status": "not_needed",
            "final_chars": initial_chars,
            "threshold_met": True,
            "target_required": False,
            "target_met": True,
        })
        return result

    if verbose:
        print(
            f"  dedup {side}/{kind}: {initial_chars} > {threshold}; running strict duplicate removal",
            flush=True,
        )

    for stage in ("strict", "relaxed"):
        current_chars = _file_chars(path)
        if stage == "relaxed" and current_chars <= threshold:
            break
        try:
            if kind == "cases":
                prompt = _case_dedup_prompt(
                    skills_dir=skills_dir,
                    side=side,
                    stage=stage,
                    current_chars=current_chars,
                    threshold_chars=threshold,
                    target_chars=target_chars,
                )
            else:
                current = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
                prompt = _general_dedup_prompt(
                    side=side,
                    stage=stage,
                    content=current,
                    threshold_chars=threshold,
                    target_chars=target_chars,
                )
            data, raw = _llm_call_json(
                client,
                model,
                system=(
                    "Return only valid JSON. Remove redundant skill experience while preserving distinct, "
                    "actionable operational knowledge."
                ),
                user=prompt,
            )
            required_max = target_chars if stage == "relaxed" else None
            if kind == "cases":
                stage_result = _apply_case_dedup_stage(
                    skills_dir=skills_dir,
                    side=side,
                    archive_dir=archive_dir,
                    stage=stage,
                    data=data,
                    required_max_chars=required_max,
                )
                if stage_result["status"] == "rejected_all_cases":
                    invalid_attempt = stage_result
                    corrected_data, corrected_raw = _llm_call_json(
                        client,
                        model,
                        system=(
                            "Return only valid JSON. Preserve representative experience. Never return "
                            "all active case IDs in archive_case_ids."
                        ),
                        user=_case_dedup_all_cases_correction_prompt(prompt),
                    )
                    stage_result = _apply_case_dedup_stage(
                        skills_dir=skills_dir,
                        side=side,
                        archive_dir=archive_dir,
                        stage=stage,
                        data=corrected_data,
                        required_max_chars=required_max,
                    )
                    stage_result["correction_reason"] = "initial_response_selected_all_cases"
                    stage_result["invalid_attempts"] = [invalid_attempt]
            else:
                stage_result = _apply_general_dedup_stage(
                    path=path,
                    archive_dir=archive_dir,
                    checkpoint_name=checkpoint_name,
                    stage=stage,
                    data=data,
                    required_max_chars=required_max,
                )
        except Exception as exc:
            stage_result = {
                "stage": stage,
                "status": "failed",
                "before_chars": current_chars,
                "after_chars": _file_chars(path),
                "error": str(exc),
            }
        result["stages"].append(stage_result)
        if verbose:
            print(
                f"  dedup {side}/{kind} {stage}: {stage_result['status']} "
                f"({stage_result['before_chars']} -> {stage_result['after_chars']} chars)",
                flush=True,
            )

    final_chars = _file_chars(path)
    relaxed_ran = any(stage["stage"] == "relaxed" for stage in result["stages"])
    threshold_met = final_chars <= threshold
    target_met = not relaxed_ran or final_chars <= target_chars
    result.update({
        "status": "ok" if threshold_met and target_met else "incomplete",
        "final_chars": final_chars,
        "threshold_met": threshold_met,
        "target_required": relaxed_ran,
        "target_met": target_met,
    })
    return result


# ── Skill compaction ──────────────────────────────────────────────────────────

def compact_skill_files(
    client: OpenAI,
    model: str,
    skills_dir: Path = SKILLS_DIR,
    *,
    archive_dir: Optional[Path] = None,
    verbose: bool = True,
) -> List[str]:
    """Compact cases without letting an LLM rewrite/truncate whole files.

    Full-file LLM compaction is unsafe for large markdown files because it can
    silently drop later cases. The checkpoint-safe compaction here only shrinks
    cases already marked as compressed after promotion, preserving case IDs and
    counters in the active case files.
    """
    changed = compact_compressed_case_bodies(skills_dir, archive_dir=archive_dir)
    if verbose:
        if changed:
            print(f"  Compacted compressed case bodies: {changed}")
        else:
            print("  No compressed case bodies to compact")
    return changed


# ── Batch pattern learning ────────────────────────────────────────────────────

def _collect_batch_results(initial_timestamp: str, layer_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Collect summaries + last diagnosis for each feature in the batch."""
    results = []
    for summary_path in sorted(LOGS_DIR.glob(f"layer-*/feature-*/{initial_timestamp}/agent_loop_summary.json")):
        parts = summary_path.parts
        layer_part = next((p for p in parts if p.startswith("layer-")), None)
        feature_part = next((p for p in parts if p.startswith("feature-")), None)
        if not layer_part or not feature_part:
            continue
        layer_id = int(layer_part.replace("layer-", ""))
        if layer_ids and layer_id not in layer_ids:
            continue
        feature_id = feature_part.replace("feature-", "")

        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Collect diagnoses from all agent proposals in this feature's run
        diagnoses: List[str] = []
        ts_dir = summary_path.parent
        for proposal_file in sorted(ts_dir.glob("agent_proposal_r*.json")):
            try:
                p = json.loads(proposal_file.read_text(encoding="utf-8"))
                d = p.get("diagnosis", "")
                if d:
                    diagnoses.append(d[:300])
            except Exception:
                pass

        results.append({
            "layer_id": layer_id,
            "feature_id": feature_id,
            "final_chain_score": summary.get("final_chain_score", 0),
            "final_input_activation_rate": summary.get("final_input_activation_rate", 0),
            "final_input_boundary_non_activation_rate": summary.get("final_input_boundary_non_activation_rate", 0),
            "final_output_score": summary.get("final_output_score", 0),
            "rounds_run": summary.get("rounds_run", 0),
            "agent_token_cost": summary.get("agent_token_cost", {}).get("total_tokens", 0),
            "diagnoses": diagnoses,
        })
    return results


def _summarise_results(results: List[Dict[str, Any]]) -> str:
    """Build a compact text summary of batch results for the LLM."""
    lines = []
    gate1_fail = [r for r in results if r["final_input_activation_rate"] < 0.8 or r["final_input_boundary_non_activation_rate"] < 0.8]
    gate2_fail = [r for r in results if r["final_output_score"] < 0.5 and r["final_input_activation_rate"] >= 0.8]
    gate3_fail = [r for r in results if r["final_chain_score"] < 4 and r["final_input_activation_rate"] >= 0.8 and r["final_output_score"] >= 0.5]

    lines.append(f"Batch: {len(results)} features total")
    lines.append(f"  Still failing Gate 1: {len(gate1_fail)}")
    lines.append(f"  Still failing Gate 2 (Gate 1 OK): {len(gate2_fail)}")
    lines.append(f"  Still failing Gate 3 (Gates 1+2 OK): {len(gate3_fail)}")
    lines.append("")

    def _feature_block(r: Dict[str, Any]) -> str:
        diag_str = " | ".join(r["diagnoses"][:2]) if r["diagnoses"] else "(no diagnosis)"
        return (
            f"  L{r['layer_id']}-F{r['feature_id']}: "
            f"act={r['final_input_activation_rate']:.2f} bnd={r['final_input_boundary_non_activation_rate']:.2f} "
            f"out={r['final_output_score']:.2f} chain={r['final_chain_score']} rounds={r['rounds_run']}\n"
            f"    diagnosis: {diag_str[:200]}"
        )

    if gate1_fail:
        lines.append("== Gate 1 failures ==")
        for r in gate1_fail[:12]:
            lines.append(_feature_block(r))
        lines.append("")

    if gate2_fail:
        lines.append("== Gate 2 failures ==")
        for r in gate2_fail[:8]:
            lines.append(_feature_block(r))
        lines.append("")

    if gate3_fail:
        lines.append("== Gate 3 failures ==")
        for r in gate3_fail[:6]:
            lines.append(_feature_block(r))

    return "\n".join(lines)


def learn_from_batch(
    client: OpenAI,
    model: str,
    initial_timestamp: str,
    layer_ids: Optional[List[int]] = None,
    skills_dir: Path = SKILLS_DIR,
    *,
    verbose: bool = True,
) -> Dict[str, str]:
    """Synthesise cross-feature patterns from a completed batch into the cases files.
    Returns dict of {filename: content_appended}."""
    results = _collect_batch_results(initial_timestamp, layer_ids)
    if not results:
        if verbose:
            print(f"  No batch results found for timestamp={initial_timestamp}")
        return {}

    batch_summary = _summarise_results(results)
    if verbose:
        print(f"  Collected {len(results)} feature results")
        print(batch_summary[:600])

    # Read existing cases files for context (so LLM avoids duplicating)
    existing_cases: Dict[str, str] = {}
    for fn in _CASES_FILES:
        p = skills_dir / fn
        if p.exists():
            existing_cases[fn] = p.read_text(encoding="utf-8")[:3000]

    prompt = (
        f"You are analysing results from a batch SAE feature interpretation run to extract "
        f"reusable patterns for the skills knowledge base.\n\n"
        f"## Batch results\n{batch_summary}\n\n"
        f"## Existing cases (abbreviated, do not duplicate)\n"
        + "\n".join(f"### {fn}\n{txt[:800]}\n" for fn, txt in existing_cases.items())
        + "\n\n"
        f"## Task\n"
        f"Based on the failure patterns above, write NEW ## sections for the relevant cases files.\n"
        f"Each section must follow this format:\n"
        f"  ## [Pattern name] (batch {initial_timestamp[:8]})\n"
        f"  **When:** <the metric signature / trace pattern that triggers this>\n"
        f"  **Fix:** <specific harness parameter + value>\n"
        f"  **Evidence:** <cite layer/feature IDs and metric deltas from the batch>\n\n"
        f"Only write patterns that appear in >=2 features. Skip patterns already covered above.\n"
        f"Return a JSON object with keys matching the cases filenames that need updating:\n"
        f'{{"skill_input_cases.md": "...", "skill_output_cases.md": "...", "skill_chain_cases.md": "..."}}\n'
        f"Omit a key if no new patterns apply to that file. Return only valid JSON."
    )

    try:
        updates, raw = _llm_call_json(
            client,
            model,
            system="Return only valid JSON. No markdown fences.",
            user=prompt,
        )
    except Exception as e:
        if verbose:
            print(f"  Failed to parse LLM output as JSON: {e}")
        return {}

    written: Dict[str, str] = {}
    for filename, new_section in updates.items():
        if filename not in _CASES_FILES:
            continue
        if not new_section or not new_section.strip():
            continue
        path = skills_dir / filename
        skills_dir.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        path.write_text(existing.rstrip() + "\n\n" + new_section.strip() + "\n", encoding="utf-8")
        if verbose:
            print(f"  Appended {len(new_section)} chars to {filename}")
        written[filename] = new_section
    return written


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compact skill files and/or learn from batch results.")
    p.add_argument("--compact-only", action="store_true", help="Only compact skill files, skip batch learning")
    p.add_argument("--initial-timestamp", default=None, help="Batch timestamp to learn from")
    p.add_argument("--layer-ids", nargs="*", type=int, default=None)
    p.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--llm-model", default=DEFAULT_MODEL_NAME)
    p.add_argument("--llm-api-key-file", default=DEFAULT_API_KEY_FILE)
    p.add_argument("--archive-dir", type=Path, default=None, help="Directory for archived full compressed cases")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    api_key = read_api_key(args.llm_api_key_file)
    client = OpenAI(base_url=args.llm_base_url, api_key=api_key)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] === skill_compact ===")

    if not args.compact_only and args.initial_timestamp:
        print(f"Learning from batch {args.initial_timestamp}...")
        learn_from_batch(client, args.llm_model, args.initial_timestamp, args.layer_ids)

    print("Promoting effective cases...")
    promoted = promote_effective_cases(client, args.llm_model)
    if promoted:
        print(f"Promoted/skipped effective cases: {len(promoted)}")

    print("Compacting cases files...")
    compacted = compact_skill_files(client, args.llm_model, archive_dir=args.archive_dir)
    if compacted:
        print(f"Compacted: {compacted}")
    else:
        print("Nothing to compact.")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Done.")


if __name__ == "__main__":
    main()
