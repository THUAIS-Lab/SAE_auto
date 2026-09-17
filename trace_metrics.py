"""trace_metrics.py — Single source of truth for gate metric extraction from trace.json.

Both agent_runner.py and agent_proposer.py import from here so they can never drift.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_float(mapping: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    """Return the float value for the first key that is present and not None."""
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return _as_float(mapping[key], default)
    return default


def _get_per_hyp(input_eval: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return per-hypothesis list, handling both 'per_hypothesis' and 'hypothesis_results' keys."""
    return input_eval.get("per_hypothesis") or input_eval.get("hypothesis_results") or []


def extract_trace_metrics(trace: Dict[str, Any]) -> Dict[str, float]:
    """Extract Gate 1/2/3 metrics from a trace dict.

    Gate 1 (input): uses the single best-performing hypothesis (highest activation rate),
    independent of which hypothesis the chain selected. This avoids false 0.0 readings
    when the chain's best pair happens to be a low-activation hypothesis.

    Gates 2 & 3: anchored to the chain's best pair.

    Returns a flat dict with keys:
        input_activation_rate, input_boundary_non_activation_rate,
        output_score, best_chain_score
    """
    # ── Gates 2 & 3: find the single best pair ────────────────────────────────
    pairs: List[Dict[str, Any]] = [
        p for p in ((trace.get("chain") or {}).get("pairs", []) or [])
        if isinstance(p, dict)
    ]
    best_pair_idx_raw = (trace.get("chain") or {}).get("best_pair_idx", None)

    best_pair: Optional[Dict[str, Any]] = None
    if best_pair_idx_raw is not None:
        best_pair = next(
            (p for p in pairs if p.get("idx") == best_pair_idx_raw),
            None,
        )
    if best_pair is None and pairs:
        best_pair = max(pairs, key=lambda p: _as_float(p.get("chain_judge_score", 0)))

    if best_pair is not None:
        best_chain_score = _as_float(best_pair.get("chain_judge_score", 0))
        output_score = _first_float(
            best_pair,
            ["output_score", "output_final_score", "llm_token_match_final_score"],
        )
    else:
        best_chain_score = 0.0
        output_score = 0.0

    # ── Gate 1: best hypothesis by activation rate ────────────────────────────
    # Pick the hypothesis with the highest score_non_zero_rate regardless of which
    # hypothesis the chain selected. Handles both 'per_hypothesis' and
    # 'hypothesis_results' key names used by different pipeline versions.
    input_eval = (
        (trace.get("input_round") or {}).get("eval")
        or trace.get("input_eval")
        or {}
    )
    per_hyp: List[Dict[str, Any]] = _get_per_hyp(input_eval)

    best_hyp: Optional[Dict[str, Any]] = None
    if per_hyp:
        best_hyp = max(
            per_hyp,
            key=lambda h: (
                _as_float(h.get("score_non_zero_rate", 0.0)),
                _as_float(h.get("score_boundary_non_activation_rate", 0.0)),
            ),
        )

    input_activation_rate = _as_float(best_hyp.get("score_non_zero_rate"), 0.0) if best_hyp is not None else 0.0
    input_boundary_non_activation_rate = _as_float(best_hyp.get("score_boundary_non_activation_rate"), 0.0) if best_hyp is not None else 0.0

    return {
        "input_activation_rate": input_activation_rate,
        "input_boundary_non_activation_rate": input_boundary_non_activation_rate,
        "output_score": output_score,
        "best_chain_score": float(best_chain_score),
    }
