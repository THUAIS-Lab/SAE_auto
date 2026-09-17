# Diagnosis Guide

High-level decision tree. Read this first, then follow the link to the relevant skill file.

## Priority Order (Sequential Gates)

Fix in strict order. Do not jump to a later gate while an earlier one is still failing.

### Gate 1 — Input Side
Metrics:
- input_round.eval.overall_activation_rate >= 0.8
- input_round.eval.overall_boundary_non_activation_rate >= 0.8

If failing: read **skill_input.md** for diagnosis and fix strategies.
Do NOT propose output or chain fixes while Gate 1 is open.

### Gate 2 — Output Side
Metric:
- best output score >= 0.5, from output_round.per_hypothesis[*].output_score

If Gate 1 passes but Gate 2 fails: read **skill_output.md**.

**Special case:** If chain_judge_score >= 4 exists on any pair but output < 0.5, this is
a Chain-Output Mismatch. Read **skill_output.md → Diagnosis D** for a targeted output-only fix.
Do NOT re-run the full pipeline.

### Gate 3 — Causal Chain
Metric:
- chain_judge_score (best pair) >= 4/5

Only pursue chain optimization when Gates 1 and 2 both pass.
If failing: read **skill_chain.md**.

## Early Stop

Call finish(action='stop') when ALL three gates pass simultaneously.

## Loop-Break Rule

If the same gate fails for >= 2 consecutive rounds without measurable improvement:
- Switch strategy class (not just a numeric tweak of the same knob).
- Cite in finish().diagnosis what was tried and why the next approach is different.

## Detection-like Features

If semantic_similarity is high AND chain_judge_score <= 2 after >= 2 rounds:
- Mark as likely detection-style feature.
- Call finish(action='stop') with diagnosis rather than retrying indefinitely.
