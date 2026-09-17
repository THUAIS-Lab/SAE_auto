# Skill: Causal Chain (Gate 3)

Reference for improving chain_judge_score after Gates 1 and 2 both pass.
Triggered when chain_judge_score (best pair) < 4/5.

---

## Priority and Prerequisites

**Gate 3 is the last priority. Never sacrifice Gate 1 or Gate 2 to improve it.**

Before touching chain score, verify:
- input_round.eval.overall_activation_rate >= 0.8
- input_round.eval.overall_boundary_non_activation_rate >= 0.8
- best output score >= 0.5

If any of these are not passing, fix them first. Do NOT adjust output hypotheses in a way
that might help the chain judge but hurts Gate 2's token match score.

**Anti-pattern to avoid:** Rewriting output hypotheses to semantically "match" the input side
in order to get a higher chain score. If the output hypothesis change causes Gate 2's
best output score to drop below 0.5, the change is wrong: revert it.
The output hypothesis must describe what the feature actually promotes in the output
distribution, not what would look good alongside the input hypothesis.

---

## Key Files to Read
- round_5/*-step5-intervention-results.json -> token_change_by_hypothesis[*].top_positive_tokens (output evidence)
- round_6/*-step6-output-hypotheses.json -> output hypotheses
- round_7/*-step7-output-hypothesis-scores.json -> Gate 2 score to verify it stays passing
- round_8/*-step8-chain-hypotheses.json -> chain pairs, chain_judge_score, and judge reasons

## What chain_judge_score Measures

The chain judge evaluates whether the input-side activation story and the output-side token-change story
form a coherent causal narrative. Low score usually means one of:
- A. Input grounding is superficially OK but hypothesis is vague or generic
- B. Output evidence exists but does not logically connect to the input concept
- C. The two hypotheses describe unrelated phenomena (disconnected pair)
- D. The feature is polysemantic — input and output genuinely belong to different semantic domains

## Diagnosis A — Vague or Generic Hypothesis Pair

Symptom: chain reason cites lack of specificity or "too broad."

Fix:
- Regenerate hypotheses with a tighter, more concrete scope (skip_observation_and_design=False).
- Add extra_input_guidance to narrow input hypothesis.
- Add extra_output_guidance to anchor output hypothesis to specific observable token changes.

## Diagnosis B — Output Evidence Does Not Connect to Input Concept

Symptom: topk_positive_tokens are sparse or unrelated to the input hypothesis concept.
Check: token_change_by_hypothesis[*].top_positive_tokens in chain.json.

Fix:
1. Increase max_activation_scale (e.g., 4.0–8.0) to produce a stronger output signal.
   If top_positive_tokens is empty, the intervention is too weak to generate clear output shifts.
2. Try intervention_scope="all_tokens" to get broader intervention coverage.
3. Increase top_k (e.g., 60–100) only if top_positive_tokens is non-empty but too sparse.
   If top_positive_tokens is already empty, larger top_k won't help.

## Diagnosis B2 — Chain Score is High but Output Score is Low

Symptom: chain_judge_score >= 4 on a pair, but output_final_score < 0.5.

This means the causal chain is sound, but the output hypothesis text doesn't match
the actual token surface forms well enough. The fix is to improve the output hypothesis
WITHOUT touching the input side.

**Action:** Read **skill_output.md → Diagnosis D** for the output-only re-run procedure.
Do NOT regenerate input hypotheses or redo the full pipeline.

## Diagnosis C — Disconnected Pair (Input + Output Describe Unrelated Things)

Symptom: chain reason says "input and output hypotheses are unrelated."

Fix: regenerate from scratch. Do NOT skip_observation_and_design.
- The current hypothesis pair is fundamentally misaligned; no parameter tweak will fix it.
- After regenerating, verify Gate 2 score stays >= 0.5.

## Diagnosis D — Polysemantic Feature (Stop Condition)

Symptom: ALL of the following are true after Gates 1 and 2 both pass:
- chain_judge_score <= 2 across 2+ rounds
- feature_type == input_type (overlap_score >= 0.7 in chain.pairs[best]) — input and output describe the same concept
- The chain reason consistently cites that input and output domains are unrelated
  (e.g., input activates on legal terms, output promotes code terminators)
- Changing hypotheses or parameters has not moved chain score above 2

Interpretation: This feature is likely polysemantic or its causal pathway is genuinely
uninterpretable. The input trigger and output effect belong to different semantic spaces,
and no hypothesis rewriting will create a plausible causal narrative.

Action: call finish(action='stop') with this diagnosis. Do not retry.
Example diagnosis: "Gates 1 and 2 pass (act=0.93, match=0.66). Chain score stuck at 2/5
across 2 rounds. feature_type=input_type, overlap_score=0.82. Chain judge consistently cites that input
(legal/settlement tokens) and output (code terminators) are from unrelated domains.
Feature is likely polysemantic. Stopping."

---

## Score Thresholds

| Score | Interpretation | Action |
|-------|---------------|--------|
| 5 | Strong causal chain | early stop |
| 4 | Acceptable chain | early stop |
| 3 | Weak but plausible | try one more round targeting Diagnosis A or B |
| 2 | Weak chain | one retry; if no improvement, check Diagnosis D |
| 1 | No chain | check Diagnosis D immediately; likely stop |

## When to Stop Retrying

Call finish(action='stop') when ANY of these conditions hold:
1. Chain score has not improved across 2 rounds AND you cannot identify a new fix strategy.
2. Diagnosis D applies (polysemantic feature).
3. A fix attempt caused Gate 2 to drop below 0.5 — stop rather than sacrifice Gate 2 for Gate 3.

Do not repeat the same tweak class more than once.
