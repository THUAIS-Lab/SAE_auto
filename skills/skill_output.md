# Skill: Output Side (Gate 2)

Reference for diagnosing and fixing output-side failures.
Triggered when best output score < 0.5, after Gate 1 passes.

## Key Files to Read
- round_5/*-step5-intervention-results.json -> token_change results, top_positive_tokens, intervention settings
- round_6/*-step6-output-hypotheses.json -> generated output hypotheses
- round_7/*-step7-output-hypothesis-scores.json -> per-hypothesis match scores and matched token evidence

## Intervention Scope

Two options for where to apply feature steering. Choose based on what you observe in step5 token_change evidence — do not follow a fixed order.

### max_activation_token (SAE-guided position)
- Steers at the token where the SAE feature activates most strongly
- Semantically grounded: the steering is applied where the feature naturally fires
- Best empirical results in most cases
- Parameters: max_activation_scale (default 2.0; try 3.0–3.5 if evidence is still weak)

### last_token_only (penultimate token, position -2)
- Steers at the second-to-last token regardless of SAE activation
- Consider switching to this when: top-k output tokens are semantically scattered with no clear direction, OR output_score is consistently low across multiple max_activation attempts
- Parameters: last_token_scale (default 1.0; try 1.5–2.0 if evidence is sparse)

Do NOT use all_tokens — it dilutes the steering signal across all positions.

## Diagnosis A — Sparse or Diffuse Token-Change Evidence

Cause: intervention is too weak or too narrow; delta tokens do not clearly reflect the feature.

Fixes:
1. First try: increase max_activation_scale to 3.0 or 3.5
2. Increase top_k to widen evidence coverage: 10 → 20 → 30
3. If top-k tokens remain semantically incoherent (no clear direction) after scaling: switch to intervention_scope=last_token_only, last_token_scale=1.0–2.0
4. If both scopes produce weak evidence: try custom_steering_prompts with neutral prefixes to reduce lexical leakage

## Diagnosis B — Lexical Leakage (Input Tokens Pollute Output Evidence)

Cause: steering prompts contain words from the input hypothesis, polluting delta tokens.

Symptom: topk_positive_tokens overlap heavily with activation sentence keywords.

Fix: apply neutral steering.
- Set skip_observation_and_design=True (reuse existing hypotheses).
- Set custom_steering_prompts to short, neutral prefixes:
  ["The answer is", "In conclusion,", "Therefore,", "The result is", "I think that", "According to the evidence,"]
- Rules for custom prompts:
  - <= 8 words each
  - No nouns from the input hypothesis
  - Declarative and neutral tone

## Diagnosis C — Weak Output Hypothesis Quality

Cause: output hypotheses do not match the actual token-change evidence.

Fix:
- Add extra_output_guidance to anchor hypothesis to specific token patterns:
  - "Focus on the specific token type that changes most when the feature is activated."
  - "Output hypothesis should describe a concrete surface-form change, not a broad semantic shift."
- If evidence looks reasonable but hypothesis is still misaligned: regenerate (skip_observation_and_design=False).

## Combined Fix Sequence

If score < 0.3 (severe):  increase max_activation_scale to 3.0–3.5 + top_k increase; if tokens remain incoherent, try last_token_only
If score 0.3–0.5 (borderline):  increase max_activation_scale first; if no improvement after one round, try last_token_only

---

## Diagnosis D — Chain is Strong but Output Score is Low (Chain-Output Mismatch)

**Trigger:** chain_judge_score >= 4 but output_final_score < 0.5 on the same pair.

This is a common pattern: the causal chain is solid but the output hypothesis describes
the output tokens too abstractly or too broadly for the LLM token-matcher to recognize them.

### Root Cause

The output hypothesis may be conceptually correct but phrased in a way that doesn't
match the specific surface-form tokens. E.g., hypothesis says "C++ container member functions"
but tokens are "empty", "front", "resize" — the matcher needs a more token-anchored description.

### Fix (Output-Only Re-run)

**Do NOT re-run the full pipeline.** Only re-run steps 6-7 with better output guidance:

1. Read the chain pair with chain_judge_score >= 4. Note its `input_hypothesis` and `output_hypothesis`.
2. Read `topk_positive_tokens` from step 5 output for the corresponding hypothesis.
3. Set `extra_output_guidance` to anchor the output hypothesis to specific tokens:
   - "The output hypothesis must name specific token surface forms that appear in the top-k output changes. "
     "E.g., if top tokens are ['empty', 'front', 'resize'], the hypothesis should say "
     "'C++ STL container methods like empty, front, resize, assign, and erase' rather than "
     "'C++ container operations'. Be concrete and list actual tokens."
4. Set `rerun_steps = [6, 7, 8, 9]` to regenerate output hypotheses without touching input side.
5. If `extra_output_guidance` is not enough: also try `intervention_scope=max_activation_token`
   with `max_activation_scale=3.0` to get stronger token-change evidence, then re-run steps 6-9.

### Key Principle

The input hypothesis and chain logic are already good — do not regenerate them.
Only fix the output hypothesis text to better describe the actual tokens.
Preserving the chain means keeping `skip_observation_and_design=True` and only rerunning steps 6+.


<!-- auto-distilled from 43 cases -->
## [Principle: Enforce Verbatim Token Enumeration]
**When to apply**: When output_score < 0.5 and topk_positive_tokens contain heterogeneous, multilingual, or whitespace-sensitive tokens (e.g., `▁<`, `Externé`, `).;`, `▁$=$`) — especially if prior hypotheses used abstractions like “punctuation”, “non-English terms”, or “code symbols”.  
**What to change**: Require *exact, unmodified token strings* from `topk_positive_tokens` — including leading `▁`, Unicode, casing, and punctuation — embedded directly in the hypothesis using rigid templating (e.g., `'The exact tokens are: [token1, token2, token3]'` or `'Token examples: ▁<, ▁", ▁\n'`). Ban all grouping labels, categories, or paraphrases.  
**Why**: The token-matcher is surface-form–sensitive and fails on semantic generalizations; verbatim enumeration eliminates ambiguity and aligns hypothesis syntax with matcher expectations.

## [Principle: Contextualize Tokens via Real Phrasal Usage]
**When to apply**: When tokens appear in consistent syntactic roles (e.g., adjectives before nouns in service descriptions, substrings in identifiers) but hypotheses describe only isolated lexical properties (e.g., “tokens starting with ‘Char’”) without usage context.  
**What to change**: Anchor tokens to *actual observed multi-token phrases* from intervention outputs (e.g., `'previous projects'`, `'versatile producer'`, `'idk answer'`) and require hypotheses to describe how tokens function *within those phrases*.  
**Why**: Abstract lexical patterns often mismatch the matcher’s reliance on contextual token co-occurrence; grounding in real phrase fragments improves surface alignment and avoids overgeneralization.

## [Principle: Normalize SAE Token Representations for Human Readability]
**When to apply**: When `topk_positive_tokens` include SAE-specific encodings (e.g., `▁$=$`, `()).`, `▁\n`) and hypotheses repeat those raw forms instead of translating them into human-readable descriptions of visual appearance + functional role.  
**What to change**: Mandate conversion of SAE tokens into descriptive, dual-format explanations: *“whitespace-prefixed '='”*, *“closing parenthesis followed by period”*, *“newline character ‘\n’ with leading space”* — always pairing glyph + meaning.  
**Why**: Raw SAE representations (especially with `▁`, escapes, or concatenations) are opaque to the matcher’s linguistic heuristics; normalized descriptions bridge the gap between feature activation and interpretable surface form.


<!-- auto-distilled from 45 cases -->
## [Principle: Enforce Strict Token Enumeration Syntax]
**When to apply**: When output_score < 0.5 and `topk_positive_tokens` contain heterogeneous or visually distinctive tokens (e.g., with `▁`, mixed casing, multilingual, or punctuation-adjacent forms), *and* prior hypotheses used free-form descriptive phrasing (e.g., “like X, Y, and Z” or “such as…”) instead of rigid syntactic templates.  
**What to change**: Mandate a fixed, matcher-optimized output syntax: `'The exact tokens are: [t1], [t2], [t3]'` (bracketed, comma+space–separated, no trailing punctuation, no explanatory clauses). Require *verbatim inclusion* of all leading whitespace markers (`▁`), Unicode, and casing — and prohibit any text before or after the bracketed list.  
**Why**: The token-matcher relies on precise lexical anchoring; variable phrasing (e.g., “tokens like…”, “including…”, “such as…”) introduces syntactic noise that breaks surface-form alignment — while rigid templating ensures deterministic, parseable hypothesis structure.

## [Principle: Prioritize Token Identity Over Shared Morphology]
**When to apply**: When hypotheses describe tokens via substring patterns (“tokens starting with ‘Char’”), lexical categories (“indefinite pronouns”), or semantic roles (“adjectives in service descriptions”) — *and* `topk_positive_tokens` include orthographically distinct variants (e.g., `Char`, `char`, `Charlene`, `Charlottesville`) or cross-lingual cognates (e.g., `fame`, `Fame`, `honneur`, `fama`, `Shame`).  
**What to change**: Replace morphological or categorical generalizations with explicit enumeration of *all observed surface forms*, preserving case, diacritics, and spacing. If grouping is unavoidable, anchor it *only* to shared surface identity (e.g., “case-varied forms of ‘fame’: Fame, fame, honneur, fama, Shame”) — never to inferred roots or abstractions.  
**Why**: The matcher compares literal string matches, not lemmatized or normalized forms; substring- or category-based hypotheses fail because they omit actual evidence tokens or misrepresent their surface diversity — enumeration guarantees coverage and avoids false negatives from overgeneralization.

## [Principle: Require Cross-Lingual Token Transparency]
**When to apply**: When `topk_positive_tokens` include non-English or mixed-script tokens (e.g., `Externé`, `发表于`, `progettazione`, `biztons`) *and* hypotheses either omit them, translate them without quoting originals, or group them under vague labels (“non-English terms”).  
**What to change**: Force dual-format presentation for each non-English token: *exact token string* + parenthetical human-readable gloss (e.g., `'Externé (Czech for “external”)', '发表于 (Chinese for “published at”)'`). Require *all* such tokens to appear verbatim in the hypothesis — no filtering, no aggregation, no language-label-only summaries.  
**Why**: The matcher treats tokens as opaque byte strings; omitting or paraphrasing non-English tokens severs the link between activation evidence and hypothesis — while verbatim inclusion + gloss satisfies both matcher precision *and* human interpretability requirements.

## [Diagnosis E: Input-Output Domain Mismatch]
**When to apply**: When input hypotheses correctly identify token patterns (e.g., LaTeX commands `\vec`, batch-prefixed terms, surname prefixes `Des`) and achieve activation_rate=1.0, but output hypotheses incorrectly generalize to an unrelated domain (e.g., HTML/XML tags when input is about academic/math tokens) and `support_ratio` < 0.4.
**What to change**: Set `extra_output_guidance` to:
```
Output hypothesis MUST describe ONLY the EXACT token surface forms observed in input activations (e.g., 'vec', 'batch', 'Des'), including casing and punctuation. DO NOT generalize to programming/markup concepts when input tokens are academic/technical terms. Cite specific examples from observation.input_top_activations.
```
**Why**: Output hypotheses should *describe the same tokens* that the input hypothesis targeted — not invent a new domain. When they drift to a different domain, the token-matcher's signal collapses because it's looking for the wrong things.

## [Diagnosis F: Avoid Framework-Specific Output Hypotheses]
**When to apply**: When output hypothesis names a specific technology framework (e.g., “Flutter/Dart widget classes”, “Java MXBean interfaces”) but `topk_positive_tokens` actually span multiple frameworks or domains (e.g., Flutter `ArrowToggle`, .NET `GetEnumerator`, generic `IGraphics`). Score < 0.4 despite passing Gate 1.
**What to change**: Set `extra_output_guidance` to:
```
Describe ONLY concrete token surface forms observed in topk_positive_tokens. DO NOT name specific frameworks (Flutter/Dart, Spring, .NET) unless ALL top tokens belong to that framework. If tokens span multiple domains, describe the common surface pattern (e.g., 'PascalCase identifiers with Toggle/Mode/Constraint suffixes') instead of framework names.
```
**Why**: Framework-specific hypotheses exclude valid tokens from other frameworks and over-restrict the matcher scope. Cross-domain token sets need surface-form description (case, suffix patterns), not domain attribution.

## [Principle: Focus Output Hypothesis on One Pattern at a Time]
**When to apply**: When output hypothesis attempts to cover multiple distinct input token patterns simultaneously (e.g., combining 'bi' prefixes, 'agents', and 'matter' into one vague hypothesis) and best output score < 0.3.
**What to change**: Set `extra_output_guidance` to:
```
Output hypothesis MUST focus on ONE specific token pattern from input activations. Identify the MOST FREQUENT token surface form (e.g., 'bi' prefix appearing in 3/5 top tokens) and describe ONLY that pattern with exact casing and context. Discard hypotheses that cover multiple unrelated token types.
```
**Why**: The token-matcher is designed to evaluate one focused prediction — broad hypotheses covering multiple unrelated patterns produce scores near zero because they're too vague to match any one cluster of evidence tokens.
