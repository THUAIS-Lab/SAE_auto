# Skill: Input Side (Gate 1)

Reference for diagnosing and fixing input-side failures.
Triggered when input_round.eval.overall_activation_rate < 0.8 OR input_round.eval.overall_boundary_non_activation_rate < 0.8.

## Key Files to Read
- round_1/*-observation-input.json -> raw observation tokens and examples
- round_2/*-step2-input-hypotheses.json -> input hypotheses
- round_3/*-step3-input-experiments.json -> designed activation/boundary sentences
- round_4/*-step4-input-experiment-scores.json -> per-sentence activation results (most important)

---

## STEP 0 — Identify Feature Type Before Choosing Fix Strategy

**This step is mandatory before applying any fix.** The correct strategy depends entirely on whether the feature is lexically-specific or semantic. Applying the wrong strategy causes regression.

Look at the Neuronpedia top activation tokens in round_1/*-observation-input.json (or trace.json -> input_round.observation.input_top_activations):

| Feature Type | Token Pattern | Example | Correct Strategy |
|---|---|---|---|
| **Lexically-specific** | Top tokens are the **same word or stem** (homogeneous) | all `▁settle*`, all `▁convict*` | → Diagnosis A: orthographic narrowing |
| **Semantic** | Top tokens are **diverse but thematically related** | `▁successful`, `▁production`, `▁confidence`, `▁hooked` | → Diagnosis C: semantic hypothesis repair |

**Critical rule:** Never apply orthographic narrowing (Diagnosis A) to a semantic feature. It will degrade working hypotheses and lower the overall activation rate.

Evidence from layer-12 smoke test (2026-05-01):
- Feature-400 (lexical): top tokens all `▁settle*` → orthographic fix raised H1 from 0.20 → 1.00
- Feature-8000 (semantic): top tokens diverse positive-outcome words → orthographic fix (suffix -ful/-tion/-ence) dropped H1 from 0.80 → 0.20 by round 3

---

## Diagnosis A — Low Activation Rate (Lexically-Specific Features Only)

**Prerequisite:** Confirmed via Step 0 that top activation tokens are homogeneous (same root).

Cause: The LLM generates semantic/conceptual hypotheses, but the feature only fires on specific token surface forms. Paraphrases and synonyms do not activate it.

Fixes (in order):
1. Add `extra_input_guidance` to force orthographic specificity:
   - "The top Neuronpedia tokens are all variants of the same word stem (e.g., '▁settle', '▁settlement'). Focus hypotheses on the exact token surface form: leading space, capitalization, suffix patterns. Do NOT use synonyms or paraphrases — only sentences containing the exact token form will activate this feature."
2. Regenerate hypotheses (skip_observation_and_design=False) with the above guidance.
3. Design sentences that **embed the exact target token** (not synonyms). E.g., for a `▁settle` feature, use "...they decided to settle..." not "...they chose to take up residence...".

## Diagnosis B — Low Boundary Non-Activation Rate

Cause: hypothesis is over-generalized; boundary sentences that should NOT activate the feature still do.

Fixes (in order):
1. Narrow the hypothesis scope to exclude near-neighbor patterns.
2. Add extra_input_guidance:
   - "Boundary non-activation is low: exclude near-neighbor lexical patterns from hypothesis scope."
   - "The hypothesis must distinguish tokens that share the same root but differ in suffix or spacing."
3. If boundary sentences are poorly designed: regenerate (skip_observation_and_design=False).

---

## Diagnosis C — Low Activation Rate (Semantic Features)

**Prerequisite:** Confirmed via Step 0 that top activation tokens are diverse but thematically related.

Cause: One or more hypotheses cover the wrong semantic territory, OR designed sentences fail to reliably elicit the target semantic class. The problem is hypothesis content, not surface form.

Diagnostic check: read round_4/*-step4-input-experiment-scores.json per-hypothesis results:
- Is **at least one hypothesis already achieving activation_rate >= 0.8**? (e.g., H1=0.80 for feature-8000)
  - YES → The feature IS semantic. Only fix the failing hypotheses. Do NOT touch the passing one.
  - NO → All hypotheses are wrong; regenerate all from scratch with semantic guidance.

Fixes for failing hypotheses:
1. Identify the semantic mismatch. Ask: does the failing hypothesis describe what actually activates (check Neuronpedia tokens)? Common mistakes:
   - H covers negative emotions but the feature fires on positive outcomes
   - H is too broad (e.g., "any emotional intensity") vs the feature's actual narrow semantic cluster
   - H describes a plausible but unrelated semantic concept
2. Rewrite failing hypotheses to precisely match the semantic cluster of the Neuronpedia tokens. Use the passing hypothesis as a reference for the correct semantic level.
3. Add `extra_input_guidance`:
   - "This is a semantic feature: top tokens are diverse but thematically united (e.g., positive-outcome words). Fix failing hypotheses by rewriting them to match the actual semantic cluster. Do NOT narrow to orthographic patterns (suffix/prefix)—that will break working hypotheses."
4. Set skip_observation_and_design=False so hypotheses are regenerated.

**Do NOT:**
- Apply suffix/prefix/orthographic constraints to semantic features
- Change a hypothesis that is already achieving activation_rate >= 0.8
- Introduce negative/opposite semantic content just to create contrast

---

## Both Rates Low Simultaneously

Distinguish by feature type first (Step 0):
- Lexically-specific: regenerate with orthographic guidance (Diagnosis A).
- Semantic: regenerate all hypotheses with semantic guidance (Diagnosis C, "all wrong" branch).

---

## extra_input_guidance Templates

| Situation | Feature Type | Guidance string |
|-----------|---|-----------------|
| Low activation, lexical feature | Lexical | "Top Neuronpedia tokens are homogeneous (same stem). Focus on exact token surface form: leading space, capitalization, suffix/prefix. Design sentences embedding the exact token, not synonyms." |
| Low activation, semantic feature, some H passing | Semantic | "This is a semantic feature with diverse but thematically related top tokens. Fix only the failing hypotheses by rewriting their semantic scope. Do NOT apply orthographic constraints—they will break passing hypotheses." |
| Low activation, semantic feature, all H failing | Semantic | "This is a semantic feature. All hypotheses are wrong. Regenerate by identifying the shared semantic theme across the Neuronpedia top tokens and writing three distinct but correct hypotheses about that theme." |
| Boundary leakage | Any | "Boundary non-activation is low: tighten hypothesis scope and exclude near-neighbor lexical patterns." |
| Orthographic fix applied to semantic feature (regression) | Semantic | "Previous rounds applied orthographic constraints which degraded a working hypothesis. Revert to semantic hypothesis design. The feature fires on a semantic class, not a surface form." |

---

## Loop-Break: When Orthographic Fix Causes Regression

If a previous round applied orthographic narrowing AND the overall activation_rate decreased:
- This confirms the feature is semantic, not lexical.
- Set skip_observation_and_design=False.
- Add the "regression revert" extra_input_guidance from the table above.
- Restore the semantic framing of the best-performing original hypothesis as a reference.



<!-- auto-distilled from 106 cases -->
## [Code-Context Exclusivity Principle]  
**When to apply**: When top Neuronpedia tokens appear *only* in syntactically valid, executable code (e.g., `if`, `0x`, `[R33]`) and *never* in natural-language contexts (comments, strings, prose), as confirmed by trace inspection.  
**What to change**: Enforce absolute context boundaries in hypotheses and sentence design: (1) activation sentences must contain the target token *only* as a standalone lexical unit within valid code syntax; (2) boundary sentences must exclude the token *in all contexts*, including strings/comments; (3) explicitly ban semantic generalizations (e.g., “conditional logic”) in favor of structural constraints (e.g., “`if` keyword in JS statement position”).  
**Why**: Code-token features are hypersensitive to syntactic role—not meaning—so semantic framing or lax context handling (e.g., accepting `'if'` in comments) causes false positives/negatives and collapses boundary non-activation.

## [Tokenization-Aware Surface Form Precision]  
**When to apply**: When top tokens are punctuation, single characters, or subword fragments whose activation depends on *tokenization artifacts*: leading spaces (`▁.`), bracketing (`[R33]`), spacing rules (`word. (next)`), or affix adjacency (`takder` vs `tidak`).  
**What to change**: Replace semantic or functional descriptions (e.g., “sentence-final period”) with hypotheses and sentences that reference *exact token surface forms and their immediate tokenization environment*. Require concrete, replicable patterns (e.g., `▁(` not “opening parenthesis”, `word.` followed by space+`(` not “after a word”).  
**Why**: These features fire on tokenizer outputs—not linguistic roles—so describing them semantically (e.g., “marks boundaries”) misaligns hypothesis scope and admits invalid boundary cases.

## [Strict Near-Neighbor Lexical Exclusion]  
**When to apply**: When boundary non-activation is low *and* failure analysis shows leakage from morphologically or orthographically adjacent forms (e.g., `tak` vs `tidak`, `smoker` vs `vaper`, `parts` vs `nested`)—especially when those near-neighbors share roots, prefixes, or common substrings but differ in tokenization or semantic class.  
**What to change**: Explicitly enumerate excluded near-neighbor forms in hypotheses and boundary instructions; require *positive inclusion only for attested Neuronpedia tokens* (e.g., `'▁tak'`, `'smoker'`) and *negative exclusion of all unattested variants*, even plausible ones. Do not rely on semantic distance—use exact string/token matching.  
**Why**: Near-neighbor leakage indicates the feature’s decision boundary is defined at the subword or lexical-token level—not semantic similarity—so conceptual exclusion (“avoid synonyms”) fails where orthographic exclusion (“exclude any token containing ‘tida’”) succeeds.

## [Cross-Category Proper Noun Generalization]  
**When to apply**: When top tokens are diverse capitalized proper nouns spanning *multiple unrelated domains* (e.g., religious `Revelation`, technical `ModelForm`, personal `Jenkins`) and prior hypotheses overfit to one domain (e.g., only religious terms).  
**What to change**: Rewrite hypotheses to unify *surface form* (capitalized proper noun) + *distributional context* (title, institutional name, technical identifier) *without requiring semantic coherence* across categories. Use disjunctive, domain-agnostic phrasing: “appearing in X, Y, or Z contexts” rather than seeking a shared abstract theme.  
**Why**: Capitalization-triggered features often encode orthographic *formatting conventions* across domains—not a single semantic concept—so forcing thematic unity (e.g., “divine authority”) discards valid activations and degrades activation rate.


<!-- auto-distilled from 118 cases -->
## [Semantic Category Boundary Enforcement]
**When to apply**: When activation rate is high but boundary non-activation is low *and* the top tokens belong to a well-defined semantic category (e.g., “indefinite pronouns”, “standard library function categories”, “official labels in taxonomies”) — especially when boundary leakage stems from including *related but out-of-category* terms (e.g., using “calculate” near “prepare/rotate/str” or “Bristol West” near “food groups”) that share surface or contextual similarity but fall outside the feature’s actual semantic scope.  
**What to change**: Reframe boundary design instructions to require *positive exclusion of entire semantic categories*, not just lexical variants: explicitly name the category (“indefinite pronouns”, “taxonomy identifiers”, “stdlib operation verbs”) and mandate that boundary sentences use *only* terms from *disjoint, non-overlapping semantic domains*. Replace vague exclusions (“avoid similar words”) with domain-anchored contrasts (“use tool nouns, not role nouns”; “use geographic descriptors, not administrative labels”).  
**Why**: Semantic features fire on *category membership*, not gradient similarity — so boundary failures arise from fuzzy category boundaries, not lexical ambiguity. Enforcing strict inter-category separation eliminates false positives caused by conceptual proximity.

## [Subword Fragment vs. Full-Token Activation Disambiguation]
**When to apply**: When top tokens include repeated subword fragments (e.g., `'m'`, `'ank'`, `'bl'`) appearing *across multiple distinct full tokens* (e.g., `'a.m.'`, `'commbank'`, `'BLM'`) — and activation/boundary behavior shifts across rounds depending on whether hypotheses reference the fragment alone, the full token, or contextual conditions (e.g., “after ‘bl’”, “in acronyms”). Confirmed when successive rounds alternate between over-constrained (e.g., requiring trailing punctuation) and under-constrained (e.g., ignoring `▁` prefix) hypotheses.  
**What to change**: Anchor hypotheses *exclusively* to one of two mutually exclusive levels — *either* (a) the exact subword string *as a tokenizer output* (e.g., `'m'`, `'ank'`) with no contextual conditions, *or* (b) the full attested token (e.g., `'▁commbank'`, `'▁a.m.'`) with mandatory `▁` and full surface fidelity — and eliminate all hybrid or conditional formulations (e.g., “‘m’ when after ‘a.’”). Validate choice by checking Neuronpedia trace: if the same fragment appears in *unrelated* full tokens (e.g., `'m'` in both `'a.m.'` and `'BLM'`), prefer level (a); if it only appears in one canonical form, prefer level (b).  
**Why**: Mixing levels introduces inconsistent tokenization assumptions — e.g., treating `'m'` as context-free while simultaneously requiring it to follow `'bl'` — which breaks reproducibility and causes oscillating boundary performance.

## [Domain-Agnostic Proper Noun Pattern Generalization]
**When to apply**: When top tokens are capitalized proper nouns spanning *three or more unrelated domains* (e.g., religious `Revelation`, technical `ModelForm`, personal `Jenkins`, geographic `Bristol`, institutional `FCC`) — and prior hypotheses attempt unification via abstract semantics (e.g., “authority”, “structure”, “formality”) or fail by anchoring to a single domain. Distinct from [Cross-Category Proper Noun Generalization] in that *no shared semantic theme exists at all*: capitalization + syntactic position (title, identifier, label) is the *only* invariant.  
**What to change**: Replace thematic unification with *distributional pattern matching*: define hypotheses around *capitalized lexical units occurring in specific syntactic slots* (e.g., “first word in title case”, “immediately after colon in structured metadata”, “standalone token following `class` or `import` in code”) — and require boundary sentences to use either lowercase common nouns *or* capitalized nouns in *non-matching slots* (e.g., “in possessive phrases”, “mid-sentence appositives”). Ban all semantic predicates (“divine”, “technical”, “geographic”).  
**Why**: Capitalization-triggered features encode *orthographic formatting conventions in context*, not meaning — so forcing semantic coherence discards valid activations and induces regression when new domains appear.

## [Tokenizer Artifact Symbol Exclusion]
**When to apply**: When `input_eval.overall_score_non_zero_rate = 0.0` and designed sentences contain `▁`, `<0x01>`, or other tokenizer-specific symbols written as literal text (e.g., `// MIT License: ▁Copyright` or `The equation (▁x + 1) = 0`).
**What to change**: Remove all tokenizer artifacts from sentence text. Write natural text; the tokenizer produces `▁word` automatically when a word follows a space. For code features, use real syntax without `▁` (e.g., write `ComVisible` not `▁ComVisible`). If hypothesis was generated with `▁` in example strings, add to `extra_input_guidance`: “NEVER include tokenizer artifacts like '▁' in sentences. Write natural text only.”
**Why**: `▁` and similar artifacts are internal tokenizer representations. Writing them literally produces malformed tokens or no-op activations — the model never encounters these strings in natural text.

## [Narrow Domain Context Requirement]
**When to apply**: When a common function word (e.g., `to`, `where`, `in`, `after`) shows near-zero activation_rate despite appearing in designed sentences, and Neuronpedia top activations occur *exclusively* in a narrow domain (e.g., mathematical expressions, historical texts, code arguments). Hypotheses fail because they describe general grammatical usage.
**What to change**: Add domain restriction to all hypotheses and `extra_input_guidance`:
```
The feature activates ONLY when [TOKEN] appears in [DOMAIN] contexts (e.g., mathematical expressions, historical atrocity descriptions). Design sentences where [TOKEN] is specifically in [DOMAIN] usage. DO NOT use general linguistic descriptions — the feature ignores cross-domain usage.
```
Set `skip_observation_and_design=False` to regenerate hypotheses with domain constraints.
**Why**: Some features encode a *token-in-context pair*, not the token alone. Neuronpedia evidence pinpoints the domain; ignoring it produces ~0 activation because the model fires on context, not the token form.

## [Code-Syntax Features Require Actual Code Snippets]
**When to apply**: When Neuronpedia top activations are exclusively in raw code (e.g., `` ` ``, `\def`, `Assembly`) and designed activation sentences are natural-language *descriptions* of code rather than actual code. Low activation_rate despite semantic relevance.
**What to change**: Set `skip_observation_and_design=False` and add to `extra_input_guidance`:
```
This feature activates on raw code syntax tokens, not natural language descriptions of code. Design activation sentences as actual code snippets: use Go struct definitions for backtick features, raw TeX macros for \def features, C# attribute lines for Assembly features. DO NOT write explanatory sentences about code — the model only activates when processing actual code tokens.
```
**Why**: Code-syntax features are orthographic (they fire on the token surface form), not semantic. NL descriptions of code never produce the target token and therefore never activate.

## [Multi-Functional Feature Consolidation]
**When to apply**: When Neuronpedia observation tokens are diverse and unrelated (e.g., `DockStyle`, `isempty`, `pygame`), and per-token narrow hypotheses each achieve low activation rate. A single broad hypothesis covering all tokens may succeed where fragmented ones fail.
**What to change**: Instead of creating separate per-token hypotheses, generate one unified hypothesis describing the abstract pattern that connects all tokens. Add to `extra_input_guidance`:
```
Focus on unifying patterns across ALL observation tokens. Do NOT create separate hypotheses for individual tokens — hypothesize a general pattern like 'activates on technical terms in programming contexts' that covers all observed examples.
```
**Why**: Features may activate on *general patterns* (e.g., “code identifiers”, “completion-verb past tense”) that span many distinct tokens. Narrow per-token hypotheses each test a too-small sample that doesn't generalize, causing cascade failure across all hypotheses.
