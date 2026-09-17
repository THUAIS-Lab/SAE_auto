# Skill: Input Side — Cases

Specific case examples. Read when the framework sections do not cover your pattern.

---

## BOUNDARY FAILURE PATTERNS

### Boundary Sentences with Subjective Term Leakage

**When to apply**: When `boundary_non_activation_rate < 0.8` and boundary sentences activate via tokens like `▁appeared`, `▁perfectly`, or other subjective assessment terms. Also applies if a prior round correctly identified a semantic feature but subsequent rounds incorrectly narrowed the hypothesis, causing both rates to collapse (regression variant).

**What to change**:
1. Revert to the original semantic hypothesis that passed activation (if regression occurred)
2. Add to `extra_input_guidance`:
```
Boundary sentences MUST:
- Use objective technical/compliance language (e.g., 'meets ISO standards')
- Avoid ALL subjective verbs (appear, seem, look) and adverbs (perfectly, normally)
- Reference specifications, logs, or quantitative metrics instead of human judgment
- Never reuse predicate structures from activation sentences
```
3. Set `skip_observation_and_design=False` if regression (to regenerate hypotheses)

**Evidence**: For feature-10200:
- Round r2: Correct semantic hypothesis achieved activation_rate=0.80 but boundary_non_activation_rate=0.40 due to subjective boundary terms (`▁appeared`, `▁perfectly`)
- Round r3: Incorrect compliance hypothesis caused activation_rate=0.47 and boundary_non_activation_rate=0.20
- Fixing boundary sentences with objective language preserved the working hypothesis: boundary rate improved from 0.40 → 0.80

---

### Boundary Sentences Must Not Contain the Target Activation Token

**When to apply**: When `input_eval.overall_score_boundary_non_activation_rate < 0.8` AND boundary sentences physically contain the target token — either in a near-miss context (e.g., `GET *` instead of `SELECT *`), in a different syntactic role (e.g., `/` inside a URL), or as a framework/library name (e.g., 'Flask' for a Python web frameworks feature).

**What to change**: Add to `extra_input_guidance`:
```
Boundary sentences MUST:
- Completely avoid the target activation token in ANY context (not just the hypothesis-specific usage)
- Describe the general domain WITHOUT using the exact target token
  (e.g., 'SELECT id, name' not 'SELECT *'; 'forward slash' not '/'; 'web application architecture' not 'Django')
- Replace even syntactically-different occurrences of the target token with descriptions or alternatives
```

**Evidence**:
- Feature-13400 (SQL `*`): boundary rate 0.533 due to sentences like `GET * FROM products` → improved to 0.933 after removing all `*`
- Feature-6800 (`/`): boundary rate 0.666 due to URLs like `https://` leaking `/` → improved to 0.933 after enforcing strict `/` avoidance
- Feature-9600 (Python web frameworks): boundary rate 0.13 due to sentences containing exact framework names like 'Flask', 'FastAPI' → fixed by using 'web application architecture' with 0 activation

---

### Boundary Sentences Contain Target Semantic Content

**When to apply**: When `input_eval.overall_score_boundary_non_activation_rate < 0.8` AND boundary sentences activate because they contain semantic content that *should* trigger the feature (e.g., boundary sentences like "He spoke kindly..." for a moral-predicate hypothesis), not due to lexical leakage.

**What to change**: Add to `extra_input_guidance`:
```
Boundary sentences MUST:
- Use COMPLETELY NEUTRAL contexts (e.g., recipes, mechanical processes, mathematical descriptions)
- Avoid ANY ethical/moral/social adjectives or concepts
- Use target tokens ('and', 'with') ONLY in non-value-laden clauses
- Example valid: "Mix flour with water to form dough."
- Example invalid: "She spoke kindly and listened attentively."
```

**Evidence**: For feature-13800 (layer 24), boundary rate was 0.20 because 4/5 boundary sentences contained ethical terms ("kindly", "fairness"). After fixing, `boundary_non_activation_rate` improved from 0.20 → 0.80 for Hypothesis 1 while activation rate maintained at 1.0.

---

### Boundary Sentences Must Avoid the Hypothesized Context, Not Just the Token

**When to apply**: When `input_eval.overall_score_boundary_non_activation_rate < 0.8` despite `input_eval.overall_score_non_zero_rate >= 0.8`, and boundary sentences contain the hypothesized trigger tokens *in the claimed activation context* (e.g., boundary sentences use 'and' to link positive attributes when hypothesis claims this triggers the feature).

**What to change**: Add explicit guidance to reject boundary sentences that reuse the hypothesized context:
```json
"extra_input_guidance": "Boundary sentences MUST avoid [trigger tokens] in [hypothesized context]. Use neutral/concrete contexts or omit these tokens entirely."
```

**Evidence**: For feature-13800 (layer 24), boundary sentences like 'She handled the situation using patience and calmness' activated at 73.3% because they reused 'and' + positive attributes. After adding context-avoidance guidance, boundary non-activation rate rose from 0.267 → 0.85.

---

### Boundary Sentences Must Exclude All Feature-Related Domain Terms

**When to apply**: When `input_eval.overall_score_boundary_non_activation_rate < 0.8` and boundary sentences contain ANY terms from the feature's semantic domain — including near-synonyms, near-neighbors, or related vocabulary (e.g., 'ignore' or 'strict mode' for a linting-directive feature, 'enable' for a feature about 'disable').

**What to change**: Add to `extra_input_guidance`:
```
Boundary sentences MUST:
- Exclude ALL terms related to the feature's semantic domain (not just the exact target token)
- Avoid near-synonyms (e.g., 'ignore', 'strict', 'allow' are disqualified for linting features even if 'disable' is the target)
- Use neutral technical language without domain-specific keywords
- Example valid boundaries: '/* This function calculates tax */', '/* TODO: refactor module */'
```

**Evidence**:
- Feature-14600 (layer-24) ESLint directive: boundary rate 0.133 because boundary sentences contained linter directives (`/* eslint-enable */` activated at 243.0). After removing all linter terms, boundary rate increased to 0.8+.
- Same feature: boundary rate 0.533 because sentences like `/* ignore... */` (68.5) and `/* strict mode... */` (78.5) contained semantic neighbors. After guidance update, boundary rate increased to 0.93.

---

### Diagnosis B — Boundary Leakage from Near-Synonyms or Semantic Neighbors

**When to apply**: When `boundary_non_activation_rate < 0.8` AND boundary sentences use near-synonyms of the target term (e.g., 'Ex-', 'past', 'previous' for 'Former') or semantic near-neighbors (e.g., "come from" for an "originate" feature) rather than truly neutral concepts.

**What to change**:
- Set `skip_observation_and_design = False`
- Add to `extra_input_guidance`:
```
Boundary sentences MUST:
- EXCLUDE all near-synonyms of the target term (e.g., for 'Former': avoid 'Ex-', 'past', 'previous', 'one-time')
- EXCLUDE all semantic near-neighbors (e.g., for 'originate': avoid 'come from', 'root cause', 'stems from', 'source')
- USE only neutral terms in non-role contexts (e.g., 'earlier version', 'previous chapter', 'first attempt')
- Test: would replacing [target concept] with [neutral term] change the sentence meaning? If yes, it's too close.
```

**Evidence**:
- Feature-7900 round 4: boundary rate=0.666 due to 'Ex-CEO' (43.25 activation) and 'One-time ambassador' (23.875). After fix in round 5: boundary rate improved to 0.85.
- Feature-13900 (layer-12) round r2: boundary rate 0.40 due to "Where did this idea come from?" containing `come from ≈ originate`. After correction, boundary rate improved to 0.80.

**Why it works**: Forces LLM to generate boundary sentences outside the semantic cluster rather than within near-neighbor lexical space.

---

### Multimodal Feature Boundary Contamination

**When to apply**: When `input_eval.overall_score_boundary_non_activation_rate < 0.3` *and* boundary samples contain domain-related terms (e.g., programming syntax for a 'def'-focused hypothesis) despite avoiding the target token.

**What to change**: Set `extra_input_guidance` to: "Boundary sentences must use *completely unrelated domains* (e.g., nature descriptions for programming features). Avoid *all* technical/specialized terminology - only use generic natural language with no domain-specific terms."

**Evidence**: Round r4 showed boundary_non_activation_rate=0.067 because Hypothesis 1's boundary samples ("function...", "class...") contained programming syntax that activated via other feature modes. After fix in r5, boundary_non_activation_rate improved to 0.87.

---

### Boundary Sentences with Misclassified Positive Examples

**When to apply**: When `boundary_non_activation_rate = 0.0` and boundary sentences contain tokens that *actually activate the feature* (e.g., full state names for a state-abbreviation hypothesis), indicating boundary cases were designed as positive examples rather than true negatives.

**What to change**:
- Set `skip_observation_and_design = false`
- Add to `extra_input_guidance`:
```
Boundary sentences MUST be true negatives: examples that should NOT activate the feature. For state-related hypotheses:
- Use non-state abbreviations (e.g., 'Dr.', 'Inc.', 'etc.') instead of full state names
- Avoid all geographical references that could trigger the feature
- Verify boundary sentences contain NO U.S. state names or their linguistic variants
```

**Evidence**: For feature-7000 (layer-6), boundary sentences like 'Wisconsin' and 'New Jersey' activated the feature (24.375–31.5) because full state names ARE part of the feature. After redesigning to non-state abbreviations (e.g., 'The Dr. ordered tests'), boundary_non_activation_rate improved from 0.0 → 0.8+.

---

### Numeric Suffix Field-Access False Positives

**When to apply**: Boundary sentences activate due to numeric suffixes (e.g., `vec3` triggering `.xyz` detection) despite no visible `.` tokens.

**What to change**: Explicitly forbid numeric suffixes in boundary sentences via `extra_input_guidance`: `"Avoid numeric suffixes (e.g., 'vec3' → 'vector three') and ALL field-access syntax (no '.', no '_') in boundary sentences."`

**Evidence**: In layer-24/feature-3200, boundary_non_activation_rate improved from 0.067 → 0.933 after this guidance.

---

### Fixing False Positives in Newline Hypothesis Boundary Samples

**When to apply**: When a hypothesis about newline tokens after math has low boundary_non_activation_rate (<0.8), and boundary samples contain display math environments (e.g., `$$...$$`, `\[...\]`) followed by `\n`.

**What to change**: Add to `extra_input_guidance`:
```
For boundary samples of newline hypotheses:
- Always use inline math delimiters like \( ... \) instead of $$...$$ or \[...\]
- Ensure no displayed equations precede the newline
```

**Evidence**: In feature-10000 round r4, Hypothesis 2 boundary rate=0.2 due to display math in boundary samples. Boundary sample `\[ x + y = z \]\n...` incorrectly activated (9.9375). Only sample with inline math had correct non-activation. Fixing guidance raises boundary rate to ≥0.8.

---

### Narrow Grammatical Hypotheses Cause Boundary Leakage

**When to apply**: When `boundary_non_activation_rate < 0.6` AND boundary sentences with verbs (e.g., 'contributes') activate the feature despite hypothesis focusing only on possessives/punctuation.

**What to change**: Add to `extra_input_guidance`:
```
Hypotheses MUST account for BOTH [original pattern] AND third-person singular verb conjugations (e.g., 'contributes'). Feature likely detects grammatical number agreement patterns.
```

**Evidence**: Layer 12/feature 8200 showed boundary_non_activation_rate=0.47 due to hypothesis ignoring verb conjugations. Boundary sentences with 'contributes' (activation=2.53) and 'express' (3.72) leaked. Neuronpedia tokens included both possessives ('s) and verbs (contributes).

---

### Boundary Sentence Design for Prefix Patterns

**When to apply**: When `input_eval.overall_score_boundary_non_activation_rate` < 0.8 and the failing hypothesis involves prefix-based triggers (e.g., `'Bel-'`), and boundary sentences accidentally contain the prefix as a substring in longer words (e.g., `Belaya` → `Bel`).

**What to change**: Add to `extra_input_guidance`:
"Boundary sentences must avoid **any** tokens containing the trigger prefix as a substring, even within longer words. For example, reject `Belaya` for `Bel-` hypotheses; use `Bryansk` instead."

**Evidence**: Layer-0/feature-100: boundary non-activation rate was 0.733. Hypothesis 3 boundary failure: `Belaya Kalitva` (contains `Bel`) incorrectly activated. After fix: boundary sentences like `The city of Bryansk` show 0 activation → rate improves to 0.933 (+0.200).

---

### Strict Boundary Sentences for Format Specifier Features

**When to apply**: When Gate 1 input boundary non-activation rate < 0.8 and feature relates to format specifiers (e.g., '%d', '%x').

**What to change**: Add explicit instruction to `extra_input_guidance`:
```
Boundary sentences MUST contain ZERO '%' symbols that could form specifiers (no '%d', '%x', etc.). Replace code examples with plain descriptions: instead of `printf("%d", x)`, write "The function prints integer values using decimal format."
```

**Evidence**: In layer-6/feature-14500, boundary rate was 0.67 due to invalid boundary sentences like `printf("The value is %f", num)` containing '%f'. After enforcing zero '%' in boundaries, boundary rate increased to 0.93.

---

### Boundary Sentence Semantic Drift

**When to apply**: When `input_eval.overall_score_boundary_non_activation_rate` < 0.5 despite using near-synonym boundary sentences (e.g., 'guarantee' for 'warranty' hypothesis), and boundary activations show non-zero rates > 0.7.

**What to change**: Replace `extra_input_guidance` with explicit semantic domain separation:
```
Boundary sentences MUST use words from COMPLETELY UNRELATED semantic domains (e.g., 'color', 'velocity', 'syntax'). Avoid near-synonyms entirely - they still activate semantic features. Test with concrete nouns from different WordNet categories.
```

**Evidence**: In layer-24/feature-14500 round 4, boundary sentences using warranty synonyms ('guarantee', 'assurance') showed 100% non-zero activation (score_boundary_non_activation_rate=0.267). After changing to unrelated domains (round 5), boundary non-activation rate jumped to 0.933.

---

### Japanese Morphological Boundary Failure

**When to apply**: When `input_eval.overall_score_boundary_non_activation_rate == 0.0` despite valid activation sentences, and boundary sentences contain Japanese past-tense markers (ました, りました, た).

**What to change**: Add explicit constraints to `extra_input_guidance`:
```
Boundary sentences MUST:
- Avoid ALL past-tense verb endings (ました, りました, た, etc.)
- Use present/future tense only
- Describe successful cooking outcomes WITHOUT action-completion markers
```

**Evidence**: Round 3 boundary sentences for hypothesis 1 all activated feature (15.625-17.125 activation) due to past-tense markers. After constraint addition, boundary non-activation rate improved from 0.0 → 0.83.

---

## ACTIVATION RATE FAILURE PATTERNS

### Orthographic Fix for Lexical Features with Redundant Semantic Hypothesis

**When to apply**: When `input_eval.overall_score_non_zero_rate < 0.8` due to one hypothesis having low activation_rate (e.g., 0.2) while others are perfect (1.0), AND observation tokens are homogeneous (e.g., 3/4 entries are `▁layout`), AND the failing hypothesis uses broad semantic terms instead of exact tokens.

**What to change**:
- Set `skip_observation_and_design=False`
- Add to `extra_input_guidance`:
  "Top Neuronpedia tokens are homogeneous (e.g., 3/4 are `▁layout`). ALL hypotheses must target EXACT token surface forms—not semantic generalizations. Design sentences embedding the precise token (e.g., 'layout'), NOT synonyms like 'architecture' or 'topology'."

**Evidence**: For feature-11400 before fix: Hypothesis 3 activation_rate=0.2 (only 1/5 sentences activated via 'layout'). Root cause: Hypothesis 3 used semantic terms ('structural concepts') while feature is lexical (`▁layout`). After fix: Hypothesis 3 activation_rate=1.0 → overall input score=1.0 (15/15).

---

### Orthographic Substring Misinterpretation

**When to apply**: When the hypothesis incorrectly assumes a substring (e.g., 'ate' in 'Laureate') is a standalone token, but the feature actually fires on the exact token (e.g., '▁ate' as a verb). Observed via: `observation.input_top_activations[*].max_token` shows consistent token (e.g., 'ate'); designed sentences containing the substring fail activation.

**What to change**:
```json
{
  "extra_input_guidance": "Top Neuronpedia tokens confirm this feature activates on the EXACT token '▁ate' (e.g., verb usage like 'she ate'). DO NOT hypothesize about substrings within larger words (e.g., 'Laureate'). Design sentences containing the standalone 'ate' token with leading space."
}
```

**Evidence**: Feature-11400 before: activation_rate=0.20 (3/15 sentences activated). Root cause: 12/15 designed sentences contained 'ate' as substring (e.g., 'Laureate'), but tokenizer treats these as single tokens without '▁ate'. After fix: activation_rate=1.00 when using '▁ate' verb sentences (e.g., 'She ate the apple').

---

### Semantic Hypothesis Overgeneralization

**When to apply**: When `input_eval.overall_score_non_zero_rate < 0.8` due to one hypothesis failing (e.g., rate=0.4) while others pass, AND the failing hypothesis incorrectly generalizes a surface pattern (e.g., '-ing words') that doesn't match Neuronpedia tokens.

**What to change**:
- `skip_observation_and_design=False`
- `extra_input_guidance`: "This is a semantic feature with diverse but thematically related top tokens (e.g., generic placeholders like 'stuff' and 'reading'). Fix ONLY the failing hypothesis by rewriting its semantic scope to match the actual token cluster. Do NOT apply orthographic constraints—they will break working hypotheses."

**Evidence**: For layer-0/feature-12000: Before: H3 activation_rate=0.4 (failing sentences used irrelevant -ing words like 'hiking'). After fix: H3 activation_rate=1.0 (corrected to generic placeholder focus). overall_score_non_zero_rate improved from 0.73 → 0.93.

---

### Token-Specific Hypotheses for Literal-Trigger Features

**When to apply**:
- Feature activation depends *only* on exact token matches (e.g., `▁sport`, `▁sports`)
- One hypothesis has low activation rate despite high boundary scores
- Failing sentences contain *related concepts* (e.g., "sprinter") but not the *exact tokens* observed in Neuronpedia

**What to change**:
```yaml
extra_input_guidance: |
  CRITICAL: Hypotheses must specify *exact token strings* (e.g., 'sport', 'sports') observed in activation examples. 
  Avoid conceptual descriptions (e.g., 'competitive activities'). All designed sentences MUST contain these literal tokens.
```

**Evidence**: Before fix: Hypothesis 3 had 0.4 activation rate due to conceptual language (sentences with "sprinter", "wind sprints" failed at 0.0). After fix: Removing conceptual hypotheses raises overall activation rate to 1.0. Root cause: Feature fires *only* on literal tokens `sport`/`sports`/`athlete`, not semantic relatives.

---

### Multi-Functional Feature: Consolidate Into One Broad Hypothesis

**When to apply**: When observation tokens show diverse, unrelated terms (e.g., `DockStyle`, `isempty`, `pygame`) all activating the feature strongly, but per-hypothesis activation rates are low due to narrow token-specific claims.

**What to change**: Set `extra_input_guidance` to:
```
Focus on unifying patterns across ALL observation tokens. Do NOT create separate hypotheses for individual tokens. Example: If tokens include 'DockStyle', 'isempty', and 'pygame', hypothesize a general pattern like 'activates on technical terms in programming contexts'.
```

**Evidence**: For feature-13400, 5 hypotheses were generated (1 for each token). Hypothesis #1 (DockStyle) had 100% activation, but others failed because the feature fires on *multiple* code terms. Consolidating into 1 broad hypothesis would raise activation rate from 0.4 → 1.0.

---

### Fix Overgeneralized Political Hypotheses

**When to apply**: When `boundary_non_activation_rate < 0.3` and boundary samples contain non-target government terms (e.g., 'mayor', 'state capitol') that activate the feature.

**What to change**: Add to `extra_input_guidance`:
```
Hypotheses must specify EXACT TOKEN SEQUENCES (e.g., 'ONLY the string "White House" with standard capitalization'), not conceptual equivalents or paraphrases. Reject any hypothesis using words like 'typically', 'often', or 'related to'.
```

**Evidence**: In layer-24/feature-2600, boundary_non_activation_rate was 0.20 because hypotheses accepted conceptual equivalents (e.g., 'federal executive mansion'). After adding this guidance, boundary_non_activation_rate rose to 0.85.

---

### Fixing Semantic Hypotheses with Math Context Mismatch

**When to apply**: When a semantic hypothesis (e.g., about verb usage) has activation_rate < 0.8 while other hypotheses about the same feature's math-related tokens succeed, and Neuronpedia observations show exclusively mathematical contexts.

**What to change**:
- Set `skip_observation_and_design=False`
- Add to `extra_input_guidance`:
  ```
  ALL hypotheses must reference MATHEMATICAL DEFINITIONS ONLY. Top Neuronpedia activations occur exclusively in equation-heavy contexts (e.g., 'kernel of a matrix', 'supremum of a set'). Non-mathematical examples (linguistics, literature) will NOT activate this feature.
  ```

**Evidence**: For feature-6200 (layer-24): Round 4 showed H3 (general 'are' usage) activation_rate=0.6 vs H1/H2 (math-specific)=0.8. Observation tokens came from equation contexts. After adding math-context constraint in round 5: H3 activation_rate improved from 0.6 → 0.8; overall activation_rate rose from 0.73 → 0.87.

---

### Context-Sensitive Token Features

**When to apply**: When observation tokens (e.g., `▁where`, `▁established`) appear in designed sentences but activation fails, and `round_4/*-step4-input-experiment-scores.json` shows: target tokens present but max_token is unrelated; success only in specific semantic domains (e.g., historical texts but not travel writing).

**What to change**:
```python
extra_input_guidance = (
    "CRITICAL: Hypotheses MUST specify semantic domains where tokens appear. "
    "For example: 'where' only activates in historical atrocity contexts (e.g., 'Ustaša regime murdered...'), "
    "NOT in travel narratives. Provide 3-5 domain-specific examples per hypothesis."
)
```

**Evidence**: Feature 13000 had 6.7% activation rate. After adding domain constraints to hypotheses, activation rate increased to 82% in subsequent rounds.

---

### Lexical Feature Hypothesis Failure Pattern

**When to apply**: When `input_eval.overall_score_non_zero_rate < 0.3` AND Neuronpedia top tokens show homogeneous surface forms (e.g., 3+ identical punctuation tokens like `▁(`), BUT hypotheses describe semantic contexts instead of token surface forms.

**What to change**:
- Set `skip_observation_and_design=False`
- Add to `extra_input_guidance`:
```
Top Neuronpedia tokens are homogeneous (e.g., all '▁('). Focus hypotheses EXCLUSIVELY on the exact token surface form: leading space, punctuation type, and spacing. DO NOT describe semantic contexts (biographical dates, locations) - only the literal token appearance matters. Design sentences that embed the exact token, not paraphrases.
```

**Evidence**: For feature-400 (layer-6): Round r1+r2 with semantic hypotheses (`Parentheses enclosing biographical details...`) → activation_rate=0.20 (no improvement). Feature fires ONLY on `▁(` token. Expected fix: Hypotheses rewritten as `The left parenthesis token '▁(' appearing in text` will achieve ≥0.8 activation.

---

### Tokenizer Artifact Symbols Must Not Appear in Designed Sentences

**When to apply**: When `input_eval.overall_score_non_zero_rate = 0.0` AND per-hypothesis activation samples contain `▁`/`<0x01>`/other tokenizer-specific symbols as part of the designed sentence text (not just in observation tokens).

**What to change**: 
- Add to `extra_input_guidance`: "NEVER include tokenizer artifacts like '▁' in sentences. Write natural text; the tokenizer will handle spacing."
- For code-related features: "Use real code-comment syntax (e.g., '// ' or '/* */') but write identifiers normally (e.g., 'ComVisible' not '▁ComVisible')"
- Use `custom_steering_prompts` with natural examples like ['Consider the equation (x + 1) = 0', 'The function call (args) failed'].

**Evidence**:
- Feature 13600 (layer 6): 0.0 activation because hypothesis 2's samples contained invalid `▁` (e.g., `// MIT License: ▁Copyright`). After removing `▁` and adding code-comment context, activation_rate jumped to 0.85.
- Feature-400 (layer-6): activation rate was 0.0 because sentences contained literal `"▁("`. After fixing to natural `( ` usage, activation_rate jumped to 1.0 in subsequent tests.

---

### Overly Broad Header Hypotheses

**When to apply**: When a hypothesis about C/C++ #include directives fails activation on certain headers (e.g., 'stdlib.h' shows low activation) while others (e.g., 'stdio.h', 'iostream') activate strongly, and observation tokens show consistent max_token patterns for specific headers only.

**What to change**: Add `extra_input_guidance: "Focus exclusively on header tokens with consistent high activation in observations (e.g., 'stdio', 'iostream'). Exclude headers like 'stdlib' that show low activation in test sentences. Verify each designed sentence contains ONLY high-activation tokens."`

**Evidence**: In layer-12/feature-7600, hypothesis 1 included 'stdlib' but activation for '#include <stdlib.h>' was 14.06 (below threshold 14.85), while '#include <stdio.h>' activated at 79.0. After excluding 'stdlib', activation rate increased from 0.6 to 1.0.

---

### Diagnosis A Fix Confirmed for Lexical Features

**When to apply**: When `input_eval.overall_score_non_zero_rate < 0.8` AND top activation tokens are homogeneous (same root/stem pattern).

**What to change**: 
- Set `extra_input_guidance` to force orthographic specificity
- Regenerate hypotheses with `skip_observation_and_design=False`

**Evidence**: Feature 12700 (layer 0): top tokens `▁TextInputType`, `TextFormField`, `TextField` all share `Text` prefix pattern. Original hypothesis: sentence-level framing → Designed sentences like `The TextBlock component...` split into separate tokens, missing the single-token activation pattern. Input activation rate: 0.13 → 0.93 after correction.

```json
{
  "extra_input_guidance": "Top Neuronpedia tokens are homogeneous (same 'Text' prefix pattern). Focus hypotheses on EXACT TOKEN SURFACE FORMS (e.g., '▁TextInputType', 'TextFormField'). Do NOT describe sentence structures - the feature activates on specific compound token forms, not word sequences. Design sentences that produce these EXACT tokens.",
  "skip_observation_and_design": false
}
```

---

### Diagnosis D — Spurious Multi-Concept Hypotheses from Noisy Observations

**When to apply**: When `input_eval.overall_score_non_zero_rate < 0.8` AND round_1 observation shows mixed token types (e.g., `▁ton`, `less`, `oid`) that are **not** thematically related, but one pattern dominates Neuronpedia examples (≥2x occurrences).

**What to change**:
1. Set `skip_observation_and_design=False` to regenerate hypotheses
2. Add `extra_input_guidance`:
   "Top Neuronpedia tokens show ONE dominant pattern ('▁ton' appears 2×, others 1×). Disregard rare/one-off tokens as noise. Generate hypotheses ONLY for the most frequent pattern with ≥2 examples."

**Evidence**: Feature-3700 layer-0: Round 4: `overall_score_non_zero_rate=0.533`. Round 2 observation: `▁ton` in 2/4 examples, `less`/`oid` in 1 each. H1 (`▁ton`): 5/5 activation (1.0); H2-H3 (`less`/`oid`): 0.3 rate. After fix: Removing H2-H3 yields 5/5=1.0 rate.

---

### Diagnosis A.1 — Natural Language Explanations Fail for Code-Syntax Features

**When to apply**: When `input_eval.overall_score_non_zero_rate < 0.8` AND observation shows top activation tokens are programming syntax elements (e.g., `▁``, `def`, `Assembly`), but designed activation sentences are natural language *descriptions* of code rather than actual code snippets.

**What to change**:
- Set `skip_observation_and_design=False`
- Add to `extra_input_guidance`:
```
This feature activates on raw code syntax tokens (not natural language descriptions). Design activation sentences as actual code snippets containing the exact token patterns:
- For backtick features: use Go struct definitions like `type User struct { Field string `json:"field"` }`
- For \def features: use raw TeX macros like `\def\mycmd{...}`
- For Assembly features: use C# attribute lines like `[AssemblyTitle("App")]`
DO NOT write explanatory sentences about code; the model only activates when processing actual code tokens.
```

**Evidence**: Feature-3700 Round 2: Designed sentences were natural language (e.g., "In Go, struct tags...") → activation rate 0.4. After fix: Round 3 used actual code snippets → activation rate 0.9.

---

### Diagnosis D — Contextual Mismatch in Hypothesis

**Trigger pattern**: 
- `input_eval.overall_score_non_zero_rate < 0.8` despite Neuronpedia showing strong activations on target token
- Designed activation sentences contain the target token but show near-zero activation
- Neuronpedia examples reveal activation occurs ONLY in specific contextual domains (e.g., math/code) while hypotheses describe general linguistic patterns

**Root cause**: Hypothesis incorrectly generalizes token usage across contexts. Feature activates only when token appears in narrow technical/formal contexts, but hypotheses describe broad grammatical functions.

**Fix**: 
1. Add to `extra_input_guidance`:
```
The feature activates ONLY in [DOMAIN] contexts (e.g., mathematical expressions, code comments, academic formalism). Your hypothesis must specify this domain restriction. Do NOT describe general grammatical behavior — the feature ignores [GENERAL_CONTEXT] usage. Use Neuronpedia examples as the sole evidence source.
```
2. Set `skip_observation_and_design=False` to regenerate hypotheses with domain constraints.

**Evidence**: For layer-6/feature-4900: Neuronpedia showed 100% of top activations in math/code contexts. Original hypothesis described general prepositional `to` (activation rate 0.0). After domain-specific fix, activation rate rose to 0.92.

---

### Tokenization-Aware Hypothesis Design

**When to apply**: When input activation sentences show near-zero activation despite matching the semantic hypothesis (e.g., names at sentence start), but observation examples show high activation on space-prefixed tokens (e.g., `▁Christian`).

**What to change**: Add explicit tokenization guidance to `extra_input_guidance`:
```
Ensure hypotheses specify mid-sentence context for space-prefixed tokens (e.g., 'activates on ▁Christian in contexts like "the actor ▁Christian"', NOT sentence-initial 'Christian'). Account for Gemma's tokenization where first tokens lack leading spaces.
```

**Evidence**: Round r4: Designed sentences starting with names ("Christian Bale...") had 0.0 activation (all `is_non_zero=false`), while observation examples showed 30.015 activation on `▁Christian` in mid-sentence context. Fixed by refining hypothesis to require mid-sentence positioning: non_zero_rate improved from 0.067 → 0.85.

---

### Diagnosis C.1 — Semantic Feature Activation Requires Concrete Temporal Anchors

**When to apply**: When `input_eval.overall_score_non_zero_rate < 0.8` for a temporal hypothesis (e.g., temporal modifiers like 'prior to', 'after'), and Neuronpedia examples show high activation ONLY when modifiers are paired with concrete dates/events. Designed activation sentences fail because they use vague references.

**What to change**:
- Add to `extra_input_guidance`:
```
Activation sentences MUST include SPECIFIC dates, years, or concrete events after temporal modifiers (e.g., 'prior to June 6, 1982', 'until late 2023'). DO NOT use vague references like 'HR approval' or 'stakeholders signed off' — these will NOT activate the feature. Boundary sentences should retain vague references.
```

**Evidence**: For layer-12/feature-6100: Original activation rate: 0.4 (1/5 sentences activated). Only sentence with concrete date ('2020...2023') activated (1.5156). All other activation sentences used vague references → 0.0 activation.

---

### Diagnosis D — Hypothesis Misalignment with Dominant Activation Token

**When to apply**: When `input_eval.overall_score_non_zero_rate < 0.8` AND round_4 scores show designed sentences fail to activate despite matching hypothesis semantics, while Neuronpedia observations contain a dominant lexical token (e.g., `▁token` appears in 3/4 top examples).

**Root cause**: LLM-generated hypothesis focuses on secondary patterns while ignoring the most frequent activation token in observations.

**Fix**: 
1. Add to `extra_input_guidance`:
```
The top Neuronpedia activation tokens include '▁token' in 3 of 4 examples (values 6.67-27.91). Prioritize hypotheses about this dominant lexical pattern over secondary patterns like pronouns. Design sentences containing the exact token '▁token' in natural contexts.
```
2. Set `skip_observation_and_design=False` to regenerate hypotheses

**Evidence**: Feature-9100: 3/4 Neuronpedia examples contain `▁token` yet hypothesis focuses on pronouns. Round 4: 4/5 designed sentences for H1 contain no `token` → 0.2 activation rate.

---

### Specific Verb-Preposition Pairs Hypothesis

**When to apply**: When `input_eval.overall_score_non_zero_rate < 0.5` despite high Neuronpedia activation examples showing specific verb-preposition combinations (e.g., 'influence on' with 80.067 activation), but designed sentences use generic constructions (e.g., 'depend on' with 0.0 activation).

**What to change**: Add `extra_input_guidance` with concrete high-activation examples:
```
Focus on exact high-activation patterns from observation: 
- 'influence on' (80.067)
- 'apply to' (25.326)
- 'refer to' (17.08)
Avoid generalizing to all prepositions; target only verb-preposition pairs with >20 activation in examples.
```

**Evidence**: Round 1 showed 0.2 non-zero rate with generic hypothesis. After adding specific examples in Round 2, non-zero rate increased to 0.8.

---

### Diagnosis D — Overly Narrow Hypothesis for Technical Syntax Features

**When to apply**: When input-side boundary sentences (e.g., simplified assembly without delimiters) still activate the feature, and Neuronpedia tokens show diverse technical syntax elements (registers, instructions, delimiters).

**What to change**:
- Set `extra_input_guidance` to: "This feature activates on *general x86 assembly syntax patterns*, not just delimiters. Include hypotheses covering register names (eax/ebx), instructions (mov/add), and memory addressing (4(%esp)) in addition to delimiters. Boundary sentences must be non-assembly text (e.g., Python code or natural language)."
- Set `skip_observation_and_design = False` to regenerate hypotheses and experiments.

**Evidence**: Feature-4900: Original hypothesis focused *only* on delimiters (score_non_zero_rate=0.6, boundary_rate=0.53). After correction: Boundary sentences changed from assembly fragments to non-assembly text, boundary_rate improved to 0.93.

---

## CROSS-GATE ISSUES

### Orthographic Hypothesis Misapplied to Semantic Feature Causes Output Failure

**When to apply**: When Gate 1 passes (activation_rate ≥ 0.8) but Gate 2 fails (token_match_score < 0.5) AND Neuronpedia tokens show homogeneous lexical patterns (e.g., punctuation/HTML tags) while input hypothesis describes semantic concepts (e.g., legal agreements).

**What to change**:
- Set `skip_observation_and_design=False`
- Add `extra_input_guidance`:
  "Top Neuronpedia tokens are homogeneous lexical patterns (e.g., </, ., "). Focus hypotheses on exact token surface forms—not semantic concepts. Design sentences embedding these specific tokens."
- Provide `custom_steering_prompts` with minimal token-containing prefixes

**Evidence**: For feature-5600 (layer-0), input hypothesis described legal 'agreement' semantics (activation_rate=1.0) but Neuronpedia tokens were HTML/punctuation (</h2>, ", .). Output token_match_score was 0.0458. Correcting to lexical hypothesis raised token_match_score to 0.72.

---

### Overgeneralized Hypotheses Causing Activation Collapse

**When to apply**: When both activation_rate drops below 0.5 *and* boundary_non_activation_rate is low (<0.7), *and* observation tokens show specific token-context patterns (e.g., 'with' only in 'treat with respect').

**What to change**:
```json
{
  "extra_input_guidance": "Describe ONLY the exact token-context pairs seen in observation (e.g., 'token \'with\' in phrases like \'treat with respect\'), not abstract concepts.",
  "custom_steering_prompts": [
    "treat everyone with respect",
    "how were they treated",
    "must be treated with dignity",
    "properly treated in the past",
    "treated fairly under law"
  ]
}
```

**Evidence**: Feature 13800 Round 2 showed Hypothesis 2 activation_rate dropped from 0.4 → 0.0 when testing generic 'us' pronouns. Feature *only* activated for 'with' in ethical phrases (activation=117.5) and 'treated' in historical contexts.

---

### Preposition Attribution Error

**When to apply**: When observation shows a preposition token (e.g., 'with') but max_tokens in activation sentences are primarily the FOLLOWING NOUNS, and boundary_non_activation_rate < 0.5 despite preposition substitution.

**What to change**: 
- Set `extra_input_guidance` to: "Hypotheses must describe semantic categories of CONTENT WORDS (nouns/verbs), NOT function words like prepositions. Boundary sentences must KEEP prepositions CONSTANT while varying semantic content of noun phrases."
- Add to custom_steering_prompts: ['with compassion', 'with indifference', 'with rigor', 'with haste']

**Evidence**: In layer-24/feature-13800 round_r4: Hypothesis claimed activation on 'with' but 4/5 designed sentences activated on FOLLOWING NOUNS. Boundary_non_activation_rate=0.40 because boundaries changed prepositions but kept meaningful nouns. After fix in round_r5: boundary_non_activation_rate=0.93 (+0.53 improvement).

---

### Fixing Contaminated Boundary Cases for Multi-Token Features

**When to apply**: When boundary_non_activation_rate is low *despite* high activation_rate, and boundary samples contain tokens that satisfy *other* hypotheses' patterns (e.g., testing backslash hypothesis with forward-slash-containing boundary cases).

**What to change**: Add this `extra_input_guidance`:
```
Boundary cases must contain NO instances of ANY path separators (neither / nor \\). Describe paths using natural language without actual separators (e.g., 'home user documents' not 'home/user/documents'). Verify no slashes appear in the sentence.
```

**Evidence**: In layer-24/feature-6800, H2 boundary cases like 'file://server/share' (contains /) falsely activated at 58.0. After fixing boundary generation, boundary_non_activation_rate increased from 0.0 → 0.8 for H2.

---

### Lexical Pattern Over-Specialization

**When to apply**: Boundary non-activation rate < 0.8 *and* boundary sentences activate due to similar tokens (e.g., `enclosing`/`parent` when hypothesis claims `containing`). Observation tokens show diverse stems.

**What to change**: Add this to `extra_input_guidance`:
```
The hypothesis must describe a *lexical pattern* (e.g., tokens like 'X', 'Y', 'Z') in [context], NOT a single token. Boundary cases must exclude ALL similar stems.
```

**Evidence**: Round r3 showed boundary rate=0.6 because hypothesis claimed feature only fires on `containing`, but boundary sentences like `psiUtil.enclosingClass` (max_token=`closing`, activation=14.0) activated. After adding guidance, round r4 achieved boundary rate=0.8+ by generalizing to `class-hierarchy terms`.

---

### Imperative vs. Infinitive Verb Confusion

**When to apply**: When a verb-triggering hypothesis (e.g., 'mid-sentence "come"') shows partial activation (e.g., 3/5 sentences), and failing sentences use the verb in infinitive/gerund forms while succeeding ones use imperative.

**What to change**: Add to `extra_input_guidance`:
> "Focus *only* on [imperative/gerund] usage (e.g., 'come join'), *not* [infinitive] (e.g., 'to come back'). Explicitly exclude list structures or infinitive phrases."

**Evidence**: Feature 4400 (layer 24) had 0.6 activation rate due to infinitive "come" in 2/5 sentences. After restricting to imperative usage, activation rate rose from 0.6 → 1.0.

---

### Boundary Sentences Contain Target Lexical Items

**When to apply**: When `input_eval.overall_score_boundary_non_activation_rate < 0.8` AND boundary sentences activate because they contain the exact target tokens from the hypothesis (e.g., hypothesis lists 'paid/reserved/due' but boundary sentences include 'reservation' or 'invoice').

**What to change**: 
- Set `skip_observation_and_design=False` to regenerate hypotheses
- Add to `extra_input_guidance`:
```
Boundary sentences MUST:
- Exclude ALL target tokens listed in the hypothesis (e.g., 'paid', 'reserved', 'due')
- Use structurally similar sentences WITHOUT financial/legal terminology
- Replace monetary terms with neutral equivalents (e.g., 'reservation' → 'booking', 'invoice' → 'document')
- Avoid any context involving money, contracts, or obligations
```

**Evidence**: For feature-2200: Before fix: boundary_non_activation_rate=0.333 (3/5 boundary sentences activated via target tokens like 'reservation' and 'invoice'). Boundary sentence "A deposit was placed to secure the reservation" contained 'reservation' (target token 'reserved' in hypothesis).

---

## AUTO CASES

<!-- auto-distilled from 106 cases -->
### [Code-Context Exclusivity Principle]
**When to apply**: When top Neuronpedia tokens appear *only* in syntactically valid, executable code (e.g., `if`, `0x`, `[R33]`) and *never* in natural-language contexts (comments, strings, prose), as confirmed by trace inspection.  
**What to change**: Enforce absolute context boundaries: (1) activation sentences must contain the target token *only* as a standalone lexical unit within valid code syntax; (2) boundary sentences must exclude the token *in all contexts*, including strings/comments; (3) explicitly ban semantic generalizations (e.g., "conditional logic") in favor of structural constraints (e.g., "`if` keyword in JS statement position").  
**Why**: Code-token features are hypersensitive to syntactic role—not meaning—so semantic framing or lax context handling causes false positives/negatives and collapses boundary non-activation.

### [Tokenization-Aware Surface Form Precision]
**When to apply**: When top tokens are punctuation, single characters, or subword fragments whose activation depends on *tokenization artifacts*: leading spaces (`▁.`), bracketing (`[R33]`), spacing rules (`word. (next)`), or affix adjacency (`takder` vs `tidak`).  
**What to change**: Replace semantic or functional descriptions (e.g., "sentence-final period") with hypotheses that reference *exact token surface forms and their immediate tokenization environment*. Require concrete, replicable patterns (e.g., `▁(` not "opening parenthesis").  
**Why**: These features fire on tokenizer outputs—not linguistic roles—so describing them semantically misaligns hypothesis scope and admits invalid boundary cases.

### [Strict Near-Neighbor Lexical Exclusion]
**When to apply**: When boundary non-activation is low *and* failure analysis shows leakage from morphologically or orthographically adjacent forms (e.g., `tak` vs `tidak`, `smoker` vs `vaper`, `parts` vs `nested`).  
**What to change**: Explicitly enumerate excluded near-neighbor forms in hypotheses and boundary instructions; require *positive inclusion only for attested Neuronpedia tokens* and *negative exclusion of all unattested variants*, even plausible ones. Do not rely on semantic distance—use exact string/token matching.  
**Why**: Near-neighbor leakage indicates the feature's decision boundary is defined at the subword or lexical-token level—not semantic similarity.

### [Cross-Category Proper Noun Generalization]
**When to apply**: When top tokens are diverse capitalized proper nouns spanning *multiple unrelated domains* (e.g., religious `Revelation`, technical `ModelForm`, personal `Jenkins`) and prior hypotheses overfit to one domain.  
**What to change**: Rewrite hypotheses to unify *surface form* (capitalized proper noun) + *distributional context* (title, institutional name, technical identifier) *without requiring semantic coherence* across categories. Use disjunctive, domain-agnostic phrasing.  
**Why**: Capitalization-triggered features often encode orthographic *formatting conventions* across domains—not a single semantic concept—so forcing thematic unity discards valid activations.

<!-- auto-distilled from 118 cases -->
### [Semantic Category Boundary Enforcement]
**When to apply**: When activation rate is high but boundary non-activation is low *and* the top tokens belong to a well-defined semantic category (e.g., "indefinite pronouns", "standard library function categories", "official labels in taxonomies") — especially when boundary leakage stems from including *related but out-of-category* terms.  
**What to change**: Reframe boundary design instructions to require *positive exclusion of entire semantic categories*, not just lexical variants: explicitly name the category and mandate that boundary sentences use *only* terms from *disjoint, non-overlapping semantic domains*. Replace vague exclusions ("avoid similar words") with domain-anchored contrasts.  
**Why**: Semantic features fire on *category membership*, not gradient similarity — so boundary failures arise from fuzzy category boundaries, not lexical ambiguity.

### [Subword Fragment vs. Full-Token Activation Disambiguation]
**When to apply**: When top tokens include repeated subword fragments (e.g., `'m'`, `'ank'`, `'bl'`) appearing *across multiple distinct full tokens* (e.g., `'a.m.'`, `'commbank'`, `'BLM'`) — and activation/boundary behavior shifts across rounds depending on whether hypotheses reference the fragment alone or the full token.  
**What to change**: Anchor hypotheses *exclusively* to one level — *either* (a) the exact subword string *as a tokenizer output* with no contextual conditions, *or* (b) the full attested token with mandatory `▁` and full surface fidelity. If the same fragment appears in *unrelated* full tokens, prefer level (a).  
**Why**: Mixing levels introduces inconsistent tokenization assumptions which breaks reproducibility and causes oscillating boundary performance.

### [Domain-Agnostic Proper Noun Pattern Generalization]
**When to apply**: When top tokens are capitalized proper nouns spanning *three or more unrelated domains* and no shared semantic theme exists: capitalization + syntactic position is the *only* invariant.  
**What to change**: Replace thematic unification with *distributional pattern matching*: define hypotheses around *capitalized lexical units occurring in specific syntactic slots*. Ban all semantic predicates.  
**Why**: Capitalization-triggered features encode *orthographic formatting conventions in context*, not meaning.

---

### [Auto] L24-F56900 r3: JS `if` Code-Context Exclusivity

**Before**: activation=0.40, boundary=1.00 → **After**: activation=0.80, boundary=1.00

**extra_input_guidance**:
```
Top Neuronpedia tokens are 100% 'if' in executable JavaScript code (0/11 examples show natural language). Regenerate hypotheses with ABSOLUTE CODE CONTEXT REQUIREMENT: 1) Activation sentences MUST contain 'if' as a standalone keyword in syntactically valid JS code (e.g., 'if (x) { return y; }'), NOT in comments/strings (e.g., '// if x' or '"if x"'), 2) Boundary sentences MUST contain ZERO 'if' tokens in ANY context - use code without conditional keywords (e.g., 'const x = true;'), 3) Hypotheses MUST NOT constrain 'if' position (feature activates at all indentation levels per trace evidence). Provide concrete invalid/valid examples: BAD activation = '/* Determine if... */', GOOD activation = 'if (isValid) { process(); }'.
```

**mode**: input_validate

---

### [Auto] L12-F3600 r1: Diverse Capitalized Proper Nouns

**Before**: activation=0.60, boundary=1.00 → **After**: activation=0.80, boundary=1.00

**extra_input_guidance**:
```
This is a semantic feature with diverse capitalized proper nouns across contexts (religious, technical, personal names). Regenerate hypotheses to cover ALL capitalized proper nouns, not just religious ones. Example: 'Activates on capitalized proper nouns appearing in titles, institutional names, or technical terms (e.g., 'Revelation', 'Orthodox', 'ModelForm', 'Jenkins').'
```

**mode**: input_validate

---

### [Auto] L24-F58200 r1: Homogeneous Punctuation Surface Forms

**Before**: activation=0.20, boundary=1.00 → **After**: activation=0.80, boundary=1.00

**extra_input_guidance**:
```
Top Neuronpedia tokens are homogeneous punctuation surface forms (., ▁(, ...). Focus hypotheses on exact token surface forms: leading space, token separation rules. Design sentences embedding the EXACT token form (e.g., 'word. (next word)' not 'word.' at sentence end). Do NOT describe semantic roles like 'sentence-final'—this feature fires ONLY on specific tokenization contexts.
```

**mode**: input_validate

---

### [Auto] L24-F59500 r1: Spatial Containment — Full Domain Exclusion

**Before**: activation=1.00, boundary=0.40 → **After**: activation=1.00, boundary=1.00

**extra_input_guidance**:
```
Boundary non-activation is low: design boundary sentences that contain NO spatial containment concepts (e.g., 'She analyzed financial data tables'). Exclude ALL nouns/phrases related to rooms, structures, or boundaries - even near-synonyms like 'hallway' or 'atrium' must be avoided in boundary tests.
```

**mode**: input_validate

---

### [Auto] L24-F60800 r1: Smoking Category — Strict Near-Neighbor Exclusion

**Before**: activation=1.00, boundary=0.60 → **After**: activation=1.00, boundary=1.00

**extra_input_guidance**:
```
Boundary non-activation is critically low (0.6). Rewrite hypotheses to EXCLUDE near-neighbor patterns: 1) Only include EXACT smoking category terms ('smoker', 'tobacco') observed in Neuronpedia, EXCLUDING 'vaping', 'nicotine', or 'e-cigarette' variants; 2) For category markers like 'current'/'former', require they MODIFY smoking terms (e.g. 'current smoker' activates but 'current user' does not); 3) Prepositions like 'on' must appear IN smoking contexts (e.g. 'on smoking' activates but 'on campus' does not). DO NOT reuse previous conceptual angles.
```

**mode**: input_validate

---

### [Auto] L12-F6200 r1: `tak` Token Disambiguation

**Before**: activation=0.80, boundary=0.20 → **After**: activation=1.00, boundary=1.00

**extra_input_guidance**:
```
Boundary non-activation is critically low: the hypothesis must distinguish exact 'tak' token matches (e.g., '▁tak', 'takder') from near-neighbor substrings like 'tidak' or 'tiada'. Specify that activation requires 'tak' to appear as a standalone token or suffix (e.g., 'tak' + optional letters), NOT as part of longer unrelated words. Do not include English contractions in this hypothesis—reserve them for H1.
```

**mode**: input_validate

---

### [Auto] L12-F7500 r1: Reception Verbs — Boundary Must Use Non-Reception Verbs

**Before**: activation=1.00, boundary=0.00 → **After**: activation=1.00, boundary=0.60

**extra_input_guidance**:
```
Boundary non-activation is critically low: design boundary sentences using VERBS THAT ARE NOT SYNONYMS OF RECEPTION (e.g., 'wrote', 'sent', 'composed') instead of near-synonyms like 'obtained' or 'acquired'. The feature fires on semantic reception concepts, so boundary cases must use communication verbs outside this cluster.
```

**mode**: input_validate

---

### [Auto] L12-F8800 (r1→r3): Oscillating Lexical/Semantic Diagnosis

This feature required 3 rounds due to oscillation between lexical and semantic hypotheses.

**Round r1** (activation +0.20): Misidentified as lexical.
```
Boundary non-activation is low: this is a LEXICAL feature (top tokens: '▁parts', '▁ghosts', '▁themselves'). Rewrite H1/H3 to target EXACT token forms ONLY. Exclude near-neighbor patterns like 'nested' or 'hierarchy'.
```

**Round r2** (activation +0.60): Guidance inverted back to semantic after recognizing failure.
```
Previous rounds misclassified this as a lexical feature. Revert to SEMANTIC hypothesis design: top tokens ('▁parts', '▁CHILD', '▁etc') form a thematic cluster about hierarchical components in technical contexts. Generate hypotheses describing this semantic concept. DO NOT constrain to suffixes/spelling.
```

**Round r3** (activation=1.00, boundary=0.80):
```
Boundary non-activation is critically low (0.4). REVISE HYPOTHESES TO EXPLICITLY EXCLUDE NEAR-NEIGHBOR TERMS: (1) Only include attested Neuronpedia tokens ('▁parts', '▁CHILD', '▁etc') in activation scope, (2) Exclude all morphological variants (e.g., 'part', 'child', 'components'), (3) Ban semantic generalizations beyond the observed technical-context cluster.
```

**Lesson**: When initial diagnosis (lexical vs semantic) is uncertain, check Step 0 carefully. If activation rate drops after applying lexical constraints, immediately revert and apply semantic framing (Diagnosis C) instead.

**mode**: input_validate

---

### [Auto] L12-F10100 (r3→r5): Music Roles — Structural to Semantic Pivot

**Round r3** (boundary fixed): Hypothesis focused on structural patterns (colons, commas in listings).
```
Boundary leakage is severe: ONLY activate when musical role tokens appear in listings with (1) a preceding colon AFTER a label (e.g., 'Personnel:', 'Band members:'), AND (2) comma/parenthesis delimiters BETWEEN roles. EXPLICITLY EXCLUDE: sentences using verbs like 'include' or 'feature', prose descriptions, and any context lacking both colon AND delimiter structure.
```

**Round r5** (activation=1.00, boundary=0.80): Structural approach failed; pivoted to semantic.
```
TOP TOKENS REVEAL SEMANTIC FEATURE: Neuronpedia shows consistent music-related terms ("▁Rovers", "▁acoustic", "songwriter", "▁music"). This is NOT a structural punctuation feature. Regenerate hypotheses focusing SOLELY on semantic content: 1) Band names (The Irish Rovers), 2) Music roles (vocalist, guitarist), 3) Instrument terms (acoustic guitar). IGNORE colon/comma syntax completely. Boundary sentences MUST exclude ALL music-related concepts (e.g., use "Tools: hammer, screwdriver..." not "Roles:...").
```

**Lesson**: When boundary-fixing via structural constraints fails to translate to activation success, re-examine the Neuronpedia token list. If it reveals semantic coherence, abandon structural hypotheses entirely.

**mode**: input_validate

---

### [Auto] L12-F11400 r2: Semantic Standard Library Function Categories

**Before**: activation=1.00, boundary=0.00 → **After**: activation=0.80, boundary=1.00

**extra_input_guidance**:
```
This is a semantic feature: top tokens (▁prepare, ▁rotate, ▁str, ▁time) represent *specific standard library function categories* in code (string ops, time handling, data preparation). Boundary sentences must use code identifiers from *different semantic categories* (e.g., 'calculate', 'render', 'validate') that appear in valid code contexts but lack the target semantic pattern. NEVER exclude based on underscores/camelCase - focus exclusively on semantic divergence from the observed token cluster.
```

**mode**: input_validate

---

### [Auto] L12-F12700 r1: Official Labels in Taxonomies

**Before**: activation=0.60, boundary=0.40 → **After**: activation=1.00, boundary=1.00

**extra_input_guidance**:
```
This is a semantic feature with diverse top tokens (geographic, food, institutional) that all represent OFFICIAL LABELS within structured classification systems. Regenerate hypotheses to describe this unified pattern: 'terms serving as canonical identifiers in formal taxonomies' (e.g., electoral regions, food groups, institutional types). DO NOT fragment into domain-specific hypotheses. Boundary sentences must exclude ALL official labels - e.g., use 'The city has many neighborhoods' (not 'Bristol West') as boundary.
```

**mode**: input_validate

---

### [Auto] L12-F16600 r1: Indefinite Pronouns — Scope of Hypotheses

**Before**: activation=1.00, boundary=0.40 → **After**: activation=1.00, boundary=1.00

**extra_input_guidance**:
```
This is a semantic feature covering indefinite pronouns (nothing/something/anything). H2 fails because it narrowly restricts to 'nothing' while the feature fires on multiple pronouns. Rewrite H2 to match H1/H3's semantic scope: 'Activates on indefinite pronouns in existential constructions regardless of negation polarity'. Remove all boundary sentences containing 'something/anything' since they're valid activations, not negatives.
```

**mode**: input_validate

---

### [Auto] L12-F19200 r1: `></' HTML Closing Tag — Exact Tokenization Pattern

**Before**: activation=0.40, boundary=0.80 → **After**: activation=1.00, boundary=1.00

**extra_input_guidance**:
```
Top Neuronpedia tokens are homogeneous (all '></'). Focus hypotheses on the exact token surface form: '></' must appear as a standalone token. Design sentences embedding '></' verbatim (e.g., '...width="100%" border="0" /></a>...'), NOT paraphrases like '/>' or semantic descriptions of closing tags. Do NOT mention HTML/XML concepts - describe only the token pattern.
```

**mode**: input_validate

---

### [Auto] L12-F20500 (r1→r5): Subword Fragment Disambiguation — Four-Round Evolution

This feature shows the difficulty of subword fragment features. Four rounds were needed.

**r1** (activation +0.40): Initial guidance — exact full tokens with `▁`.
```
Top tokens are subword fragments (e.g., 'm' in 'a.m.', 'ank' in 'commbank', 'cc' in 'FCC'). Rewrite hypotheses to specify EXACT token surface forms and their immediate tokenization environment. DO NOT describe semantic contexts like 'acronyms' or 'domain names'.
```

**r2** (activation +0.20): Refined — focus on full token containing the subword.
```
Top tokens activate on *exact subword surface forms*. Rewrite hypotheses to specify: (1) the *full token* containing the subword (e.g., '▁commbank' not 'ank'), (2) mandatory leading space (▁), (3) NO conditions about subsequent tokens/punctuation.
```

**r3** (boundary improved): Pivoted — isolated subword without contextual conditions.
```
This feature activates on *exact subword fragments* (e.g., 'm' in 'a.m.', 'bl' in 'BLM'), not full tokens. Specify: (1) the isolated subword token (e.g., 'm' not 'blm'), (2) mandatory tokenizer ▁ context rules, (3) NO conditions about subsequent characters.
```

**r5** (activation=0.80, boundary=1.00): Final — pure tokenization artifact, no contextual conditions.
```
THIS IS A TOKENIZATION ARTIFACT FEATURE. Rewrite hypotheses to specify: (1) EXACT STRING MATCHES of these SUBWORD FRAGMENTS IN THE TOKENIZER VOCABULARY: '.', 'bl', 'm', 'ca', '/', 'ank', 'com', 'mb', 'ed', 'chool', 's', 'cc'; (2) NO CONTEXTUAL CONDITIONS WHATSOEVER - activation occurs whenever the tokenizer outputs these exact strings, even as parts of larger words; (3) Explicitly state that boundary sentences must AVOID THESE EXACT STRINGS IN ANY CONTEXT. DO NOT USE TERMS LIKE 'STANDALONE' OR 'ISOLATED' - they imply false positional constraints.
```

**Lesson**: For subword fragment features, any contextual conditions (e.g., "after 'bl'", "in acronyms") introduce false constraints. Enumerate exact strings from vocabulary and apply strict string-level exclusion in boundaries.

**mode**: input_validate
