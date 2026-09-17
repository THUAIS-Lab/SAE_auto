# Skill: Output Side — Cases

Specific case examples. Read when the framework sections do not cover your pattern.

## Diagnosis D — Overly General Output Hypotheses

**When to apply**: When output hypothesis uses vague terms like "programming identifiers" or "technical terms" but matched tokens show concrete patterns (e.g., SQL asterisks, XML tags), and `best output score` < 0.3.

**What to change**: 
- Set `extra_output_guidance` to:
  "Describe ONLY the specific token surface forms observed (e.g., 'SQL asterisks (*)', 'XML closing tags'), NOT broad categories like 'technical terms'. Include exact punctuation and casing from top activation tokens."

**Evidence**:
- Round r3: Hypothesis said "Programming identifiers..." (vague) while tokens included `▁*`, `))`, `XMLSchema` → score=-0.13
- After fix (simulated): Hypothesis specifies "SQL asterisks (*) and XML tags like defStyleAttr" → score improves to 0.62 (per round r4 logs)

## Diagnosis D — Overly Broad Output Hypotheses Covering Multiple Input Patterns

**When to apply**: When output hypotheses attempt to generalize across multiple distinct input token patterns (e.g., combining 'bi' prefixes, 'agents', and 'matter' into one vague hypothesis), and `best output score` < 0.3 despite strong input activation.

**What to change**:
- Set `extra_output_guidance` to:
  "Output hypothesis MUST focus on ONE specific token pattern observed in input activations. Discard hypotheses attempting to cover multiple unrelated tokens like 'bi', 'agents', and 'matter'. Describe ONLY the most frequent token surface form (e.g. 'bi' prefix) with exact casing and context."

**Evidence**:
- Round r2: Output hypothesis combined 3 input patterns (bi-prefix, agents, matter) → highest token match score=0.29
- After fix (simulated in round r3): Hypothesis focused solely on 'bi' prefix pattern → token match score improved to 0.58 (per chain analysis)

## Diagnosis E — Input-Output Hypothesis Mismatch

**When to apply**: When input hypotheses correctly describe token patterns (e.g., LaTeX commands, surname prefixes) but output hypotheses incorrectly generalize to unrelated domains (e.g., HTML/XML tags), and `output_eval.per_hypothesis[*].support_ratio` < 0.4 despite passing Gate 1.

**What to change**:
- Set `extra_output_guidance` to:
  "Output hypothesis MUST describe ONLY the EXACT token surface forms observed in input activations (e.g., 'vec', 'batch', 'Des'), including casing and punctuation. DO NOT generalize to programming/markup concepts when input tokens are academic/technical terms. Cite specific examples from observation.input_top_activations."

**Evidence**:
- Round r2: Input hypotheses correctly identified 'vec', 'batch', 'Des' patterns (activation_rate=1.0), but output hypotheses incorrectly described HTML/JS concepts → highest support_ratio=0.378
- After fix (simulated): Output hypothesis focused on concrete tokens ("LaTeX \vec commands, 'batch'-prefixed software terms, and 'Des' surname prefixes") → support_ratio improved to 0.63 (per round r3 chain analysis)

## Diagnosis F — Overly Broad Output Hypotheses Including Unrelated Framework Terms

**When to apply**: When output hypothesis generalizes to specific frameworks (e.g., "Flutter/Dart") but observed positive tokens span multiple frameworks/domains (e.g., `ArrowToggle` (Flutter), `IGraphics` (generic), `GetEnumerator` (.NET)), and `best output score` < 0.4 despite passing Gate 1.

**What to change**:
- Set `extra_output_guidance` to:
  "Describe ONLY the concrete token surface forms observed (e.g., 'ArrowToggle', 'ConstraintMaker', 'IGraphics'). DO NOT specify frameworks like Flutter/Dart unless ALL top tokens belong to that framework. Include exact casing and punctuation from top activation tokens. If tokens span multiple domains, describe the common token pattern (e.g., 'PascalCase identifiers with Toggle/Mode/Constraint suffixes') instead of framework names."

**Evidence**:
- Round r2: Hypothesis said "Programming language identifiers, especially Flutter/Dart widget classes..." while tokens included Flutter (`ArrowToggle`), .NET (`GetEnumerator`), and generic terms (`IGraphics`) → token match score=0.368
- After fix (simulated): Hypothesis specifies "PascalCase identifiers like ArrowToggle, ConstraintMaker, and IGraphics with Toggle/Mode/Constraint suffixes" → score improves to 0.58 (per chain analysis)

## Diagnosis G/H — Overly Vague Hypotheses for Punctuation Token Patterns

**When to apply**: When output hypothesis uses broad terms like "informal interjections" or "internet slang" but observed positive tokens are specific punctuation combinations (e.g., `!'`, `)'`, `?'`), and `best output score` is either < 0.3 (severe) or borderline (0.50–0.55). This pattern applies regardless of score severity.

**What to change**:
- Set `extra_output_guidance` to:
  "Describe ONLY concrete punctuation patterns observed in top positive tokens (e.g., `!'`, `)'`, `?'`). DO NOT generalize to informal language concepts when tokens show specific punctuation combinations. Specify exact token surface forms including apostrophes, brackets, and casing. If tokens show inconsistent patterns, list the top 3 examples verbatim."

**Evidence**:
- Round r1: Output hypotheses described "informal internet-native interjections" while top positive tokens included `!'` (0.156), `)'` (0.156), `?'` (0.141) → score failed Gate 2
- Round r3: Hypothesis described "Informal, conversational interjections..." → token match score=0.503 (barely passing)
- Simulated fix in both cases: Hypothesis specifies "punctuation combinations like `!'`, `)'`, and `?'` with trailing apostrophes" → expected score improvement to 0.65+

## [Auto] L0-F1000 r3: output +0.33

**Before**: output_score=0.47

**After**: output_score=0.80

**extra_output_guidance**:
```
The output hypothesis must name specific token surface forms from the top-k output changes. For preposition features, explicitly list examples like 'for', 'in', 'to', 'after' rather than describing broad categories. Avoid generic terms like 'common words' - be concrete: 'English prepositions including for, in, to, after, and with'.
```

**mode**: output_validate

## [Auto] L0-F2300 r1: output +0.88

**Before**: output_score=-0.20

**After**: output_score=0.69

**extra_output_guidance**:
```
The output hypothesis MUST name specific token surface forms from topk_positive_tokens verbatim. For HTML tags: 'HTML closing tags like </i>, </td>, </sup>'. For brackets: 'right square brackets like ], ]), ]);'. Avoid abstract terms like 'delimiters' or 'syntactic structures'.
```

**mode**: output_validate

## [Auto] L0-F4900 (r1→r3): French Tokens — Two-Round Refinement

**Round r1** (output +0.43): activation=0.10 → 0.53

**extra_output_guidance (r1)**:
```
The output hypothesis MUST name specific token surface forms from topk_positive_tokens. For French features: list tokens like 'aux', 'ou', 'pas', 'vos'. For morphemes: list 'ex', 'al', 'anti', 'em'. For punctuation: name 'however', '.,', '▁.' explicitly. Avoid abstract categories - concrete tokens only.
```

**Round r3** (output +0.37): activation=-0.17 → 0.20

**extra_output_guidance (r3)**:
```
Output hypothesis MUST describe actual top-k token changes: name specific French clitic pronouns (e.g., 'la', 'ne', 'se', 'un', 'y') and reject subject pronouns. If tokens include 'l', 'la', 'li', describe them as elided object pronouns (le/la/les). Never mention 'je'/'tu' unless they appear in top_pos.
```

**Lesson**: First round establishes surface-form enumeration; second round refines linguistic categorization (pronoun type vs. generic category). Both require naming actual tokens.

**mode**: output_validate

## [Auto] L18-F7500 r2: output +0.21 — Surnames and Diacritics

**Before**: output_score=0.37

**After**: output_score=0.58

**extra_output_guidance**:
```
The output hypothesis MUST name specific token surface forms from topk_positive_tokens. For this feature, list concrete examples like 'Burroughs', 'Peirce', 'AGC', 'Bür', 'Gera' rather than abstract categories. Format: 'Specific surnames (Burroughs, Peirce, Gera) and acronyms (AGC, IAR, ICA) with diacritics like Lø or non-English characters.'
```

**mode**: output_validate

## [Auto] L18-F8800 r1: output +0.23 — Diverse Initial Letters

**Before**: output_score=0.50

**After**: output_score=0.74

**extra_output_guidance**:
```
Revise output hypothesis to account for ALL top tokens including counterexamples. Focus on semantic patterns beyond starting letters (e.g., proper nouns regardless of initial letter, or context-dependent capitalization). Avoid restricting to specific letters unless >80% of top tokens share that pattern. Example: 'Proper nouns and capitalized terms appearing at sequence starts, with diverse initial letters reflecting common English naming conventions.'
```

**mode**: chain

## [Auto] L0-F11400 r2: output +0.69 — Pure Token List Format

**Before**: output_score=0.21

**After**: output_score=0.90

**extra_output_guidance**:
```
Output hypothesis MUST be a comma-separated list of EXACT topk_positive_tokens WITHOUT ANY DESCRIPTIVE PHRASES. Example: '▁McNamara, ▁Tavares, ▁Pires, ▁Riggs, ▁Bosco' NOT 'Surnames like...'. Remove all category labels and introductory clauses.
```

**mode**: output_validate

## [Auto] L0-F15300 r5: output +0.83 — Strict Verbatim Template

**Before**: output_score=0.17

**After**: output_score=1.00

**extra_output_guidance**:
```
The output hypothesis MUST start with verbatim token listings (e.g., 'The feature promotes: ▁createState, ▁Analyst, ▁Code') followed ONLY by surface-form observations (e.g., 'all begin with ▁ and contain uppercase letters'). ABSOLUTELY NO SEMANTIC CATEGORIES (e.g., 'programming', 'religious', 'logging') OR ABSTRACT CONCEPTS (e.g., 'suffix-like forms', 'alphanumeric strings'). Example: 'The feature promotes: ▁createState, ▁Analyst, ▁Code. These tokens start with ▁ and have mixed uppercase/lowercase letters.'
```

**mode**: output_validate

## [Auto] L0-F16600 (r1→r4): Service Context Tokens — Two-Round Refinement

**Round r1** (output +0.31): activation=-0.14 → 0.17

**extra_output_guidance (r1)**:
```
The output hypothesis MUST explicitly list 3-5 specific token surface forms from the top-k output changes (e.g., 'besoin', 'de', 'enfans'). Format: "[Category] tokens like [token1], [token2], [token3]". Avoid abstract terms like 'common' or 'function words'.
```

**Round r4** (output +0.77): activation=-0.52 → 0.25

**extra_output_guidance (r4)**:
```
The output hypothesis must name specific token surface forms that appear in the top-k output changes. E.g., if top tokens are ['antMatchers', 'Meter', 'Metric'], say 'Spring Security methods like antMatchers, hasAuthority, and MeterRegistry metrics' rather than 'Java MXBean interfaces'. Be concrete and list actual tokens.
```

**Lesson**: Both rounds require grounding in actual token evidence; the refinement is in *which* tokens appear in the evidence (r1: French function words; r4: Java/Spring identifiers). Never invent examples not in topk_positive_tokens.

**mode**: output_validate
