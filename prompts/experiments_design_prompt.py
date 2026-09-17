from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional

SideType = Literal["input", "output"]


# ── Hypothesis generation prompts ─────────────────────────────────────────────

def build_hypothesis_system_prompt(side: SideType) -> str:
    side_label = "input-side activation" if side == "input" else "output-side intervention"
    if side == "input":
        side_def = (
            "input-side means the hypothesis describes what kinds of input text, "
            "when fed into the model, activate the target SAE feature."
        )
        rules = (
            "Rules:\n"
            "- Give the most natural, parsimonious explanation for the observed pattern.\n"
            "- Cover only what the evidence directly supports — avoid adding unobserved conditions.\n"
            "- One coherent pattern per hypothesis; do not conflate unrelated token types.\n"
            "- At most 20 words per hypothesis."
        )
    else:
        side_def = (
            "output-side means the hypothesis describes the semantic concept that the promoted tokens "
            "represent — what the model is biased toward when this feature is activated."
        )
        rules = (
            "Rules:\n"
            "- Give the most natural, parsimonious semantic explanation for the token cluster.\n"
            "- Base the explanation on what the tokens most directly and obviously share.\n"
            "- Do not force unrelated tokens into a single abstract concept.\n"
            "- At most 20 words per hypothesis."
        )
    return (
        "You are an expert interpretability researcher for sparse autoencoder (SAE) features.\n"
        f"Your task is to infer hypotheses about one SAE feature based on observations.\n"
        f"You are generating hypotheses for the {side_label} behavior.\n"
        f"Definition: {side_def}\n"
        f"{rules}\n"
        "Do not output any extra commentary."
    )


def build_hypothesis_user_prompt(
    *,
    side: SideType,
    observation: Dict[str, Any],
    num_hypothesis: int,
    extra_guidance: Optional[str] = None,
) -> str:
    source = str(observation.get("source", "")).strip()
    side_label = "input-side activation" if side == "input" else "output-side intervention"

    if side == "input":
        if source == "bos_token":
            obs_desc = (
                "Observation source: BOS token scan — each entry is a vocabulary token "
                "that activates this feature when placed immediately after <bos>."
            )
        elif source == "gradient_token":
            obs_desc = (
                "Observation source: gradient token scan — each entry is a vocabulary token "
                "ranked by input-embedding gradient evidence for increasing the target SAE feature "
                "under the recorded prompt. Gradient scores are directional sensitivity evidence, "
                "not direct activation values."
            )
        else:
            obs_desc = (
                "Observation source: Neuronpedia activation examples — real text segments "
                "where the feature fires, with per-token activation values and the peak token."
            )
        constraints = (
            "Constraints:\n"
            "- Each hypothesis ≤ 20 words.\n"
            "- Give the most natural explanation for what the activating tokens share.\n"
            "- Cover only what the evidence directly shows; do not add unobserved conditions.\n"
            "- One coherent pattern per hypothesis — do not conflate unrelated token types."
        )
    else:
        obs_desc = (
            "Observation source: Top promoted tokens from feature steering intervention "
            "(tokenchange). You only have token surface forms — no context sentences."
        )
        constraints = (
            "Constraints:\n"
            "- Each hypothesis ≤ 20 words.\n"
            "- Give the most natural, parsimonious semantic explanation for what the tokens share.\n"
            "- Base the explanation on what the tokens most directly and obviously have in common.\n"
            "- One coherent concept per hypothesis — do not force unrelated tokens into one group."
        )

    guidance_block = ""
    if extra_guidance and str(extra_guidance).strip():
        guidance_block = "\n\nAdditional guidance:\n" + str(extra_guidance).strip()

    return (
        f"Background: You are analyzing one SAE feature.\n"
        f"{obs_desc}\n\n"
        f"Task: Generate exactly {num_hypothesis} distinct hypotheses for the "
        f"{side_label} explanation.\n"
        f"{constraints}"
        f"{guidance_block}\n\n"
        "Output format (JSON only):\n"
        '{\n  "hypotheses": ["hypothesis 1", "hypothesis 2"]\n}\n\n'
        "Observation:\n"
        f"{json.dumps(observation, ensure_ascii=False, indent=2)}"
    )


# ── Experiment sentence design prompts ────────────────────────────────────────

def build_system_prompt(side: SideType) -> str:
    side_label = "input-side activation" if side == "input" else "output-side intervention"
    side_def = (
        "input-side means the hypothesis describes what kinds of input sentences, "
        "when fed into the model, activate the target SAE feature."
        if side == "input" else
        "output-side means the hypothesis describes how the model's output changes "
        "after the target SAE feature value is intervened on."
    )
    return (
        "You are an expert interpretability researcher for sparse autoencoder (SAE) features.\n"
        "You design validation experiments for hypotheses about one SAE feature.\n"
        f"The current task is for {side_label} hypotheses.\n"
        f"Definition: {side_def}\n"
        "Follow the user's required output format exactly.\n"
        "Do not output extra commentary."
    )


def build_user_prompt(
    *,
    side: SideType,
    hypothesis: str,
    num_sentences: int,
    activation_examples: Optional[List[Dict[str, Any]]] = None,
) -> str:
    if side == "input":
        examples_block = ""
        if activation_examples:
            lines = []
            for i, ex in enumerate(activation_examples, 1):
                toks = ex.get("activation_tokens", [])
                peak = max(toks, key=lambda t: float(t.get("value", 0)), default=None) if toks else None
                peak_str = f" (peak: '{peak['token']}')" if peak else ""
                text = "".join(t.get("token", "") for t in toks).strip()
                lines.append(f"{i}. {text}{peak_str}")
            examples_block = (
                "\n\nReal activation examples (for reference only — do NOT copy these verbatim):\n"
                + "\n".join(lines)
            )
        return (
            "Background:\n"
            "An SAE feature is hypothesized to activate for a specific semantic pattern.\n"
            "You need to generate test sentences or short snippets that are likely to activate this feature.\n\n"
            "Task:\n"
            f"Given the hypothesis below, generate exactly {num_sentences} sentences or short snippets.\n"
            "Each sentence or snippet must directly reflect the literal meaning of the hypothesis and be likely "
            "to trigger the corresponding SAE feature activation.\n"
            "While staying faithful to the hypothesis, vary the surrounding context as much as reasonably possible.\n"
            "Use diverse syntax, discourse settings, and nearby content instead of repeating one narrow scenario.\n"
            "You may use a multi-line snippet when the hypothesis specifically involves formatting, layout, or newline-sensitive structure.\n"
            "Keep each sentence or snippet natural, clear, and under 60 words.\n"
            "Avoid near-duplicates.\n\n"
            "Output format (JSON only):\n"
            '{\n  "sentences": ["sentence 1", "sentence 2"]\n}\n\n'
            f"Hypothesis:\n{hypothesis}"
            f"{examples_block}"
        )

    return (
        "Background:\n"
        "You are preparing output-side intervention validation placeholders for an SAE feature.\n\n"
        "Task:\n"
        "Return the following list exactly as JSON.\n\n"
        "Output format (JSON only):\n"
        '{\n  "sentences": ["The explanation is simple:", "I think", "We"]\n}\n\n'
        f"Hypothesis:\n{hypothesis}"
    )


def build_boundary_system_prompt() -> str:
    return (
        "You are an expert at designing adversarial boundary test cases for SAE feature explanations.\n"
        "Follow the user's required output format exactly.\n"
        "Do not output extra commentary."
    )


def build_boundary_user_prompt(
    *,
    hypothesis: str,
    boundary_case_count: int,
    trigger_block: str = "",
) -> str:
    return (
        "Task: generate boundary sentences or short snippets that should NOT activate the SAE feature described below.\n\n"
        "Rules for a good boundary case:\n"
        "- Same domain or syntactic structure as the hypothesis describes.\n"
        "- Must NOT contain the specific trigger tokens or their morphological variants.\n"
        "- Use synonyms, related terms, or structurally similar but lexically different phrasing.\n"
        "- A reader who knows the feature would agree it should NOT fire on this boundary case.\n"
        "- You may use a multi-line snippet when the hypothesis specifically involves formatting, layout, or newline-sensitive structure.\n"
        f"{trigger_block}\n\n"
        f"Hypothesis:\n{hypothesis}\n\n"
        f"Generate exactly {boundary_case_count} boundary sentences or short snippets.\n"
        "Return JSON only:\n"
        '{\n  "boundary_cases": ["case 1", "case 2"]\n}'
    )
