from __future__ import annotations

from typing import Optional, Sequence


def build_chain_judge_system_prompt() -> str:
    return (
        "You are evaluating whether an input-side SAE hypothesis and an output-side SAE hypothesis "
        "form a plausible causal chain.\n"
        "You must score plausibility on a 1-5 scale using the rubric below.\n"
        "Return JSON only.\n\n"
        "Rubric:\n"
        "1 = no clear semantic connection\n"
        "2 = weak or speculative connection\n"
        "3 = plausible but not strong\n"
        "4 = clear and reasonable connection\n"
        "5 = very strong, obvious connection\n"
    )


def build_chain_judge_user_prompt(
    *,
    input_hypothesis: str,
    output_hypothesis: str,
    top_tokens: Sequence[str],
    extra_guidance: Optional[str] = None,
) -> str:
    token_text = ", ".join(top_tokens)
    body = (
        "Input-side hypothesis:\n"
        f"{input_hypothesis}\n\n"
        "Output-side hypothesis:\n"
        f"{output_hypothesis}\n\n"
        "Observed output-side top tokens (from token change):\n"
        f"{token_text}\n\n"
        "Task: Provide a plausibility score (1-5) and a short reason.\n"
        "Return JSON only in this format:\n"
        "{\n"
        '  "score": 3,\n'
        '  "reason": "short reason"\n'
        "}"
    )
    if extra_guidance and str(extra_guidance).strip():
        body += "\n\nAdditional guidance:\n" + str(extra_guidance).strip()
    return body


def build_chain_synthesis_system_prompt() -> str:
    return (
        "You are writing a concise causal chain explanation for one SAE feature.\n"
        "You are given an input-side hypothesis, an output-side hypothesis, "
        "evidence tokens from feature intervention, and a plausibility score.\n"
        "Synthesize these into a single, concrete chain explanation.\n"
        "Requirements: factual, at most 50 words, no hedging phrases like 'may' or 'might'.\n"
        'Return JSON only: {"chain_explanation": "..."}'
    )


def build_chain_synthesis_user_prompt(
    *,
    input_hypothesis: str,
    output_hypothesis: str,
    chain_judge_score: int,
    chain_judge_reason: str,
    topk_delta_tokens: Sequence[str],
) -> str:
    tokens_str = ", ".join(topk_delta_tokens[:6])
    return (
        f"Input-side hypothesis: {input_hypothesis}\n"
        f"Output-side hypothesis: {output_hypothesis}\n"
        f"Chain plausibility score: {chain_judge_score}/5\n"
        f"Judge reasoning: {chain_judge_reason}\n"
        f"Top output tokens influenced by intervention: {tokens_str}\n\n"
        "Write one chain explanation (<=50 words) describing the causal relationship "
        "from activation context to output effect. Be concrete and specific.\n"
        'Return JSON only: {"chain_explanation": "..."}'
    )
