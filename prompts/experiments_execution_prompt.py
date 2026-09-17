from __future__ import annotations

from typing import Sequence


def build_input_activation_context(*, hypothesis: str, designed_sentences: Sequence[str]) -> str:
    lines = []
    lines.append("Task background:")
    lines.append(
        "You are validating an input-side SAE hypothesis: it describes what kinds of input sentences, "
        "when fed into the model, activate the target SAE feature."
    )
    lines.append("Each sentence should be semantically aligned with the hypothesis and likely to activate the target SAE feature.")
    lines.append("")
    lines.append("Hypothesis:")
    lines.append(hypothesis.strip())
    lines.append("")
    lines.append("Candidate activation sentences:")
    for index, sentence in enumerate(designed_sentences, start=1):
        lines.append(f"{index}. {sentence.strip()}")
    lines.append("")
    lines.append("Evaluation target:")
    lines.append("Measure feature activation for each sentence and compute non-zero activation rate.")
    return "\n".join(lines)


def build_input_boundary_context(*, hypothesis: str, boundary_sentences: Sequence[str]) -> str:
    lines = []
    lines.append("Task background:")
    lines.append(
        "You are validating boundary cases for an input-side SAE hypothesis: it describes what kinds of "
        "input sentences, when fed into the model, activate the target SAE feature."
    )
    lines.append(
        "Each boundary sentence should look similar to the hypothesis semantics but should ideally remain "
        "outside the true activation set."
    )
    lines.append("")
    lines.append("Hypothesis:")
    lines.append(hypothesis.strip())
    lines.append("")
    lines.append("Candidate boundary sentences:")
    for index, sentence in enumerate(boundary_sentences, start=1):
        lines.append(f"{index}. {sentence.strip()}")
    lines.append("")
    lines.append("Evaluation target:")
    lines.append("Measure feature activation for each boundary sentence and compute non-activation rate.")
    return "\n".join(lines)
