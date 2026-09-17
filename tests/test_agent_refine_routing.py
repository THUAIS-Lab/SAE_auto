from agent_proposer import _normalize_finish_for_mode


def test_promotes_nested_rerun_steps() -> None:
    result = _normalize_finish_for_mode(
        {
            "action": "retry",
            "harness": {
                "extra_output_guidance": "Describe the observed tokens precisely.",
                "rerun_steps": [6, 7, 8, 9],
            },
        },
        "output_validate",
    )

    assert result["rerun_steps"] == [6, 7, 8, 9]
    assert "rerun_steps" not in result["harness"]


def test_gate2_failure_converts_chain_only_retry_to_output_retry() -> None:
    result = _normalize_finish_for_mode(
        {
            "action": "retry",
            "harness": {
                "extra_chain_guidance": "Use the exact intervention tokens.",
                "rerun_steps": [8, 9],
            },
        },
        "output_validate",
    )

    assert result["harness"]["extra_output_guidance"] == "Use the exact intervention tokens."
    assert "extra_chain_guidance" not in result["harness"]
    assert result["rerun_steps"] == [6, 7, 8, 9]


def test_gate2_intervention_retry_starts_at_step5() -> None:
    result = _normalize_finish_for_mode(
        {
            "action": "retry",
            "harness": {
                "intervention_scope": "last_token_only",
            },
            "rerun_steps": [8, 9],
        },
        "output_validate",
    )

    assert result["rerun_steps"] == [5, 6, 7, 8, 9]


def test_chain_mode_keeps_chain_only_retry() -> None:
    result = _normalize_finish_for_mode(
        {
            "action": "retry",
            "harness": {
                "extra_chain_guidance": "Clarify the causal connection.",
            },
            "rerun_steps": [8, 9],
        },
        "chain",
    )

    assert result["harness"]["extra_chain_guidance"] == "Clarify the causal connection."
    assert result["rerun_steps"] == [8, 9]
