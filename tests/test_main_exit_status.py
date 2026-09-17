from run_main_experiment import _agent_result_failed


def test_agent_result_failure_classification() -> None:
    assert _agent_result_failed("ERROR: API unavailable")
    assert _agent_result_failed("failed_pipeline score=2/5 rounds=1")
    assert _agent_result_failed("SKIP (no trace.json)")
    assert not _agent_result_failed("OK score=2/5 rounds=1")
    assert not _agent_result_failed("SKIP (agent complete)")
