from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import agent_proposer
import agent_runner
import run_main_experiment
import step1_initial_observation
from step9_synthesize_chain_explanation import _slim_observation
from workflow_step_utils import load_bos_token_observation


def _tool_names(source: str) -> set[str]:
    return {
        tool["function"]["name"]
        for tool in agent_proposer._tools_for_input_source(source)
    }


def test_bos_scan_tool_is_source_specific() -> None:
    assert "run_bos_token_scan" not in _tool_names("neuronpedia")
    assert "run_bos_token_scan" in _tool_names("bos_token")


def test_step1_retry_command_keeps_bos_source_and_selected_prompt(tmp_path: Path) -> None:
    cmd = agent_runner._build_step1_command(
        layer_id="6",
        feature_id="900",
        timestamp="run_r1",
        logs_root=tmp_path / "logs",
        model_id="gemma-2-2b",
        sae_width="16k",
        input_observation_source="bos_token",
        bos_token_root=tmp_path / "observations",
        bos_prompt_id="feature-900-agent-r1",
    )

    assert cmd[cmd.index("--input-observation-source") + 1] == "bos_token"
    assert cmd[cmd.index("--bos-token-root") + 1] == str(tmp_path / "observations")
    assert cmd[cmd.index("--bos-prompt-id") + 1] == "feature-900-agent-r1"


def test_neuronpedia_step1_command_does_not_require_bos_paths(tmp_path: Path) -> None:
    cmd = agent_runner._build_step1_command(
        layer_id="6",
        feature_id="900",
        timestamp="run_r1",
        logs_root=tmp_path / "logs",
        model_id="gemma-2-2b",
        sae_width="16k",
        input_observation_source="neuronpedia",
        bos_token_root=None,
        bos_prompt_id=None,
    )

    assert cmd[cmd.index("--input-observation-source") + 1] == "neuronpedia"
    assert "--bos-token-root" not in cmd
    assert "--bos-prompt-id" not in cmd


def test_bos_file_access_is_limited_to_current_feature(tmp_path: Path) -> None:
    root = tmp_path / "initial"
    current = root / "layer-6" / "feature-900" / "bos_token"
    sibling = root / "layer-6" / "feature-930" / "bos_token"
    current.mkdir(parents=True)
    sibling.mkdir(parents=True)
    (current / "visible.json").write_text('{"feature": 900}', encoding="utf-8")
    (sibling / "hidden.json").write_text('{"feature": 930}', encoding="utf-8")

    token = agent_proposer._CURRENT_FEATURE_OBSERVATION_DIR.set(current)
    try:
        assert json.loads(agent_proposer._run_read_file("initial_observation/visible.json"))["feature"] == 900
        assert "not allowed" in agent_proposer._run_read_file(
            "initial_observation/../../feature-930/bos_token/hidden.json"
        )
        assert "not allowed" in agent_proposer._run_read_file(str(sibling / "hidden.json"))
    finally:
        agent_proposer._CURRENT_FEATURE_OBSERVATION_DIR.reset(token)


def test_loaded_bos_observation_records_selected_prompt(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "layer-6" / "feature-900" / "bos_token" / "prompt-0001"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "top_tokens.json").write_text(
        json.dumps(
            {
                "prompt_id": "prompt-0001",
                "prompt_template": "<bos>",
                "top_tokens": [{"token_text": "token", "activation": 3.0}],
            }
        ),
        encoding="utf-8",
    )

    observation = load_bos_token_observation(
        bos_root=str(tmp_path),
        layer_id="6",
        feature_id="900",
        prompt_id="prompt-0001",
    )

    assert observation["source"] == "bos_token"
    assert observation["bos_token_scan_meta"]["prompt_id"] == "prompt-0001"
    assert observation["bos_token_scan_meta"]["prompt_template"] == "<bos>"


def test_trace_preserves_bos_prompt_but_neuronpedia_trace_has_no_bos_field() -> None:
    bos = _slim_observation(
        {
            "input_source": "bos_token",
            "input_side_observation": {
                "source": "bos_token",
                "bos_token_scan_meta": {
                    "prompt_id": "agent-r1",
                    "prompt_template": "<bos>{token}",
                },
            },
        }
    )
    neuronpedia = _slim_observation({"input_source": "neuronpedia"})

    assert bos["input_source"] == "bos_token"
    assert bos["bos_token_scan_meta"]["prompt_id"] == "agent-r1"
    assert "bos_token_scan_meta" not in neuronpedia


def test_initial_pipeline_passes_bos_source(monkeypatch, tmp_path: Path) -> None:
    commands = []
    monkeypatch.setattr(
        run_main_experiment,
        "_run_step",
        lambda cmd, **_kwargs: commands.append(cmd) or True,
    )
    run_main_experiment._run_initial_pipeline(
        layer=6,
        fid=900,
        ts="run",
        sae_path="sae",
        model_path="model",
        device="cuda",
        llm_base_url="url",
        llm_model="llm",
        llm_api_key_file=None,
        inference_server_url="server",
        inference_timeout_sec=60.0,
        no_inference_server=False,
        logs_root=tmp_path / "logs",
        log_file=tmp_path / "run.log",
        input_observation_source="bos_token",
        bos_token_root=tmp_path / "observations",
        bos_prompt_id="prompt-0001",
    )
    step1 = commands[0]
    assert step1[step1.index("--input-observation-source") + 1] == "bos_token"
    assert step1[step1.index("--bos-prompt-id") + 1] == "prompt-0001"


def test_bos_step1_never_calls_neuronpedia(monkeypatch, tmp_path: Path) -> None:
    prompt_dir = tmp_path / "initial" / "layer-6" / "feature-900" / "bos_token" / "prompt-0001"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "top_tokens.json").write_text(
        json.dumps(
            {
                "layer_id": 6,
                "feature_id": 900,
                "prompt_id": "prompt-0001",
                "prompt_template": "<bos>",
                "top_tokens": [{"token_text": "foo", "activation": 3.0}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        step1_initial_observation,
        "fetch_and_parse_feature_observation",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("Neuronpedia must not be called")),
    )
    args = SimpleNamespace(
        layer_id="6",
        feature_id="900",
        timestamp="bos-step1-test",
        model_id="gemma-2-2b",
        width="16k",
        selection_method=1,
        observation_m=10,
        observation_n=5,
        neuronpedia_api_key=None,
        neuronpedia_timeout=30,
        input_observation_source="bos_token",
        bos_token_root=str(tmp_path / "initial"),
        bos_prompt_id="prompt-0001",
        gradient_token_root=str(tmp_path / "initial"),
        gradient_prompt_id="prompt-0001",
        logs_root=str(tmp_path / "logs"),
    )

    payload = step1_initial_observation.run_step(args)
    observation = payload["outputs"]["observation"]

    assert observation["input_source"] == "bos_token"
    assert observation["input_side_observation"]["source"] == "bos_token"


def test_bos_prompt_choice_must_be_feature_local(tmp_path: Path) -> None:
    selected = tmp_path / "layer-6" / "feature-900" / "bos_token" / "agent-r1"
    selected.mkdir(parents=True)
    (selected / "top_tokens.json").write_text("{}", encoding="utf-8")

    prompt_id, changed = agent_runner._resolve_bos_prompt_choice(
        input_observation_source="bos_token",
        bos_token_root=tmp_path,
        layer_id="6",
        feature_id="900",
        current_prompt_id="prompt-0001",
        harness={"bos_prompt_id": "agent-r1"},
    )
    assert prompt_id == "agent-r1"
    assert changed is True

    try:
        agent_runner._resolve_bos_prompt_choice(
            input_observation_source="bos_token",
            bos_token_root=tmp_path,
            layer_id="6",
            feature_id="930",
            current_prompt_id="prompt-0001",
            harness={"bos_prompt_id": "agent-r1"},
        )
    except ValueError as exc:
        assert "top_tokens.json" in str(exc)
    else:
        raise AssertionError("another feature's prompt must not be selectable")
