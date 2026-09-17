from __future__ import annotations

import json
from pathlib import Path

import agent_bos_token_scan_tool as tool


def test_server_scan_artifacts_are_feature_local(tmp_path: Path, monkeypatch) -> None:
    def fake_call(**kwargs):
        return {
            "prompt": {
                "prompt_id": kwargs["prompt_id"],
                "prompt_template": kwargs["prompt_template"],
                "target_token_position": 1,
            },
            "summary": {
                "prompt_id": kwargs["prompt_id"],
                "selected_feature_count": 1,
                "manual_tokens": {"missing_examples": []},
            },
            "feature_payloads": [
                {
                    "layer_id": 6,
                    "feature_id": 900,
                    "prompt_id": kwargs["prompt_id"],
                    "top_tokens": [{"rank": 1, "token_text": "foo", "activation": 4.0}],
                }
            ],
        }

    monkeypatch.setattr(tool, "call_bos_token_scan", fake_call)
    result = tool.run_agent_bos_token_scan(
        layer_id="6",
        feature_id="900",
        model_checkpoint_path="unused-by-server",
        sae_path="unused-by-server",
        prompt_template="<bos>import {token}",
        prompt_id="agent-r1",
        candidate_token_texts=["foo"],
        output_root=str(tmp_path),
        inference_server_url="http://127.0.0.1:8008",
        inference_timeout_sec=60.0,
        no_inference_server=False,
    )

    prompt_dir = tmp_path / "layer-6" / "feature-900" / "bos_token" / "agent-r1"
    assert result["prompt_id"] == "agent-r1"
    assert json.loads((prompt_dir / "top_tokens.json").read_text(encoding="utf-8"))["feature_id"] == 900
    assert (prompt_dir / "prompt.json").exists()
    assert (prompt_dir / "scan_summary.json").exists()
    assert (prompt_dir / "tool_summary.json").exists()
    assert (prompt_dir / "candidate_token_texts.txt").read_text(encoding="utf-8") == "foo\n"


def test_prompt_collisions_are_resolved_within_feature(tmp_path: Path, monkeypatch) -> None:
    seen = []

    def fake_call(**kwargs):
        seen.append(kwargs["prompt_id"])
        return {
            "prompt": {"prompt_id": kwargs["prompt_id"], "prompt_template": "<bos>"},
            "summary": {"manual_tokens": {"missing_examples": []}},
            "feature_payloads": [{"layer_id": 6, "feature_id": 900, "top_tokens": []}],
        }

    monkeypatch.setattr(tool, "call_bos_token_scan", fake_call)
    kwargs = dict(
        layer_id="6",
        feature_id="900",
        model_checkpoint_path="model",
        sae_path="sae",
        prompt_template="<bos>",
        prompt_id="agent-r1",
        candidate_token_texts=["foo"],
        output_root=str(tmp_path),
        inference_server_url="http://127.0.0.1:8008",
        inference_timeout_sec=60.0,
        no_inference_server=False,
    )
    first = tool.run_agent_bos_token_scan(**kwargs)
    second = tool.run_agent_bos_token_scan(**kwargs)

    assert first["prompt_id"] == "agent-r1"
    assert second["prompt_id"] == "agent-r1-001"
    assert seen == ["agent-r1", "agent-r1-001"]
