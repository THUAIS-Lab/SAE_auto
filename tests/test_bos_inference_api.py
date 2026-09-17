from __future__ import annotations

from types import SimpleNamespace

import inference_client
import inference_server


def test_bos_inference_client_payload(monkeypatch) -> None:
    captured = {}

    def fake_post_json(**kwargs):
        captured.update(kwargs)
        return {"feature_payloads": []}

    monkeypatch.setattr(inference_client, "_post_json", fake_post_json)
    result = inference_client.call_bos_token_scan(
        server_url="http://127.0.0.1:8008",
        timeout_sec=123.0,
        layer_id=6,
        feature_ids=[900],
        prompt_template="<bos>import {token}",
        prompt_id="agent-r1",
        candidate_token_texts=["foo"],
        scan_full_vocab=False,
        random_sample_size=5000,
        top_k=50,
        batch_size=64,
        activation_threshold=0.0,
        include_special_tokens=False,
        seed=42,
        feature_chunk_size=4096,
    )

    assert result == {"feature_payloads": []}
    assert captured["endpoint"] == "/bos-token-scan"
    assert captured["payload"]["feature_ids"] == [900]
    assert captured["payload"]["random_sample_size"] == 5000


def test_bos_inference_server_uses_preloaded_module(monkeypatch) -> None:
    fake_module = SimpleNamespace(model=object(), tokenizer=object(), device="cuda")
    captured = {}
    monkeypatch.setattr(inference_server.state, "module_for", lambda **_kwargs: fake_module)

    def fake_scan_tokens_with_module(**kwargs):
        captured.update(kwargs)
        return {
            "prompt": {"prompt_id": "agent-r1"},
            "summary": {"selected_feature_count": 1},
            "feature_payloads": [{"feature_id": 900, "top_tokens": []}],
        }

    monkeypatch.setattr(inference_server, "scan_tokens_with_module", fake_scan_tokens_with_module)
    request = inference_server.BosTokenScanRequest(
        layer_id=6,
        feature_ids=[900],
        prompt_template="<bos>",
        prompt_id="agent-r1",
        candidate_token_texts=["foo"],
        random_sample_size=0,
    )

    result = inference_server._bos_token_scan_sync(request)

    assert captured["module"] is fake_module
    assert captured["requested_feature_ids"] == [900]
    assert result["feature_payloads"][0]["feature_id"] == 900
