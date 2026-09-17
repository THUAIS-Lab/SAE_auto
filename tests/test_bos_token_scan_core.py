from __future__ import annotations

from types import SimpleNamespace

import torch

import input_bos_token_scan as scan


class FakeTokenizer:
    bos_token_id = 2
    all_special_ids = [2]
    vocab_size = 5

    def __init__(self) -> None:
        self._vocab = {"a": 0, "b": 1, "<bos>": 2, "c": 3, "d": 4}

    def get_vocab(self):
        return dict(self._vocab)

    def __len__(self) -> int:
        return len(self._vocab)

    def __call__(self, text, **_kwargs):
        return {"input_ids": [self._vocab[text]] if text else []}

    def convert_ids_to_tokens(self, ids):
        reverse = {value: key for key, value in self._vocab.items()}
        return [reverse[int(token_id)] for token_id in ids]


def test_scan_tokens_with_preloaded_module(monkeypatch) -> None:
    module = SimpleNamespace(tokenizer=FakeTokenizer(), device=torch.device("cpu"), model=object())

    def fake_features(_module, *, input_ids, attention_mask):
        del attention_mask
        batch, sequence = input_ids.shape
        features = torch.zeros((batch, sequence, 2), dtype=torch.float32)
        candidate_ids = input_ids[:, 1].float()
        features[:, 1, 0] = candidate_ids
        features[:, 1, 1] = 10.0 - candidate_ids
        return features

    monkeypatch.setattr(scan, "_get_sae_features", fake_features)

    result = scan.scan_tokens_with_module(
        module=module,
        layer_id=6,
        requested_feature_ids=[0],
        prompt_template="<bos>",
        prompt_id="prompt-0001",
        manual_token_texts=["a", "c", "d"],
        scan_full_vocab=False,
        random_sample_size=0,
        include_special_tokens=False,
        seed=42,
        batch_size=2,
        top_k=2,
        activation_threshold=0.0,
        feature_chunk_size=16,
    )

    assert result["prompt"]["target_token_position"] == 1
    assert result["summary"]["selected_feature_count"] == 1
    assert result["summary"]["token_scan"]["evaluated_token_count"] == 3
    payload = result["feature_payloads"][0]
    assert payload["feature_id"] == 0
    assert [item["token_text"] for item in payload["top_tokens"]] == ["d", "c"]
    assert [item["activation"] for item in payload["top_tokens"]] == [4.0, 3.0]


def test_scan_core_rejects_missing_token_pool() -> None:
    module = SimpleNamespace(tokenizer=FakeTokenizer(), device=torch.device("cpu"), model=object())

    try:
        scan.scan_tokens_with_module(
            module=module,
            layer_id=6,
            requested_feature_ids=[0],
            prompt_template="<bos>",
            prompt_id="prompt-0001",
            manual_token_texts=[],
            scan_full_vocab=False,
            random_sample_size=0,
            include_special_tokens=False,
            seed=42,
            batch_size=2,
            top_k=2,
            activation_threshold=0.0,
            feature_chunk_size=16,
        )
    except RuntimeError as exc:
        assert "No valid token ids" in str(exc)
    else:
        raise AssertionError("expected an empty token-pool error")


def test_scan_core_requires_explicit_bos_token() -> None:
    module = SimpleNamespace(tokenizer=FakeTokenizer(), device=torch.device("cpu"), model=object())
    try:
        scan.scan_tokens_with_module(
            module=module,
            layer_id=6,
            requested_feature_ids=[0],
            prompt_template="prefix {token}",
            prompt_id="not-bos",
            manual_token_texts=["a"],
            scan_full_vocab=False,
            random_sample_size=0,
            include_special_tokens=False,
            seed=42,
            batch_size=2,
            top_k=2,
            activation_threshold=0.0,
            feature_chunk_size=16,
        )
    except ValueError as exc:
        assert "must contain <bos>" in str(exc)
    else:
        raise AssertionError("BOS scan must include the BOS token")
