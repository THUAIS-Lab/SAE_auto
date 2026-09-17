from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

if os.name == "nt":
    fcntl_stub = types.ModuleType("fcntl")
    fcntl_stub.LOCK_EX = 1
    fcntl_stub.LOCK_SH = 2
    fcntl_stub.LOCK_UN = 8
    fcntl_stub.flock = lambda *_args, **_kwargs: None
    sys.modules.setdefault("fcntl", fcntl_stub)

from function import TokenUsageAccumulator
from workflow_step_utils import (
    LLMParseFailure,
    build_layer_sae_paths,
    judge_chain_pair,
    score_output_hypotheses,
    synthesize_chain_explanation,
)


def _mock_call(outputs):
    queue = iter(outputs)

    def call_llm(**_kwargs):
        return next(queue), None, {"mock": True}

    return call_llm


class LLMParseRetryTests(unittest.TestCase):
    def test_layer_paths_are_built_from_the_configured_root(self):
        paths = build_layer_sae_paths(layer_ids=[0, 6], width="16k", sae_root="/models/sae")
        self.assertEqual(paths[0], str(Path("/models/sae/layer_0/width_16k/average_l0_105")))
        self.assertEqual(paths[6], str(Path("/models/sae/layer_6/width_16k/average_l0_70")))

    def test_step7_retries_malformed_response(self):
        rows = [{
            "hypothesis_index": 1,
            "output_hypothesis": "mentions a cat",
            "token_change": {
                "topk_positive_tokens": [{"token": "cat", "delta": 2.0}],
                "topk_negative_tokens": [],
            },
        }]
        with patch(
            "workflow_step_utils.call_llm",
            side_effect=_mock_call(["not json", '{"matched_token_ids":[1],"reason":"direct match"}']),
        ) as mocked:
            result = score_output_hypotheses(
                token_change_rows=rows,
                client=object(),
                model="test",
                token_counter=TokenUsageAccumulator(),
                retry_backoff_seconds=0,
            )
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(result["per_hypothesis"][0]["matched_token_ids"], [1])
        self.assertEqual(len(result["llm_calls"]), 2)
        self.assertIn("parse_error", result["llm_calls"][0])

    def test_step7_accepts_valid_empty_match_without_retry(self):
        rows = [{
            "output_hypothesis": "mentions a dog",
            "token_change": {
                "topk_positive_tokens": [{"token": "cat", "delta": 2.0}],
                "topk_negative_tokens": [],
            },
        }]
        with patch(
            "workflow_step_utils.call_llm",
            side_effect=_mock_call(['{"matched_token_ids":[],"reason":"no direct match"}']),
        ) as mocked:
            result = score_output_hypotheses(
                token_change_rows=rows,
                client=object(),
                model="test",
                token_counter=TokenUsageAccumulator(),
                retry_backoff_seconds=0,
            )
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(result["per_hypothesis"][0]["matched_token_ids"], [])

    def test_step7_raises_after_retry_budget_is_exhausted(self):
        rows = [{
            "output_hypothesis": "mentions a dog",
            "token_change": {
                "topk_positive_tokens": [{"token": "cat", "delta": 2.0}],
                "topk_negative_tokens": [],
            },
        }]
        with patch(
            "workflow_step_utils.call_llm",
            side_effect=_mock_call(["bad", "still bad", "bad again"]),
        ):
            with self.assertRaises(LLMParseFailure) as context:
                score_output_hypotheses(
                    token_change_rows=rows,
                    client=object(),
                    model="test",
                    token_counter=TokenUsageAccumulator(),
                    retry_backoff_seconds=0,
                )
        self.assertEqual(len(context.exception.llm_calls), 3)

    def test_step8_retries_malformed_response(self):
        with patch(
            "workflow_step_utils.call_llm",
            side_effect=_mock_call([
                "not json",
                '{"score":4,"reason":"plausible"}',
            ]),
        ) as mocked:
            parsed, call = judge_chain_pair(
                input_hypothesis="input",
                output_hypothesis="output",
                token_change={"topk_tokens": []},
                client=object(),
                model="test",
                token_counter=TokenUsageAccumulator(),
                retry_backoff_seconds=0,
            )
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(parsed["score"], 4)
        self.assertEqual(call["attempt_count"], 2)

    def test_step9_retries_overlong_explanation(self):
        too_long = " ".join(["word"] * 51)
        with patch(
            "workflow_step_utils.call_llm",
            side_effect=_mock_call([
                '{"chain_explanation":"' + too_long + '"}',
                '{"chain_explanation":"activation increases the target output"}',
            ]),
        ) as mocked:
            explanation, call = synthesize_chain_explanation(
                best_pair={"token_change": {}},
                client=object(),
                model="test",
                token_counter=TokenUsageAccumulator(),
                retry_backoff_seconds=0,
            )
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(explanation, "activation increases the target output")
        self.assertEqual(call["attempt_count"], 2)

    def test_finish_arguments_are_strict_json(self):
        from agent_proposer import _parse_finish_arguments

        with self.assertRaises(ValueError):
            _parse_finish_arguments("not json")
        parsed = _parse_finish_arguments('{"action":"stop","diagnosis":"done"}')
        self.assertEqual(parsed["action"], "stop")

    def test_skill_compaction_json_call_retries(self):
        from skill_compact import _llm_call_json

        with patch("skill_compact._llm_call", side_effect=["bad", '{"action":"skip"}']) as mocked:
            parsed, _raw = _llm_call_json(
                object(),
                "test",
                "system",
                "user",
                retry_backoff_seconds=0,
            )
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(parsed["action"], "skip")


if __name__ == "__main__":
    unittest.main()
