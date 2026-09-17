from __future__ import annotations

import argparse

from function import TokenUsageAccumulator
from workflow_step_utils import (
    DEFAULT_API_KEY_FILE,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL_NAME,
    load_step_payload,
    make_llm_client,
    print_step_result,
    score_output_hypotheses,
    write_step_payload,
)


def run_step(args: argparse.Namespace) -> dict:
    layer_id = str(args.layer_id)
    feature_id = str(args.feature_id)
    timestamp = str(args.timestamp)

    step6 = load_step_payload(layer_id=layer_id, feature_id=feature_id, timestamp=timestamp, step_index=6, logs_root=args.logs_root)
    rows = step6["outputs"]["token_change_by_hypothesis"]

    client = make_llm_client(base_url=str(args.llm_base_url), api_key_file=args.llm_api_key_file)
    token_counter = TokenUsageAccumulator()
    score_payload = score_output_hypotheses(
        token_change_rows=rows,
        client=client,
        model=str(args.llm_model),
        token_counter=token_counter,
        max_retries=args.llm_parse_retries,
        retry_backoff_seconds=args.llm_parse_retry_backoff_seconds,
    )

    payload, path = write_step_payload(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=timestamp,
        step_index=7,
        step_name="score_output_hypotheses",
        inputs={"step6": step6["round_id"]},
        parameters=vars(args),
        outputs={
            "metric_version": score_payload["metric_version"],
            "best_hypothesis_idx_by_final_score": score_payload["best_hypothesis_idx_by_final_score"],
            "per_hypothesis": score_payload["per_hypothesis"],
            "llm_calls": score_payload["llm_calls"],
            "token_usage": token_counter.as_dict(),
        },
        logs_root=args.logs_root,
    )
    print_step_result(
        path,
        {
            "best_hypothesis_idx_by_final_score": score_payload["best_hypothesis_idx_by_final_score"],
            "round_id": payload["round_id"],
        },
    )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 7: score output-side hypotheses with LLM token match.")
    parser.add_argument("--layer-id", required=True)
    parser.add_argument("--feature-id", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--llm-api-key-file", default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--llm-parse-retries", type=int, default=2)
    parser.add_argument("--llm-parse-retry-backoff-seconds", type=float, default=1.5)
    return parser


if __name__ == "__main__":
    run_step(build_arg_parser().parse_args())
