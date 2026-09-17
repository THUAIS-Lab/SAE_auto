from __future__ import annotations

import argparse

from function import TokenUsageAccumulator
from workflow_step_utils import (
    DEFAULT_API_KEY_FILE,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL_NAME,
    judge_chain_pair,
    load_step_payload,
    make_llm_client,
    output_scores_by_idx,
    pick_best_pair,
    print_step_result,
    write_step_payload,
)


def run_step(args: argparse.Namespace) -> dict:
    layer_id = str(args.layer_id)
    feature_id = str(args.feature_id)
    timestamp = str(args.timestamp)

    step6 = load_step_payload(layer_id=layer_id, feature_id=feature_id, timestamp=timestamp, step_index=6, logs_root=args.logs_root)
    step7 = load_step_payload(layer_id=layer_id, feature_id=feature_id, timestamp=timestamp, step_index=7, logs_root=args.logs_root)
    rows = step6["outputs"]["token_change_by_hypothesis"]
    score_by_idx = output_scores_by_idx(step7["outputs"])

    client = make_llm_client(base_url=str(args.llm_base_url), api_key_file=args.llm_api_key_file)
    token_counter = TokenUsageAccumulator()
    pairs = []
    llm_calls = []
    for idx, item in enumerate(rows, start=1):
        input_hypothesis = str(item.get("input_hypothesis", "")).strip()
        output_hypothesis = str(item.get("output_hypothesis", "")).strip()
        token_change = item.get("token_change", {}) or {}
        parsed, call = judge_chain_pair(
            input_hypothesis=input_hypothesis,
            output_hypothesis=output_hypothesis,
            token_change=token_change,
            client=client,
            model=str(args.llm_model),
            token_counter=token_counter,
            extra_chain_guidance=getattr(args, "extra_chain_guidance", None),
            max_retries=args.llm_parse_retries,
            retry_backoff_seconds=args.llm_parse_retry_backoff_seconds,
        )
        call["index"] = idx
        llm_calls.append(call)
        output_score = score_by_idx.get(idx, {})
        pairs.append(
            {
                "idx": idx,
                "input_hypothesis": input_hypothesis,
                "output_hypothesis": output_hypothesis,
                "chain_judge_parsed": parsed,
                "chain_judge_score": int(parsed.get("score", 0)) if isinstance(parsed, dict) else 0,
                "chain_judge_reason": str(parsed.get("reason", "")) if isinstance(parsed, dict) else "",
                "output_final_score": output_score.get("final_score"),
                "output_support_ratio": output_score.get("support_ratio"),
                "topk_ratio": float(token_change.get("topk_ratio", 0.0) or 0.0),
                "actual_kl": float(token_change.get("actual_kl", 0.0) or 0.0),
                "token_change": token_change,
                "steering_prompts": item.get("steering_prompts", []),
            }
        )

    best_pair_idx = pick_best_pair(pairs)
    payload, path = write_step_payload(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=timestamp,
        step_index=8,
        step_name="build_chain_hypotheses",
        inputs={"step6": step6["round_id"], "step7": step7["round_id"]},
        parameters={
            **vars(args),
            "selection_rule": "max(thresholds_met), tie max(chain_judge_score), tie max(output_final_score)",
        },
        outputs={
            "pairs": pairs,
            "best_pair_idx": best_pair_idx,
            "selection_rule": "max(thresholds_met), tie max(chain_judge_score), tie max(output_final_score)",
            "llm_calls": llm_calls,
            "token_usage": token_counter.as_dict(),
        },
        logs_root=args.logs_root,
    )
    print_step_result(path, {"best_pair_idx": best_pair_idx, "round_id": payload["round_id"]})
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 8: build chain hypotheses and select the best pair.")
    parser.add_argument("--layer-id", required=True)
    parser.add_argument("--feature-id", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--llm-api-key-file", default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--llm-parse-retries", type=int, default=2)
    parser.add_argument("--llm-parse-retry-backoff-seconds", type=float, default=1.5)
    parser.add_argument("--extra-chain-guidance", default=None,
                        help="Additional guidance injected into the chain-judge user prompt.")
    return parser


if __name__ == "__main__":
    run_step(build_arg_parser().parse_args())
