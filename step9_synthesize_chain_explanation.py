from __future__ import annotations

import argparse
import json
from pathlib import Path

from function import TokenUsageAccumulator
from workflow_step_utils import (
    DEFAULT_API_KEY_FILE,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL_NAME,
    load_step_payload,
    make_llm_client,
    print_step_result,
    round_id_for_step,
    step_round_dir,
    synthesize_chain_explanation,
    write_step_payload,
)


def _slim_observation(obs: dict) -> dict:
    # step1 stores activation_examples under input_side_observation; fallback for pre-converted format
    input_side = obs.get("input_side_observation") or {}
    raw_activations = (
        input_side.get("activation_examples")
        or obs.get("input_top_activations")
        or obs.get("top_activations")
        or []
    )
    slim = {
        "input_source": input_side.get("source") or input_side.get("input_source") or obs.get("input_source"),
        "input_top_activations": [
            {
                "sentence": a.get("sentence"),
                "max_token": a.get("max_token"),
                "max_value": a.get("maxValue") or a.get("max_value"),
            }
            for a in raw_activations
        ],
    }
    bos_meta = input_side.get("bos_token_scan_meta")
    if isinstance(bos_meta, dict) and bos_meta:
        slim["bos_token_scan_meta"] = dict(bos_meta)
    return slim


def _slim_input_eval_hypothesis(h: dict) -> dict:
    return {
        "hypothesis_index": h.get("hypothesis_index"),
        "hypothesis": h.get("hypothesis"),
        "score_non_zero_rate": h.get("score_non_zero_rate"),
        "score_boundary_non_activation_rate": h.get("score_boundary_non_activation_rate"),
        "sentence_results": [
            {"sentence": r.get("sentence"), "is_non_zero": r.get("is_non_zero")}
            for r in h.get("sentence_results", [])
        ],
        "boundary_sentence_results": [
            {"sentence": r.get("sentence"), "is_non_zero": r.get("is_non_zero")}
            for r in h.get("boundary_sentence_results", [])
        ],
    }


def _slim_output_eval(step6_rows: list, step7_per_hyp: list) -> list:
    score_by_idx = {int(r.get("idx", 0)): r for r in step7_per_hyp}
    result = []
    for row in step6_rows:
        idx = int(row.get("hypothesis_index", 0))
        tc = row.get("token_change") or {}
        score_row = score_by_idx.get(idx, {})
        result.append({
            "idx": idx,
            "input_hypothesis": row.get("input_hypothesis"),
            "output_hypothesis": row.get("output_hypothesis"),
            "intervention_scope": tc.get("intervention_scope"),
            "max_activation_scale": tc.get("max_activation_scale"),
            "last_token_scale": tc.get("last_token_scale"),
            "steering_prompts": row.get("steering_prompts") or [],
            "topk_positive_tokens": [
                t.get("token") for t in (tc.get("topk_positive_tokens") or [])[:15]
            ],
            "topk_negative_tokens": [
                t.get("token") for t in (tc.get("topk_negative_tokens") or [])[:15]
            ],
            "output_score": score_row.get("final_score"),
            "matched_tokens": [
                t.get("token") for t in (score_row.get("matched_tokens") or [])
            ],
            "match_reason": score_row.get("llm_reason"),
        })
    return result


def _slim_chain_pairs(pairs: list) -> list:
    return [
        {
            "idx": p.get("idx"),
            "input_hypothesis": p.get("input_hypothesis"),
            "output_hypothesis": p.get("output_hypothesis"),
            "chain_judge_score": p.get("chain_judge_score"),
            "chain_judge_reason": p.get("chain_judge_reason"),
            "output_score": p.get("output_final_score"),
        }
        for p in pairs
    ]


def _aggregate_token_cost(step_payloads: dict) -> dict:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for payload in step_payloads.values():
        usage = (payload.get("outputs") or {}).get("token_usage") or {}
        for k in total:
            total[k] += int(usage.get(k, 0) or 0)
    return total


def _build_trace(
    *,
    layer_id: str,
    feature_id: str,
    timestamp: str,
    step_payloads: dict,
    chain_explanation: str,
) -> dict:
    step1 = step_payloads[1]
    step2 = step_payloads[2]
    step4 = step_payloads[4]
    step6 = step_payloads[6]
    step7 = step_payloads[7]
    step8 = step_payloads[8]
    return {
        "meta": {
            "layer_id": layer_id,
            "feature_id": feature_id,
            "timestamp": timestamp,
            "token_cost": _aggregate_token_cost(step_payloads),

        },
        # steps 1-4: observation → hypotheses → experiment design → activation scoring
        "input_round": {
            "observation": _slim_observation(step1["outputs"].get("observation", {})),
            "hypotheses": step2["outputs"].get("input_hypotheses", []),
            "eval": {
                "per_hypothesis": [
                    _slim_input_eval_hypothesis(h)
                    for h in step4["outputs"].get("hypothesis_results", [])
                ],
            },
        },
        # steps 5-7: intervention → output hypotheses → output scoring
        "output_round": {
            "per_hypothesis": _slim_output_eval(
                step6["outputs"].get("token_change_by_hypothesis", []),
                step7["outputs"].get("per_hypothesis", []),
            ),
        },
        # steps 8-9: chain pair selection + synthesis
        "chain": {
            "best_pair_idx": step8["outputs"].get("best_pair_idx"),
            "chain_explanation": chain_explanation,
            "pairs": _slim_chain_pairs(step8["outputs"].get("pairs", [])),
        },
    }


def run_step(args: argparse.Namespace) -> dict:
    layer_id = str(args.layer_id)
    feature_id = str(args.feature_id)
    timestamp = str(args.timestamp)

    step_payloads = {
        idx: load_step_payload(layer_id=layer_id, feature_id=feature_id, timestamp=timestamp, step_index=idx, logs_root=args.logs_root)
        for idx in range(1, 9)
    }
    pairs = step_payloads[8]["outputs"]["pairs"]
    best_pair_idx = int(step_payloads[8]["outputs"].get("best_pair_idx", 1))
    if best_pair_idx < 1 or best_pair_idx > len(pairs):
        raise ValueError(f"Invalid best_pair_idx={best_pair_idx}; pair_count={len(pairs)}")
    best_pair = pairs[best_pair_idx - 1]

    client = make_llm_client(base_url=str(args.llm_base_url), api_key_file=args.llm_api_key_file)
    token_counter = TokenUsageAccumulator()
    chain_explanation, llm_call = synthesize_chain_explanation(
        best_pair=best_pair,
        client=client,
        model=str(args.llm_model),
        token_counter=token_counter,
        max_retries=args.llm_parse_retries,
        retry_backoff_seconds=args.llm_parse_retry_backoff_seconds,
    )

    trace = _build_trace(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=timestamp,
        step_payloads=step_payloads,
        chain_explanation=chain_explanation,
    )

    round_dir = step_round_dir(layer_id=layer_id, feature_id=feature_id, timestamp=timestamp, step_index=9, logs_root=args.logs_root)
    round_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = round_dir.parent
    trace_path = trace_dir / "trace.json"
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    payload, path = write_step_payload(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=timestamp,
        step_index=9,
        step_name="synthesize_chain_explanation",
        inputs={f"step{idx}": payload["round_id"] for idx, payload in step_payloads.items()},
        parameters=vars(args),
        outputs={
            "best_pair_idx": best_pair_idx,
            "best_pair": best_pair,
            "chain_explanation": chain_explanation,
            "trace_json": str(Path(trace_path)),
            "llm_calls": [llm_call],
            "token_usage": token_counter.as_dict(),
        },
        logs_root=args.logs_root,
    )
    print_step_result(
        path,
        {"best_pair_idx": best_pair_idx, "trace_json": str(trace_path), "round_id": payload["round_id"]},
    )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 9: synthesize the final <=50-word chain explanation.")
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
