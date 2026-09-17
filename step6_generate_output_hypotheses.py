from __future__ import annotations

import argparse

from function import TokenUsageAccumulator
from workflow_step_utils import (
    DEFAULT_API_KEY_FILE,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL_NAME,
    LLMParseFailure,
    generate_output_hypothesis,
    load_step_payload,
    make_llm_client,
    print_step_result,
    write_step_failure_payload,
    write_step_payload,
)


def run_step(args: argparse.Namespace) -> dict:
    layer_id = str(args.layer_id)
    feature_id = str(args.feature_id)
    timestamp = str(args.timestamp)

    step5 = load_step_payload(layer_id=layer_id, feature_id=feature_id, timestamp=timestamp, step_index=5, logs_root=args.logs_root)
    rows = step5["outputs"]["token_change_by_hypothesis"]

    client = make_llm_client(base_url=str(args.llm_base_url), api_key_file=args.llm_api_key_file)
    token_counter = TokenUsageAccumulator()
    out_rows = []
    llm_calls = []
    for idx, item in enumerate(rows, start=1):
        try:
            output_hypothesis, call = generate_output_hypothesis(
                observation=item.get("output_observation", {}),
                client=client,
                model=str(args.llm_model),
                token_counter=token_counter,
                extra_guidance=args.extra_output_guidance,
                temperature=float(args.temperature),
                max_tokens=int(args.max_tokens),
            )
        except Exception as exc:
            failed_calls = list(llm_calls)
            if isinstance(exc, LLMParseFailure):
                failed_calls.extend(exc.llm_calls)
            _, failure_path = write_step_failure_payload(
                layer_id=layer_id,
                feature_id=feature_id,
                timestamp=timestamp,
                step_index=6,
                step_name="generate_output_hypotheses",
                inputs={"step5": step5["round_id"]},
                parameters=vars(args),
                error=exc,
                llm_calls=failed_calls,
                partial_outputs={
                    "completed_rows": out_rows,
                    "failed_index": idx,
                    "failed_input_hypothesis": item.get("input_hypothesis", ""),
                    "token_usage": token_counter.as_dict(),
                },
                logs_root=args.logs_root,
            )
            raise RuntimeError(f"Step 6 failed; failure details saved to {failure_path}") from exc
        call["index"] = idx
        call["input_hypothesis"] = item.get("input_hypothesis", "")
        llm_calls.append(call)
        out_row = dict(item)
        out_row["output_hypothesis"] = output_hypothesis
        out_rows.append(out_row)

    payload, path = write_step_payload(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=timestamp,
        step_index=6,
        step_name="generate_output_hypotheses",
        inputs={"step5": step5["round_id"]},
        parameters=vars(args),
        outputs={
            "token_change_by_hypothesis": out_rows,
            "output_hypotheses": [row.get("output_hypothesis", "") for row in out_rows],
            "llm_calls": llm_calls,
            "token_usage": token_counter.as_dict(),
        },
        logs_root=args.logs_root,
    )
    print_step_result(path, {"hypothesis_count": len(out_rows), "round_id": payload["round_id"]})
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 6: generate output-side hypotheses from intervention results.")
    parser.add_argument("--layer-id", required=True)
    parser.add_argument("--feature-id", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--llm-api-key-file", default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--extra-output-guidance", default=None)
    return parser


if __name__ == "__main__":
    run_step(build_arg_parser().parse_args())
