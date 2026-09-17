from __future__ import annotations

import argparse

from function import TokenUsageAccumulator
from workflow_step_utils import (
    DEFAULT_API_KEY_FILE,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL_NAME,
    generate_hypotheses_single_call,
    limit_input_token_observation,
    load_step_payload,
    make_llm_client,
    print_step_result,
    write_step_payload,
)


def run_step(args: argparse.Namespace) -> dict:
    layer_id = str(args.layer_id)
    feature_id = str(args.feature_id)
    timestamp = str(args.timestamp)
    if int(args.input_token_evidence_count) < 0:
        raise ValueError("--input-token-evidence-count must be >= 0.")

    step1 = load_step_payload(layer_id=layer_id, feature_id=feature_id, timestamp=timestamp, step_index=1, logs_root=args.logs_root)
    observation = step1["outputs"]["observation"]
    input_observation = observation.get("input_side_observation", observation)
    input_observation_for_llm = limit_input_token_observation(
        input_observation,
        token_count=int(args.input_token_evidence_count),
    )

    client = make_llm_client(base_url=str(args.llm_base_url), api_key_file=args.llm_api_key_file)
    token_counter = TokenUsageAccumulator()
    hypotheses, llm_call = generate_hypotheses_single_call(
        side="input",
        observation=input_observation_for_llm,
        num_hypothesis=int(args.num_hypothesis),
        client=client,
        model=str(args.llm_model),
        token_counter=token_counter,
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        extra_guidance=args.extra_input_guidance,
    )

    payload, path = write_step_payload(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=timestamp,
        step_index=2,
        step_name="generate_input_hypotheses",
        inputs={"step1": step1["round_id"]},
        parameters=vars(args),
        outputs={
            "input_hypotheses": hypotheses,
            "llm_input_observation_summary": {
                "source": input_observation_for_llm.get("source"),
                "selected_count": input_observation_for_llm.get("selected_count"),
            },
            "llm_calls": [llm_call],
            "token_usage": token_counter.as_dict(),
        },
        logs_root=args.logs_root,
    )
    print_step_result(path, {"hypothesis_count": len(hypotheses), "round_id": payload["round_id"]})
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 2: generate input-side initial hypotheses.")
    parser.add_argument("--layer-id", required=True)
    parser.add_argument("--feature-id", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--num-hypothesis", type=int, default=3)
    parser.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--llm-api-key-file", default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--extra-input-guidance", default=None)
    parser.add_argument(
        "--input-token-evidence-count",
        type=int,
        default=0,
        help=(
            "For bos_token/gradient_token observations, pass only the first N top tokens "
            "to the LLM for input hypothesis generation. Use 0 to pass all available tokens."
        ),
    )
    return parser


if __name__ == "__main__":
    run_step(build_arg_parser().parse_args())
