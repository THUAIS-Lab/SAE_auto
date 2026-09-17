from __future__ import annotations

import argparse

from function import TokenUsageAccumulator
from workflow_step_utils import (
    DEFAULT_API_KEY_FILE,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL_NAME,
    LLMParseFailure,
    design_boundary_sentences_for_input,
    design_sentences_for_input,
    extract_trigger_tokens,
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

    step1 = load_step_payload(layer_id=layer_id, feature_id=feature_id, timestamp=timestamp, step_index=1, logs_root=args.logs_root)
    step2 = load_step_payload(layer_id=layer_id, feature_id=feature_id, timestamp=timestamp, step_index=2, logs_root=args.logs_root)
    observation = step1["outputs"]["observation"]
    hypotheses = list(step2["outputs"]["input_hypotheses"])

    client = make_llm_client(base_url=str(args.llm_base_url), api_key_file=args.llm_api_key_file)
    token_counter = TokenUsageAccumulator()
    sentence_calls = []
    boundary_calls = []
    trigger_tokens = extract_trigger_tokens(observation)
    try:
        input_sentences, sentence_calls = design_sentences_for_input(
            hypotheses=hypotheses,
            num_sentences=int(args.num_sentences_per_hypothesis),
            client=client,
            model=str(args.llm_model),
            token_counter=token_counter,
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            observation=observation,
        )
        boundary_sentences, boundary_calls = design_boundary_sentences_for_input(
            hypotheses=hypotheses,
            num_sentences=int(args.num_sentences_per_hypothesis),
            client=client,
            model=str(args.llm_model),
            token_counter=token_counter,
            max_tokens=int(args.max_tokens),
            trigger_tokens=trigger_tokens,
        )
    except Exception as exc:
        llm_calls = sentence_calls + boundary_calls
        if isinstance(exc, LLMParseFailure):
            llm_calls = sentence_calls + boundary_calls + exc.llm_calls
        _, failure_path = write_step_failure_payload(
            layer_id=layer_id,
            feature_id=feature_id,
            timestamp=timestamp,
            step_index=3,
            step_name="design_input_experiments",
            inputs={"step1": step1["round_id"], "step2": step2["round_id"]},
            parameters=vars(args),
            error=exc,
            llm_calls=llm_calls,
            partial_outputs={
                "trigger_tokens": trigger_tokens,
                "token_usage": token_counter.as_dict(),
            },
            logs_root=args.logs_root,
        )
        raise RuntimeError(f"Step 3 failed; failure details saved to {failure_path}") from exc

    input_side_experiments = [
        {
            "hypothesis_index": idx,
            "hypothesis": hyp,
            "designed_sentences": designed,
            "boundary_sentences": boundary,
        }
        for idx, (hyp, designed, boundary) in enumerate(
            zip(hypotheses, input_sentences, boundary_sentences),
            start=1,
        )
    ]

    payload, path = write_step_payload(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=timestamp,
        step_index=3,
        step_name="design_input_experiments",
        inputs={"step1": step1["round_id"], "step2": step2["round_id"]},
        parameters=vars(args),
        outputs={
            "input_side_experiments": input_side_experiments,
            "trigger_tokens": trigger_tokens,
            "llm_calls": sentence_calls + boundary_calls,
            "token_usage": token_counter.as_dict(),
        },
        logs_root=args.logs_root,
    )
    print_step_result(path, {"experiment_count": len(input_side_experiments), "round_id": payload["round_id"]})
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 3: design input-side activation and boundary experiments.")
    parser.add_argument("--layer-id", required=True)
    parser.add_argument("--feature-id", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--num-sentences-per-hypothesis", type=int, default=5)
    parser.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--llm-api-key-file", default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=10000)
    return parser


if __name__ == "__main__":
    run_step(build_arg_parser().parse_args())
