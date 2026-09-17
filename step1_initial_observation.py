from __future__ import annotations

import argparse
import json
from datetime import datetime

from neuronpedia_feature_api import fetch_and_parse_feature_observation
from workflow_step_utils import (
    load_bos_token_observation,
    load_gradient_token_observation,
    print_step_result,
    round_id_for_step,
    step_round_dir,
    write_step_payload,
)


def run_step(args: argparse.Namespace) -> dict:
    layer_id = str(args.layer_id)
    feature_id = str(args.feature_id)
    timestamp = str(args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"))
    round_id = round_id_for_step(1)

    input_source = str(args.input_observation_source)
    if input_source == "neuronpedia":
        observation = fetch_and_parse_feature_observation(
            model_id=str(args.model_id),
            layer_id=layer_id,
            feature_id=feature_id,
            width=str(args.width),
            selection_method=int(args.selection_method),
            m=int(args.observation_m),
            n=int(args.observation_n),
            api_key=args.neuronpedia_api_key,
            timeout=int(args.neuronpedia_timeout),
            timestamp=timestamp,
            round_id=round_id,
            logs_root=args.logs_root,
        )
    elif input_source == "bos_token":
        observation = {
            "input_side_observation": load_bos_token_observation(
                bos_root=str(args.bos_token_root),
                layer_id=layer_id,
                feature_id=feature_id,
                prompt_id=str(args.bos_prompt_id),
            )
        }
    elif input_source == "gradient_token":
        observation = {
            "input_side_observation": load_gradient_token_observation(
                gradient_root=str(args.gradient_token_root),
                layer_id=layer_id,
                feature_id=feature_id,
                prompt_id=str(args.gradient_prompt_id),
            )
        }
    else:  # argparse enforces choices; keep direct callers safe.
        raise ValueError(f"unsupported input observation source: {input_source}")
    observation["input_source"] = input_source

    round_dir = step_round_dir(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=timestamp,
        step_index=1,
        logs_root=args.logs_root,
    )
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / f"layer{layer_id}-feature{feature_id}-observation.json").write_text(
        json.dumps(observation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (round_dir / f"layer{layer_id}-feature{feature_id}-observation-input.json").write_text(
        json.dumps(observation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    payload, path = write_step_payload(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=timestamp,
        step_index=1,
        step_name="initial_observation",
        parameters=vars(args),
        outputs={"observation": observation},
        logs_root=args.logs_root,
    )
    print_step_result(path, {"timestamp": timestamp, "round_id": payload["round_id"]})
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 1: fetch or load the initial feature observation.")
    parser.add_argument("--model-id", default="gemma-2-2b")
    parser.add_argument("--layer-id", required=True)
    parser.add_argument("--feature-id", required=True)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument(
        "--input-observation-source",
        choices=["neuronpedia", "bos_token", "gradient_token"],
        default="neuronpedia",
    )
    parser.add_argument("--bos-token-root", default="initial_observation")
    parser.add_argument("--bos-prompt-id", default="prompt-0001")
    parser.add_argument("--gradient-token-root", default="initial_observation")
    parser.add_argument("--gradient-prompt-id", default="prompt-0001")
    parser.add_argument("--width", default="16k")
    parser.add_argument("--selection-method", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--observation-m", type=int, default=2)
    parser.add_argument("--observation-n", type=int, default=2)
    parser.add_argument("--neuronpedia-api-key", default=None)
    parser.add_argument("--neuronpedia-timeout", type=int, default=30)
    return parser


if __name__ == "__main__":
    run_step(build_arg_parser().parse_args())
