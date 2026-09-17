from __future__ import annotations

import argparse
import json
from typing import List, Optional

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from inference_client import (
    DEFAULT_INFERENCE_SERVER_URL,
    DEFAULT_INFERENCE_TIMEOUT_SEC,
    InferenceServerError,
    call_run_intervention,
)
from workflow_step_utils import (
    DEFAULT_CANONICAL_MAP_PATH,
    DEFAULT_MODEL_CHECKPOINT_PATH,
    DEFAULT_SAE_ROOT,
    build_output_observation_from_tokenchange,
    compute_tokenchange,
    load_model_with_sae,
    load_step_payload,
    print_step_result,
    resolve_sae_path,
    select_steering_prompts,
    write_step_payload,
)

_SERVICE_ARG_NAMES = {"inference_server_url", "inference_timeout_sec", "no_inference_server"}


def _step_parameters(args: argparse.Namespace) -> dict:
    return {key: value for key, value in vars(args).items() if key not in _SERVICE_ARG_NAMES}


def run_step(args: argparse.Namespace) -> dict:
    layer_id = str(args.layer_id)
    feature_id = str(args.feature_id)
    timestamp = str(args.timestamp)

    step1 = load_step_payload(layer_id=layer_id, feature_id=feature_id, timestamp=timestamp, step_index=1, logs_root=args.logs_root)
    step3 = load_step_payload(layer_id=layer_id, feature_id=feature_id, timestamp=timestamp, step_index=3, logs_root=args.logs_root)
    input_side_experiments = step3["outputs"]["input_side_experiments"]

    custom_steering_prompts: Optional[List[str]] = None
    if args.custom_steering_prompts:
        custom_steering_prompts = json.loads(args.custom_steering_prompts)

    sae_path = resolve_sae_path(
        layer_id=layer_id,
        width=str(args.width),
        sae_path=args.sae_path,
        sae_root=str(args.sae_root),
        sae_average_l0=args.sae_average_l0,
        sae_canonical_map=str(args.sae_canonical_map),
        sae_release=str(args.sae_release),
        use_sae_lens_uri=bool(args.use_sae_lens_uri),
    )

    if bool(args.no_inference_server):
        module = load_model_with_sae(
            model_checkpoint_path=str(args.model_checkpoint_path),
            sae_path=sae_path,
            layer_id=layer_id,
            feature_id=feature_id,
            device=str(args.device),
        )

        rows = []
        for idx, item in enumerate(input_side_experiments, start=1):
            input_hypothesis = str(item.get("hypothesis", "")).strip()
            steering_prompts = select_steering_prompts(
                experiment_item=item,
                fallback_prompts=args.prompts,
                max_prompts=int(args.max_steering_prompts),
                custom_prompts=custom_steering_prompts,
            )
            token_change = compute_tokenchange(
                module=module,
                prompts=steering_prompts,
                feature_id=int(feature_id),
                top_k=int(args.top_k),
                intervention_scope=str(args.intervention_scope),
                max_activation_scale=float(args.max_activation_scale),
                last_token_scale=float(args.last_token_scale),
            )
            output_observation = build_output_observation_from_tokenchange(
                token_change,
                steering_prompts=steering_prompts,
                top_k=int(args.top_k),
            )
            rows.append(
                {
                    "hypothesis_index": idx,
                    "input_hypothesis": input_hypothesis,
                    "steering_prompts": steering_prompts,
                    "output_observation": output_observation,
                    "token_change": token_change,
                }
            )
    else:
        try:
            result = call_run_intervention(
                server_url=str(args.inference_server_url),
                timeout_sec=float(args.inference_timeout_sec),
                layer_id=int(layer_id),
                feature_id=int(feature_id),
                input_side_experiments=input_side_experiments,
                prompts=[str(prompt) for prompt in args.prompts],
                top_k=int(args.top_k),
                max_steering_prompts=int(args.max_steering_prompts),
                intervention_scope=str(args.intervention_scope),
                max_activation_scale=float(args.max_activation_scale),
                last_token_scale=float(args.last_token_scale),
                custom_steering_prompts=custom_steering_prompts,
            )
        except InferenceServerError as exc:
            raise RuntimeError(str(exc)) from exc
        rows = result.get("token_change_by_hypothesis")
        if not isinstance(rows, list):
            raise RuntimeError("Inference server response missing list field 'token_change_by_hypothesis'.")

    payload, path = write_step_payload(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=timestamp,
        step_index=5,
        step_name="run_intervention",
        inputs={"step1": step1["round_id"], "step3": step3["round_id"]},
        parameters={
            **_step_parameters(args),
            "resolved_sae_path": sae_path,
            "clamp_value_source": "step3_steering_prompt_max_activation",
        },
        outputs={"token_change_by_hypothesis": rows},
        logs_root=args.logs_root,
    )
    print_step_result(path, {"intervention_count": len(rows), "round_id": payload["round_id"]})
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    default_device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Step 5: run SAE intervention on input-designed sentences.")
    parser.add_argument("--layer-id", required=True)
    parser.add_argument("--feature-id", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--model-checkpoint-path", default=DEFAULT_MODEL_CHECKPOINT_PATH)
    parser.add_argument("--sae-path", default=None)
    parser.add_argument("--sae-root", default=DEFAULT_SAE_ROOT)
    parser.add_argument("--sae-release", default="gemma-scope-2b-pt-res")
    parser.add_argument("--width", default="16k")
    parser.add_argument("--sae-average-l0", default=None)
    parser.add_argument("--sae-canonical-map", default=str(DEFAULT_CANONICAL_MAP_PATH))
    parser.add_argument("--use-sae-lens-uri", action="store_true")
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--prompts", nargs="*", default=["The explanation is simple:", "I think", "We"])
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-steering-prompts", type=int, default=5)
    parser.add_argument(
        "--intervention-scope",
        choices=["last_token_only", "all_tokens", "max_activation_token"],
        default="max_activation_token",
    )
    parser.add_argument("--max-activation-scale", type=float, default=2.0)
    parser.add_argument("--last-token-scale", type=float, default=1.0)
    parser.add_argument("--custom-steering-prompts", default=None)
    parser.add_argument("--inference-server-url", default=DEFAULT_INFERENCE_SERVER_URL)
    parser.add_argument("--inference-timeout-sec", type=float, default=DEFAULT_INFERENCE_TIMEOUT_SEC)
    parser.add_argument("--no-inference-server", action="store_true")
    return parser


if __name__ == "__main__":
    run_step(build_arg_parser().parse_args())
