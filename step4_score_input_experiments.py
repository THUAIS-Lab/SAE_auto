from __future__ import annotations

import argparse

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from experiments_execution_input import execute_input_side_experiments
from inference_client import (
    DEFAULT_INFERENCE_SERVER_URL,
    DEFAULT_INFERENCE_TIMEOUT_SEC,
    InferenceServerError,
    call_score_input_experiments,
)
from workflow_step_utils import (
    DEFAULT_CANONICAL_MAP_PATH,
    DEFAULT_MODEL_CHECKPOINT_PATH,
    DEFAULT_SAE_ROOT,
    load_model_with_sae,
    load_step_payload,
    print_step_result,
    resolve_sae_path,
    write_step_payload,
)

_SERVICE_ARG_NAMES = {"inference_server_url", "inference_timeout_sec", "no_inference_server"}


def _step_parameters(args: argparse.Namespace) -> dict:
    return {key: value for key, value in vars(args).items() if key not in _SERVICE_ARG_NAMES}


def run_step(args: argparse.Namespace) -> dict:
    layer_id = str(args.layer_id)
    feature_id = str(args.feature_id)
    timestamp = str(args.timestamp)
    step3 = load_step_payload(layer_id=layer_id, feature_id=feature_id, timestamp=timestamp, step_index=3, logs_root=args.logs_root)
    experiments = step3["outputs"]["input_side_experiments"]

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
        result = execute_input_side_experiments(
            input_side_experiments=experiments,
            module=module,
            non_zero_threshold=float(args.non_zero_threshold),
            max_activation_scale=float(args.max_activation_scale),
        )
    else:
        try:
            result = call_score_input_experiments(
                server_url=str(args.inference_server_url),
                timeout_sec=float(args.inference_timeout_sec),
                layer_id=int(layer_id),
                feature_id=int(feature_id),
                input_side_experiments=experiments,
                non_zero_threshold=float(args.non_zero_threshold),
                max_activation_scale=float(args.max_activation_scale),
            )
        except InferenceServerError as exc:
            raise RuntimeError(str(exc)) from exc

    result.pop("runtime_batches", None)

    payload, path = write_step_payload(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=timestamp,
        step_index=4,
        step_name="score_input_experiments",
        inputs={"step3": step3["round_id"]},
        parameters={**_step_parameters(args), "resolved_sae_path": sae_path},
        outputs=result,
        logs_root=args.logs_root,
    )
    print_step_result(
        path,
        {
            "overall_score_non_zero_rate": result.get("overall_score_non_zero_rate"),
            "round_id": payload["round_id"],
        },
    )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    default_device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Step 4: score input-side experiments.")
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
    parser.add_argument("--non-zero-threshold", type=float, default=0.0)
    parser.add_argument("--max-activation-scale", type=float, default=2.0)
    parser.add_argument("--inference-server-url", default=DEFAULT_INFERENCE_SERVER_URL)
    parser.add_argument("--inference-timeout-sec", type=float, default=DEFAULT_INFERENCE_TIMEOUT_SEC)
    parser.add_argument("--no-inference-server", action="store_true")
    return parser


if __name__ == "__main__":
    run_step(build_arg_parser().parse_args())
