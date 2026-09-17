from __future__ import annotations

import argparse
import json
from pathlib import Path

from input_bos_token_scan import DEFAULT_CANONICAL_MAP_PATH, run_scan
from prepare_bos_feature_manifest import load_feature_manifest
from run_main_experiment import SAE_PATHS_16K
from workflow_step_utils import DEFAULT_MODEL_CHECKPOINT_PATH


CODE_DIR = Path(__file__).resolve().parent


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else CODE_DIR / path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect one BOS initial-observation prompt for one manifest layer.")
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--layer-id", type=int, required=True, choices=sorted(SAE_PATHS_16K))
    parser.add_argument("--model-checkpoint-path", default=DEFAULT_MODEL_CHECKPOINT_PATH)
    parser.add_argument("--sae-path", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompt-template", default="<bos>")
    parser.add_argument("--prompt-id", default="prompt-0001")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--activation-threshold", type=float, default=0.0)
    parser.add_argument("--feature-chunk-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-special-tokens", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest_path = _resolve(args.feature_manifest)
    output_root = _resolve(args.output_root)
    layers = load_feature_manifest(manifest_path)
    if args.layer_id not in layers:
        raise ValueError(f"layer {args.layer_id} is absent from {manifest_path}")
    selection_dir = output_root / "manifests"
    selection_dir.mkdir(parents=True, exist_ok=True)
    feature_ids_file = selection_dir / f"feature_ids_layer-{args.layer_id}.txt"
    feature_ids_file.write_text(
        "\n".join(str(value) for value in layers[args.layer_id]) + "\n",
        encoding="utf-8",
    )
    scan_args = argparse.Namespace(
        model_checkpoint_path=str(args.model_checkpoint_path),
        layer_id=int(args.layer_id),
        sae_path=str(args.sae_path or SAE_PATHS_16K[args.layer_id]),
        sae_release="gemma-scope-2b-pt-res",
        width="16k",
        sae_average_l0=None,
        sae_canonical_map=str(_resolve(DEFAULT_CANONICAL_MAP_PATH)),
        feature_ids_file=str(feature_ids_file),
        all_features=False,
        manual_token_texts_file=None,
        random_sample_size=0,
        scan_full_vocab=True,
        include_special_tokens=bool(args.include_special_tokens),
        seed=int(args.seed),
        prompt_template=str(args.prompt_template),
        prompt_id=str(args.prompt_id),
        batch_size=int(args.batch_size),
        top_k=int(args.top_k),
        activation_threshold=float(args.activation_threshold),
        feature_chunk_size=int(args.feature_chunk_size),
        save_all_activated=False,
        output_root=str(output_root),
        device=str(args.device),
    )
    summary = run_scan(scan_args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
