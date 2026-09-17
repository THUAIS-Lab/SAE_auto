from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


LAYER_PATTERN = re.compile(r"^layer_(\d+)$")
FEATURE_PATTERN = re.compile(r"^feature_(\d+)$")
DEFAULT_LAYERS = [0, 6, 12, 18, 24]


def discover_sage_features(results_root: Path) -> Dict[int, List[int]]:
    root = Path(results_root)
    if not root.is_dir():
        raise FileNotFoundError(f"SAGE results root not found: {root}")
    layers: Dict[int, List[int]] = {}
    for layer_dir in sorted(root.iterdir(), key=lambda path: path.name):
        match = LAYER_PATTERN.fullmatch(layer_dir.name)
        if not layer_dir.is_dir() or match is None:
            continue
        layer_id = int(match.group(1))
        feature_ids = []
        for feature_dir in layer_dir.iterdir():
            feature_match = FEATURE_PATTERN.fullmatch(feature_dir.name)
            if feature_dir.is_dir() and feature_match is not None:
                feature_ids.append(int(feature_match.group(1)))
        if feature_ids:
            layers[layer_id] = sorted(set(feature_ids))
    if not layers:
        raise ValueError(f"no layer_<L>/feature_<F> directories found under {root}")
    return layers


def validate_feature_map(
    layers: Dict[int, List[int]],
    *,
    expected_layers: Optional[Iterable[int]] = None,
    expected_total: Optional[int] = None,
    expected_per_layer: Optional[int] = None,
) -> None:
    if expected_layers is not None:
        wanted = sorted(int(value) for value in expected_layers)
        if sorted(layers) != wanted:
            raise ValueError(f"expected layers {wanted}, found {sorted(layers)}")
    if expected_per_layer is not None:
        bad = {layer: len(ids) for layer, ids in layers.items() if len(ids) != int(expected_per_layer)}
        if bad:
            raise ValueError(f"expected {expected_per_layer} features per layer, got {bad}")
    total = sum(len(ids) for ids in layers.values())
    if expected_total is not None and total != int(expected_total):
        raise ValueError(f"expected {expected_total} total features, found {total}")
    invalid = [feature_id for ids in layers.values() for feature_id in ids if not 0 <= feature_id < 16384]
    if invalid:
        raise ValueError(f"feature ids must be in [0, 16383], got {invalid[:20]}")


def write_feature_manifest(results_root: Path, output_path: Path, layers: Dict[int, List[int]]) -> Dict[str, object]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_files = {}
    for layer_id, feature_ids in sorted(layers.items()):
        feature_file = output.with_name(f"{output.stem}.layer-{layer_id}.txt")
        feature_file.write_text("\n".join(str(value) for value in feature_ids) + "\n", encoding="utf-8")
        feature_files[str(layer_id)] = feature_file.name
    payload: Dict[str, object] = {
        "version": 1,
        "source_root": str(Path(results_root).resolve()),
        "total_features": sum(len(ids) for ids in layers.values()),
        "layers": {str(layer): ids for layer, ids in sorted(layers.items())},
        "feature_id_files": feature_files,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def load_feature_manifest(path: Path) -> Dict[int, List[int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_layers = payload.get("layers") if isinstance(payload, dict) else None
    if not isinstance(raw_layers, dict):
        raise ValueError("feature manifest must contain an object-valued 'layers' field")
    layers = {
        int(layer): sorted(set(int(feature_id) for feature_id in feature_ids))
        for layer, feature_ids in raw_layers.items()
        if isinstance(feature_ids, list)
    }
    validate_feature_map(layers)
    return layers


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a BOS experiment feature manifest from SAGE results.")
    parser.add_argument("--sage-results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-layers", nargs="*", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--expected-total", type=int, default=100)
    parser.add_argument("--expected-per-layer", type=int, default=20)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    layers = discover_sage_features(args.sage_results_root)
    validate_feature_map(
        layers,
        expected_layers=args.expected_layers,
        expected_total=args.expected_total,
        expected_per_layer=args.expected_per_layer,
    )
    payload = write_feature_manifest(args.sage_results_root, args.output, layers)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
