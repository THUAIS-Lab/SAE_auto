#!/usr/bin/env python3
"""
batch_run_agent.py — Run Phase 3 agent loop on all features from a previous batch run.

Discovers all trace.json files under a given initial timestamp and runs the
agent loop on each.

Usage:
  python batch_run_agent.py --initial-timestamp 20260420_175328 [options]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from openai import OpenAI

from agent_runner import run_agent_loop
from function import read_api_key
from skill_compact import compact_skill_files, learn_from_batch
from workflow_step_utils import build_layer_sae_paths
from support_info.llm_api_info import (
    api_key_file as DEFAULT_API_KEY_FILE,
    base_url as DEFAULT_BASE_URL,
    model_name as DEFAULT_MODEL_NAME,
)

CODE_DIR = Path(__file__).parent

LAYER_IDS = [0, 6, 12, 18, 24]
DEFAULT_SAE_ROOT = os.environ.get(
    "SAE_ROOT",
    "gemma-scope-2b-pt-res",
)
SAE_PATHS = build_layer_sae_paths(layer_ids=LAYER_IDS, width="16k", sae_root=DEFAULT_SAE_ROOT)
DEFAULT_MODEL_PATH = os.environ.get(
    "SAE_MODEL_CHECKPOINT_PATH",
    "google/gemma-2-2b",
)


def _discover_features(initial_timestamp: str) -> List[Tuple[str, str]]:
    """Return (layer_id, feature_id) pairs that have a trace.json for the given timestamp."""
    logs = CODE_DIR / "logs"
    found = []
    for trace_path in sorted(logs.glob(f"layer-*/feature-*/{initial_timestamp}/trace.json")):
        parts = trace_path.parts
        layer_part = [p for p in parts if p.startswith("layer-")]
        feature_part = [p for p in parts if p.startswith("feature-")]
        if layer_part and feature_part:
            layer_id = layer_part[-1].replace("layer-", "")
            feature_id = feature_part[-1].replace("feature-", "")
            found.append((layer_id, feature_id))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Phase 3 agent loop.")
    parser.add_argument("--initial-timestamp", required=True)
    parser.add_argument("--model-id", default="gemma-2-2b")
    parser.add_argument("--model-checkpoint-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sae-root", default=DEFAULT_SAE_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--llm-api-key-file", default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--score-threshold", type=int, default=4)
    parser.add_argument("--input-activation-threshold", type=float, default=0.8)
    parser.add_argument("--input-boundary-threshold", type=float, default=0.8)
    parser.add_argument("--output-score-threshold", type=float, default=0.5)
    # Optional filter
    parser.add_argument("--layer-ids", nargs="*", type=int, default=None,
                        help="Only process these layer IDs (default: all)")
    parser.add_argument("--skip-skill-compact", action="store_true",
                        help="Skip skill compaction and batch learning after the run")
    args = parser.parse_args()
    sae_paths = build_layer_sae_paths(layer_ids=LAYER_IDS, width="16k", sae_root=args.sae_root)

    features = _discover_features(args.initial_timestamp)
    if args.layer_ids:
        features = [(l, f) for l, f in features if int(l) in args.layer_ids]

    if not features:
        print(f"No trace.json found for timestamp={args.initial_timestamp}")
        sys.exit(1)

    log_file = CODE_DIR / f"batch_agent_{args.initial_timestamp}_{datetime.now().strftime('%H%M%S')}.log"
    print(f"Found {len(features)} features to process. Log: {log_file}")

    results: Dict[str, str] = {}
    for layer_id, feature_id in features:
        key = f"L{layer_id:>2}-F{feature_id:>5}"
        sae_path = sae_paths.get(int(layer_id))
        if not sae_path:
            print(f"{key}: no SAE path configured, skipping")
            results[key] = "SKIP"
            continue
        try:
            summary = run_agent_loop(
                layer_id=layer_id,
                feature_id=feature_id,
                initial_timestamp=args.initial_timestamp,
                sae_path=sae_path,
                model_path=args.model_checkpoint_path,
                llm_base_url=args.llm_base_url,
                llm_model=args.llm_model,
                llm_api_key_file=args.llm_api_key_file,
                device=args.device,
                model_id=args.model_id,
                max_rounds=args.max_rounds,
                score_threshold=args.score_threshold,
                input_activation_threshold=args.input_activation_threshold,
                input_boundary_threshold=args.input_boundary_threshold,
                output_score_threshold=args.output_score_threshold,
                log_file=log_file,
            )
            score = summary.get("final_chain_score", 0)
            rounds = summary.get("rounds_run", 0)
            results[key] = f"OK score={score}/5 rounds={rounds}"
        except Exception as e:
            results[key] = f"ERROR: {e}"

    print("\n=== Batch Agent Summary ===")
    for key, status in results.items():
        print(f"  {key}: {status}")
    ok = sum(1 for s in results.values() if s.startswith("OK"))
    print(f"\n{ok}/{len(results)} succeeded")

    if not args.skip_skill_compact:
        print("\n=== Post-batch skill maintenance ===")
        try:
            api_key = read_api_key(args.llm_api_key_file)
            client = OpenAI(base_url=args.llm_base_url, api_key=api_key)
            print("Learning cross-feature patterns from batch...")
            learn_from_batch(
                client, args.llm_model,
                args.initial_timestamp,
                args.layer_ids,
            )
            print("Compacting cases files...")
            compact_skill_files(client, args.llm_model)
        except Exception as e:
            print(f"  Skill maintenance failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
