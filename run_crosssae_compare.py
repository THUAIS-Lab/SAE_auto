#!/usr/bin/env python3
"""
run_crosssae_compare.py — Cross-SAE validation: 16k skill library → 65k SAE.

Two sub-experiments:
  B.1 Robustness : full pipeline on 65k, compare metric distributions to 16k baseline.
  B.2 Transfer   : same 65k features, cold-start agent vs skill-init agent.

Round-namespace design to avoid conflicts:
  init        → logs/layer-N/feature-F/{ts}/trace.json
  run-cold    → agent starts from {ts}, creates {ts}_r1, {ts}_r2, …
  run-skill   → copies {ts}/ to {ts}_s/, agent starts from {ts}_s,
                creates {ts}_s_r1, {ts}_s_r2, … (no overlap with cold)

Usage (split across two GPUs):
  # GPU 0 — layers 0, 6, 12
  python run_crosssae_compare.py all --timestamp TS --layers 0 6 12 --device cuda:0

  # GPU 1 — layers 18, 24
  python run_crosssae_compare.py all --timestamp TS --layers 18 24 --device cuda:1

Or step by step:
  python run_crosssae_compare.py init      --timestamp TS --layers 0 6 12 --device cuda:0
  python run_crosssae_compare.py run-cold  --timestamp TS --layers 0 6 12 --device cuda:0
  python run_crosssae_compare.py run-skill --timestamp TS --layers 0 6 12 --device cuda:0
  python run_crosssae_compare.py report    --timestamp TS   # CPU only, run once for all layers
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

PYTHON = sys.executable
CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))

from support_info.llm_api_info import (
    api_key_file as DEFAULT_API_KEY_FILE,
    base_url as DEFAULT_BASE_URL,
    model_name as DEFAULT_MODEL_NAME,
)
from workflow_step_utils import build_layer_sae_paths

LAYER_IDS = [0, 6, 12, 18, 24]
N_FEATURES_PER_LAYER = 50
DEFAULT_SAE_ROOT = os.environ.get(
    "SAE_ROOT",
    "gemma-scope-2b-pt-res",
)
SAE_PATHS_65K = build_layer_sae_paths(layer_ids=LAYER_IDS, width="65k", sae_root=DEFAULT_SAE_ROOT)
DEFAULT_MODEL_PATH = os.environ.get(
    "SAE_MODEL_CHECKPOINT_PATH",
    "google/gemma-2-2b",
)
DEFAULT_SKILLS_DIR = CODE_DIR / "skills"
COLD_SKILLS_DIR = CODE_DIR / "skills_cold_65k"   # always-empty dir for cold-start


def _configured_sae_paths(args: argparse.Namespace) -> Dict[int, str]:
    return getattr(args, "sae_paths_65k", SAE_PATHS_65K)


def _skill_ts(ts: str) -> str:
    """Timestamp used as the starting point for the skill-init agent run."""
    return f"{ts}_s"


def _get_feature_ids(layer: int, n: int) -> List[int]:
    rng = random.Random(layer * 31337 + 42)
    candidates = list(range(1000, 64000, 1300))
    return sorted(rng.sample(candidates, min(n, len(candidates))))


def _log(msg: str, log_file: Optional[Path] = None) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")


def _run_step(cmd: List[str], *, label: str, log_file: Optional[Path] = None) -> bool:
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=str(CODE_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        if log_file:
            with open(log_file, "a") as f:
                f.write(line)
    proc.wait()
    ok = proc.returncode == 0
    _log(f"{label} {'OK' if ok else 'FAILED'} ({time.time()-t0:.1f}s)", log_file)
    return ok


# ── init ──────────────────────────────────────────────────────────────────────

def _run_initial_pipeline(layer, fid, ts, sae_path, model_path, device,
                          llm_base_url, llm_model, llm_api_key_file, log_file) -> bool:
    base = ["--layer-id", str(layer), "--feature-id", str(fid), "--timestamp", ts]
    llm  = ["--llm-base-url", llm_base_url, "--llm-model", llm_model]
    if llm_api_key_file:
        llm += ["--llm-api-key-file", llm_api_key_file]
    steps = [
        ("step1", [PYTHON, "step1_initial_observation.py",
                   "--model-id", "gemma-2-2b", "--layer-id", str(layer),
                   "--feature-id", str(fid), "--width", "65k", "--timestamp", ts,
                   "--observation-m", "10", "--observation-n", "5"]),
        ("step2", [PYTHON, "step2_generate_input_hypotheses.py",
                   *base, "--num-hypothesis", "3", *llm]),
        ("step3", [PYTHON, "step3_design_input_experiments.py",
                   *base, "--num-sentences-per-hypothesis", "5", *llm]),
        ("step4", [PYTHON, "step4_score_input_experiments.py",
                   *base, "--width", "65k", "--sae-path", sae_path,
                   "--model-checkpoint-path", model_path, "--device", device]),
        ("step5", [PYTHON, "step5_run_intervention.py",
                   *base, "--width", "65k", "--sae-path", sae_path,
                   "--model-checkpoint-path", model_path, "--device", device,
                   "--top-k", "30", "--intervention-scope", "max_activation_token"]),
        ("step6", [PYTHON, "step6_generate_output_hypotheses.py", *base, *llm]),
        ("step7", [PYTHON, "step7_score_output_hypotheses.py",    *base, *llm]),
        ("step8", [PYTHON, "step8_build_chain_hypotheses.py",     *base, *llm]),
        ("step9", [PYTHON, "step9_synthesize_chain_explanation.py", *base, *llm]),
    ]
    label = f"L{layer}-F{fid}"
    for name, cmd in steps:
        if not _run_step(cmd, label=f"{label}/{name}", log_file=log_file):
            return False
    trace = CODE_DIR / "logs" / f"layer-{layer}" / f"feature-{fid}" / ts / "trace.json"
    return trace.exists()


def cmd_init(args: argparse.Namespace) -> None:
    log_file = CODE_DIR / f"crosssae_init_{args.timestamp}_gpu{args.device.replace(':','')}.log"
    _log(f"=== INIT: 65k SAE, ts={args.timestamp}, device={args.device} ===", log_file)
    ok = fail = skip = 0
    for layer in (args.layers or LAYER_IDS):
        sae_path = _configured_sae_paths(args)[layer]
        if not Path(sae_path).exists():
            _log(f"Layer {layer}: 65k SAE missing at {sae_path}", log_file)
            continue
        for fid in _get_feature_ids(layer, args.n_features):
            trace = CODE_DIR / "logs" / f"layer-{layer}" / f"feature-{fid}" / args.timestamp / "trace.json"
            if trace.exists() and not args.force:
                skip += 1; continue
            if _run_initial_pipeline(layer, fid, args.timestamp, sae_path,
                                     args.model_path, args.device,
                                     args.llm_base_url, args.llm_model,
                                     args.llm_api_key_file, log_file):
                ok += 1
            else:
                fail += 1
    _log(f"Init done: {ok} OK, {skip} skipped, {fail} failed", log_file)


# ── agent batch (shared by cold + skill) ──────────────────────────────────────


def _compress_cases_to_skills(skills_dir: Path, client, model: str, log_file: Path) -> None:
    """LLM-distill auto cases into general skill principles."""
    import fcntl
    for gate in ["input", "output", "chain"]:
        cases_file = skills_dir / f"skill_{gate}_cases.md"
        skill_file  = skills_dir / f"skill_{gate}.md"
        if not cases_file.exists() or not skill_file.exists():
            continue
        cases_text = cases_file.read_text(encoding="utf-8")
        auto_count  = cases_text.count("## [Auto]")
        if auto_count < 3:
            continue
        skill_text = skill_file.read_text(encoding="utf-8")
        prompt = (
            "You are updating a skill reference for an AI agent that proposes harness configs "
            "for SAE feature interpretation.\n\n"
            f"Current skill file ({gate} side):\n```\n{skill_text}\n```\n\n"
            f"New case examples (focus on [Auto] entries):\n```\n{cases_text[-6000:]}\n```\n\n"
            "Extract 2-4 concise, general principles from the [Auto] cases NOT already covered. "
            "Only include patterns that appear in ≥2 cases or are clearly generalizable.\n"
            "Format as markdown sections:\n"
            "## [Pattern Name]\n**When to apply**: ...\n**What to change**: ...\n**Why**: ...\n\n"
            "Return ONLY new sections. If no new patterns, return exactly: NO_NEW_PATTERNS"
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=1200,
            )
            text = resp.choices[0].message.content.strip()
            if text and text != "NO_NEW_PATTERNS":
                with open(skill_file, "a", encoding="utf-8") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    try:
                        f.write(f"\n\n<!-- auto-distilled from {auto_count} cases -->\n")
                        f.write(text + "\n")
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)
                n = text.count("##")
                _log(f"[skill_compress] {gate}: +{n} pattern(s) from {auto_count} cases", log_file)
            else:
                _log(f"[skill_compress] {gate}: no new patterns ({auto_count} cases)", log_file)
        except Exception as e:
            _log(f"[skill_compress] {gate} failed: {e}", log_file)

def _run_agent_batch(args, *, initial_ts: str, skills_dir: Path, label: str) -> None:
    from agent_runner import run_agent_loop

    log_file = CODE_DIR / f"crosssae_{label.lower()}_{args.timestamp}_gpu{args.device.replace(':','')}.log"
    _log(f"=== {label}: initial_ts={initial_ts}, skills={skills_dir.name} ===", log_file)
    summary: Dict[str, str] = {}

    for layer in (args.layers or LAYER_IDS):
        sae_path = _configured_sae_paths(args)[layer]
        for fid in _get_feature_ids(layer, args.n_features):
            key = f"L{layer}-F{fid}"
            trace = CODE_DIR / "logs" / f"layer-{layer}" / f"feature-{fid}" / initial_ts / "trace.json"
            if not trace.exists():
                summary[key] = "SKIP"; continue
            try:
                r = run_agent_loop(
                    layer_id=str(layer), feature_id=str(fid),
                    initial_timestamp=initial_ts,
                    sae_path=sae_path, model_path=args.model_path,
                    llm_base_url=args.llm_base_url, llm_model=args.llm_model,
                    llm_api_key_file=args.llm_api_key_file,
                    device=args.device, model_id="gemma-2-2b",
                    max_rounds=args.max_rounds, score_threshold=4,
                    log_file=log_file, skills_dir=skills_dir, sae_width="65k",
                )
                score  = r.get("final_chain_score", 0)
                rounds = r.get("rounds_run", 0)
                tokens = r.get("agent_token_cost", {}).get("total_tokens", 0)
                summary[key] = f"OK score={score} rounds={rounds} tokens={tokens}"
            except Exception as e:
                summary[key] = f"ERROR: {e}"

    out = {"label": label, "base_timestamp": args.timestamp,
           "initial_ts": initial_ts, "results": summary}
    suffix = f"gpu{args.device.replace(':','')}"
    out_path = CODE_DIR / f"crosssae_{label.lower()}_{args.timestamp}_{suffix}_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    # Distill accumulated cases into general skill principles
    if skills_dir != COLD_SKILLS_DIR:
        try:
            from openai import OpenAI as _OAI
            _c = _OAI(api_key=os.environ.get("LLM_API_KEY",""), base_url=args.llm_base_url)
            _compress_cases_to_skills(skills_dir, _c, args.llm_model, log_file)
        except Exception as _ce:
            _log(f"[skill_compress] skipped: {_ce}", log_file)
    ok = sum(1 for s in summary.values() if s.startswith("OK"))
    _log(f"{label} done: {ok}/{len(summary)} OK → {out_path}", log_file)


def cmd_run_cold(args: argparse.Namespace) -> None:
    COLD_SKILLS_DIR.mkdir(exist_ok=True)
    for f in COLD_SKILLS_DIR.glob("*.md"):
        f.unlink()
    # cold uses the original initial timestamp directly
    _run_agent_batch(args, initial_ts=args.timestamp,
                     skills_dir=COLD_SKILLS_DIR, label="COLD")


def cmd_run_skill(args: argparse.Namespace) -> None:
    if not DEFAULT_SKILLS_DIR.exists():
        _log(f"ERROR: skills dir not found: {DEFAULT_SKILLS_DIR}")
        sys.exit(1)
    skill_ts = _skill_ts(args.timestamp)
    # Copy each feature's initial trace to the skill namespace so rounds don't conflict
    _log(f"Copying initial traces to skill namespace {skill_ts} …")
    for layer in (args.layers or LAYER_IDS):
        for fid in _get_feature_ids(layer, args.n_features):
            src = CODE_DIR / "logs" / f"layer-{layer}" / f"feature-{fid}" / args.timestamp
            dst = CODE_DIR / "logs" / f"layer-{layer}" / f"feature-{fid}" / skill_ts
            if src.exists() and not dst.exists():
                shutil.copytree(str(src), str(dst))
    _run_agent_batch(args, initial_ts=skill_ts,
                     skills_dir=DEFAULT_SKILLS_DIR, label="SKILL")


# ── report ────────────────────────────────────────────────────────────────────

def cmd_report(args: argparse.Namespace) -> None:
    # Merge all GPU shards for each condition
    def _load_all(label: str) -> List[Dict]:
        parsed = []
        for path in CODE_DIR.glob(f"crosssae_{label.lower()}_{args.timestamp}_*_summary.json"):
            data = json.loads(path.read_text())
            for key, status in data["results"].items():
                if not status.startswith("OK"):
                    continue
                layer = int(key.split("-")[0][1:])
                parts = {"key": key, "layer": layer}
                for tok in status.split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        try: parts[k] = float(v) if "." in v else int(v)
                        except ValueError: parts[k] = v
                parsed.append(parts)
        return parsed

    cold  = _load_all("cold")
    skill = _load_all("skill")

    if not cold or not skill:
        _log("ERROR: no summary files found. Run run-cold and run-skill first.")
        sys.exit(1)

    def avg(lst, f):
        v = [x[f] for x in lst if f in x]
        return sum(v)/len(v) if v else 0.0

    def gate(lst, thr, f="score"):
        v = [x[f] for x in lst if f in x]
        return sum(1 for x in v if x >= thr)/len(v) if v else 0.0

    lines = [
        "=== Cross-SAE Report: 16k skills → 65k SAE ===",
        f"Timestamp: {args.timestamp}",
        f"Cold-start OK: {len(cold)}   Skill-init OK: {len(skill)}",
        "",
        f"{'Metric':<38} {'Cold':>8} {'Skill':>8} {'Δ':>8}",
        "-"*64,
        f"{'Avg rounds':<38} {avg(cold,'rounds'):>8.2f} {avg(skill,'rounds'):>8.2f} {avg(skill,'rounds')-avg(cold,'rounds'):>+8.2f}",
        f"{'Avg tokens (k)':<38} {avg(cold,'tokens')/1e3:>8.1f} {avg(skill,'tokens')/1e3:>8.1f} {(avg(skill,'tokens')-avg(cold,'tokens'))/1e3:>+8.1f}",
        f"{'Chain ≥4 rate':<38} {gate(cold,4,'score'):>8.1%} {gate(skill,4,'score'):>8.1%} {gate(skill,4,'score')-gate(cold,4,'score'):>+8.1%}",
        "", "Per-layer (avg rounds):",
    ]
    for layer in LAYER_IDS:
        c = [x for x in cold  if x["layer"]==layer]
        s = [x for x in skill if x["layer"]==layer]
        cr, sr = avg(c,"rounds"), avg(s,"rounds")
        lines.append(f"  Layer {layer:2d}: cold={cr:.2f}  skill={sr:.2f}  Δ={sr-cr:+.2f}  (n={len(c)}/{len(s)})")

    text = "\n".join(lines)
    print(text)
    out = CODE_DIR / f"crosssae_report_{args.timestamp}.txt"
    out.write_text(text+"\n")
    _log(f"Report → {out}")


def cmd_all(args: argparse.Namespace) -> None:
    cmd_init(args)
    cmd_run_cold(args)
    cmd_run_skill(args)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["init","run-cold","run-skill","report","all"])
    p.add_argument("--timestamp", required=True)
    p.add_argument("--n-features", type=int, default=N_FEATURES_PER_LAYER)
    p.add_argument("--layers", nargs="*", type=int, default=None)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--sae-root", default=DEFAULT_SAE_ROOT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-rounds", type=int, default=5)
    p.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--llm-model", default=DEFAULT_MODEL_NAME)
    p.add_argument("--llm-api-key-file", default=DEFAULT_API_KEY_FILE)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    args.sae_paths_65k = build_layer_sae_paths(
        layer_ids=LAYER_IDS,
        width="65k",
        sae_root=args.sae_root,
    )

    {"init":cmd_init,"run-cold":cmd_run_cold,
     "run-skill":cmd_run_skill,"report":cmd_report,"all":cmd_all}[args.command](args)


if __name__ == "__main__":
    main()
