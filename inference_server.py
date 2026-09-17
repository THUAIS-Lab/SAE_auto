from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from experiments_execution_input import execute_input_side_experiments
from input_bos_token_scan import scan_tokens_with_module
from model_with_sae import ModelWithSAEModule, load_model, load_sae, load_tokenizer
from workflow_step_utils import (
    DEFAULT_MODEL_CHECKPOINT_PATH,
    DEFAULT_SAE_ROOT,
    build_layer_sae_paths,
    build_output_observation_from_tokenchange,
    compute_tokenchange,
    select_steering_prompts,
)


DEFAULT_SAE_PATHS_16K: Dict[int, str] = build_layer_sae_paths(
    layer_ids=[0, 6, 12, 18, 24],
    width="16k",
    sae_root=DEFAULT_SAE_ROOT,
)

ENV_HOST = "SAE_INFERENCE_HOST"
ENV_PORT = "SAE_INFERENCE_PORT"
ENV_MODEL_PATH = "SAE_INFERENCE_MODEL_CHECKPOINT_PATH"
ENV_DEVICE = "SAE_INFERENCE_DEVICE"
ENV_MAX_GPU_JOBS = "SAE_INFERENCE_MAX_GPU_JOBS"
ENV_SAE_PATHS = "SAE_INFERENCE_SAE_PATHS"
ENV_SAE_ROOT = "SAE_INFERENCE_SAE_ROOT"
ENV_LOG_PATH = "SAE_INFERENCE_LOG_PATH"
ENV_LOG_LEVEL = "SAE_INFERENCE_LOG_LEVEL"
ENV_WORKER_HEALTHCHECK_TIMEOUT = "SAE_INFERENCE_TIMEOUT_WORKER_HEALTHCHECK"


class ScoreInputRequest(BaseModel):
    layer_id: int
    feature_id: int
    input_side_experiments: List[Dict[str, Any]]
    non_zero_threshold: float = 0.0
    max_activation_scale: float = 2.0


class ActivationTraceRequest(BaseModel):
    layer_id: int
    feature_id: int
    texts: List[str] = Field(min_length=1)


class BosTokenScanRequest(BaseModel):
    layer_id: int
    feature_ids: List[int] = Field(min_length=1)
    prompt_template: str
    prompt_id: str
    candidate_token_texts: List[str] = Field(default_factory=list)
    scan_full_vocab: bool = False
    random_sample_size: int = Field(default=0, ge=0)
    top_k: int = Field(default=50, ge=1)
    batch_size: int = Field(default=64, ge=1)
    activation_threshold: float = 0.0
    include_special_tokens: bool = False
    seed: int = 42
    feature_chunk_size: int = Field(default=4096, ge=1)


class RunInterventionRequest(BaseModel):
    layer_id: int
    feature_id: int
    input_side_experiments: List[Dict[str, Any]]
    prompts: List[str] = Field(default_factory=lambda: ["The explanation is simple:", "I think", "We"])
    top_k: int = 30
    max_steering_prompts: int = 5
    intervention_scope: str = "max_activation_token"
    max_activation_scale: float = 2.0
    last_token_scale: float = 1.0
    custom_steering_prompts: Optional[List[str]] = None


class OutputCentricObservationRequest(BaseModel):
    """Deterministic inputs for the original VocabProj/TokenChange methods."""

    layer_id: int
    feature_id: int
    token_id_batches: List[List[int]] = Field(min_length=1)
    projection_top_k: int = Field(default=50, ge=1)
    token_change_top_k: int = Field(default=10, ge=1)
    clamp_value: float = 10.0

class InferenceState:
    def __init__(self) -> None:
        self.model_checkpoint_path = DEFAULT_MODEL_CHECKPOINT_PATH
        self.device = "cuda"
        self.max_gpu_jobs = 1
        self.sae_paths: Dict[int, str] = dict(DEFAULT_SAE_PATHS_16K)
        self.log_path = ""
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.saes: Dict[int, Dict[str, Any]] = {}
        self.semaphore: Optional[asyncio.Semaphore] = None
        self.ready = False

    def configure_from_env(self) -> None:
        self.model_checkpoint_path = os.environ.get(ENV_MODEL_PATH, DEFAULT_MODEL_CHECKPOINT_PATH)
        self.device = os.environ.get(ENV_DEVICE, self.device)
        self.max_gpu_jobs = max(1, int(os.environ.get(ENV_MAX_GPU_JOBS, "1")))
        self.sae_paths = _read_sae_paths_from_env()
        self.log_path = os.environ.get(ENV_LOG_PATH, _default_log_path())
        self.semaphore = asyncio.Semaphore(self.max_gpu_jobs)

    def load(self) -> None:
        if self.ready:
            return

        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(f"Requested device={self.device}, but CUDA is not available.")
            cuda_device = torch.device(self.device)
            if cuda_device.index is not None:
                torch.cuda.set_device(cuda_device.index)

        _emit(f"loading model={self.model_checkpoint_path} device={self.device}")
        _log_gpu_snapshot("before_model_load", self.device)
        self.model = load_model(
            self.model_checkpoint_path,
            self.device,
            use_hooked_transformer=False,
        )
        if self.model is None:
            raise RuntimeError(f"Failed to load model: {self.model_checkpoint_path}")
        _log_gpu_snapshot("after_model_load", self.device)

        self.tokenizer = load_tokenizer(self.model_checkpoint_path)
        if self.tokenizer is None:
            raise RuntimeError(f"Failed to load tokenizer: {self.model_checkpoint_path}")

        loaded_saes: Dict[int, Dict[str, Any]] = {}
        for layer, sae_path in sorted(self.sae_paths.items()):
            _emit(f"loading SAE layer={layer} path={sae_path}")
            sae = load_sae(sae_path, self.device)
            if not sae:
                raise RuntimeError(f"Failed to load SAE for layer {layer}: {sae_path}")
            loaded_saes[int(layer)] = sae
            _log_gpu_snapshot(f"after_sae_layer_{layer}", self.device)
        self.saes = loaded_saes
        self.ready = True
        _emit(f"ready pid={os.getpid()} layers={sorted(self.saes)} max_gpu_jobs={self.max_gpu_jobs}")
        _log_gpu_snapshot("ready", self.device)

    def require_ready(self) -> None:
        if not self.ready or self.model is None or self.tokenizer is None or not self.saes:
            raise HTTPException(status_code=503, detail="Inference server is not ready.")

    def module_for(self, *, layer_id: int, feature_id: int) -> ModelWithSAEModule:
        self.require_ready()
        layer = int(layer_id)
        if layer not in self.saes:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported layer_id={layer}. Available layers: {sorted(self.saes)}",
            )
        return ModelWithSAEModule(
            llm_name=self.model_checkpoint_path,
            sae_path=self.sae_paths[layer],
            sae_layer=layer,
            feature_index=int(feature_id),
            device=self.device,
            model=self.model,
            tokenizer=self.tokenizer,
            sae=self.saes[layer],
        )

    def health(self) -> Dict[str, Any]:
        logging.info("health pid=%s ready=%s", os.getpid(), self.ready)
        return {
            "ready": self.ready,
            "pid": os.getpid(),
            "model_checkpoint_path": self.model_checkpoint_path,
            "device": self.device,
            "max_gpu_jobs": self.max_gpu_jobs,
            "log_path": self.log_path,
            "loaded_layers": sorted(self.saes),
            "gpu_memory": _gpu_memory_snapshot(self.device),
        }


state = InferenceState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ = app
    state.configure_from_env()
    _setup_logging(state.log_path)
    logging.info(
        "startup_begin pid=%s model=%s device=%s max_gpu_jobs=%s layers=%s",
        os.getpid(),
        state.model_checkpoint_path,
        state.device,
        state.max_gpu_jobs,
        sorted(state.sae_paths),
    )
    try:
        await run_in_threadpool(state.load)
    except Exception:
        logging.exception("startup_failed pid=%s", os.getpid())
        raise
    try:
        yield
    finally:
        logging.info("shutdown pid=%s", os.getpid())


app = FastAPI(title="SAE Auto Inference Server", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, Any]:
    return state.health()


@app.post("/score-input-experiments")
async def score_input_experiments(request: ScoreInputRequest) -> Dict[str, Any]:
    start = time.perf_counter()
    logging.info(
        "request_begin endpoint=score-input-experiments pid=%s layer=%s feature=%s hypotheses=%s",
        os.getpid(),
        request.layer_id,
        request.feature_id,
        len(request.input_side_experiments),
    )
    semaphore = _require_semaphore()
    try:
        async with semaphore:
            result = await run_in_threadpool(_score_input_experiments_sync, request)
    except Exception:
        logging.exception(
            "request_failed endpoint=score-input-experiments pid=%s layer=%s feature=%s",
            os.getpid(),
            request.layer_id,
            request.feature_id,
        )
        raise
    finally:
        _clear_cuda_cache(state.device, "score-input-experiments")
    logging.info(
        "request_done endpoint=score-input-experiments pid=%s layer=%s feature=%s elapsed_sec=%.3f",
        os.getpid(),
        request.layer_id,
        request.feature_id,
        time.perf_counter() - start,
    )
    return result


@app.post("/activation-traces")
async def activation_traces(request: ActivationTraceRequest) -> Dict[str, Any]:
    semaphore = _require_semaphore()
    try:
        async with semaphore:
            return await run_in_threadpool(_activation_traces_sync, request)
    except Exception:
        logging.exception("request_failed endpoint=activation-traces layer=%s feature=%s", request.layer_id, request.feature_id)
        raise
    finally:
        _clear_cuda_cache(state.device, "activation-traces")


@app.post("/bos-token-scan")
async def bos_token_scan(request: BosTokenScanRequest) -> Dict[str, Any]:
    start = time.perf_counter()
    logging.info(
        "request_begin endpoint=bos-token-scan pid=%s layer=%s features=%s",
        os.getpid(),
        request.layer_id,
        request.feature_ids,
    )
    semaphore = _require_semaphore()
    try:
        async with semaphore:
            result = await run_in_threadpool(_bos_token_scan_sync, request)
    except Exception:
        logging.exception(
            "request_failed endpoint=bos-token-scan layer=%s features=%s",
            request.layer_id,
            request.feature_ids,
        )
        raise
    finally:
        _clear_cuda_cache(state.device, "bos-token-scan")
    logging.info(
        "request_done endpoint=bos-token-scan pid=%s layer=%s features=%s elapsed_sec=%.3f",
        os.getpid(),
        request.layer_id,
        request.feature_ids,
        time.perf_counter() - start,
    )
    return result


@app.post("/run-intervention")
async def run_intervention(request: RunInterventionRequest) -> Dict[str, Any]:
    start = time.perf_counter()
    logging.info(
        "request_begin endpoint=run-intervention pid=%s layer=%s feature=%s hypotheses=%s",
        os.getpid(),
        request.layer_id,
        request.feature_id,
        len(request.input_side_experiments),
    )
    semaphore = _require_semaphore()
    try:
        async with semaphore:
            result = await run_in_threadpool(_run_intervention_sync, request)
    except Exception:
        logging.exception(
            "request_failed endpoint=run-intervention pid=%s layer=%s feature=%s",
            os.getpid(),
            request.layer_id,
            request.feature_id,
        )
        raise
    finally:
        _clear_cuda_cache(state.device, "run-intervention")
    logging.info(
        "request_done endpoint=run-intervention pid=%s layer=%s feature=%s elapsed_sec=%.3f",
        os.getpid(),
        request.layer_id,
        request.feature_id,
        time.perf_counter() - start,
    )
    return result



@app.post("/output-centric-observation")
async def output_centric_observation(request: OutputCentricObservationRequest) -> Dict[str, Any]:
    """Return VP and fixed +/- clamp TC evidence in one serialized GPU job."""
    start = time.perf_counter()
    semaphore = _require_semaphore()
    try:
        async with semaphore:
            return await run_in_threadpool(_output_centric_observation_sync, request)
    except Exception:
        logging.exception(
            "request_failed endpoint=output-centric-observation layer=%s feature=%s",
            request.layer_id,
            request.feature_id,
        )
        raise
    finally:
        _clear_cuda_cache(state.device, "output-centric-observation")
        logging.info(
            "request_done endpoint=output-centric-observation layer=%s feature=%s elapsed_sec=%.3f",
            request.layer_id,
            request.feature_id,
            time.perf_counter() - start,
        )
def _score_input_experiments_sync(request: ScoreInputRequest) -> Dict[str, Any]:
    module = state.module_for(layer_id=request.layer_id, feature_id=request.feature_id)
    result = execute_input_side_experiments(
        input_side_experiments=request.input_side_experiments,
        module=module,
        non_zero_threshold=float(request.non_zero_threshold),
        max_activation_scale=float(request.max_activation_scale),
    )
    result.pop("runtime_batches", None)
    return result


def _activation_traces_sync(request: ActivationTraceRequest) -> Dict[str, Any]:
    module = state.module_for(layer_id=request.layer_id, feature_id=request.feature_id)
    enc = module.tokenizer(
        list(request.texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        add_special_tokens=True,
    )
    batch = module.analyze_feature_batch_from_tensors(
        input_ids=enc["input_ids"],
        attention_mask=enc.get("attention_mask"),
    )
    return {"traces": batch.get("traces", [])}


def _bos_token_scan_sync(request: BosTokenScanRequest) -> Dict[str, Any]:
    module = state.module_for(
        layer_id=int(request.layer_id),
        feature_id=int(request.feature_ids[0]),
    )
    return scan_tokens_with_module(
        module=module,
        layer_id=int(request.layer_id),
        requested_feature_ids=[int(value) for value in request.feature_ids],
        prompt_template=str(request.prompt_template),
        prompt_id=str(request.prompt_id),
        manual_token_texts=[str(value) for value in request.candidate_token_texts],
        scan_full_vocab=bool(request.scan_full_vocab),
        random_sample_size=int(request.random_sample_size),
        include_special_tokens=bool(request.include_special_tokens),
        seed=int(request.seed),
        batch_size=int(request.batch_size),
        top_k=int(request.top_k),
        activation_threshold=float(request.activation_threshold),
        feature_chunk_size=int(request.feature_chunk_size),
        show_progress=False,
    )


def _run_intervention_sync(request: RunInterventionRequest) -> Dict[str, Any]:
    if request.intervention_scope not in {"last_token_only", "all_tokens", "max_activation_token"}:
        raise HTTPException(
            status_code=400,
            detail="intervention_scope must be one of: last_token_only, all_tokens, max_activation_token",
        )

    module = state.module_for(layer_id=request.layer_id, feature_id=request.feature_id)
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(request.input_side_experiments, start=1):
        input_hypothesis = str(item.get("hypothesis", "")).strip()
        steering_prompts = select_steering_prompts(
            experiment_item=item,
            fallback_prompts=request.prompts,
            max_prompts=int(request.max_steering_prompts),
            custom_prompts=request.custom_steering_prompts,
        )
        token_change = compute_tokenchange(
            module=module,
            prompts=steering_prompts,
            feature_id=int(request.feature_id),
            top_k=int(request.top_k),
            intervention_scope=str(request.intervention_scope),
            max_activation_scale=float(request.max_activation_scale),
            last_token_scale=float(request.last_token_scale),
        )
        output_observation = build_output_observation_from_tokenchange(
            token_change,
            steering_prompts=steering_prompts,
            top_k=int(request.top_k),
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
    return {"token_change_by_hypothesis": rows}



def _decoder_row(module: ModelWithSAEModule, feature_id: int) -> torch.Tensor:
    sae = module.sae
    if not isinstance(sae, dict):
        raise RuntimeError("VocabProj requires dictionary-backed local SAE weights.")
    weight = sae.get("W_dec")
    if weight is None:
        weight = sae.get("decoder.weight")
    if not isinstance(weight, torch.Tensor):
        raise RuntimeError("Loaded SAE has no decoder weight matrix.")
    if not 0 <= int(feature_id) < int(weight.shape[0]):
        raise ValueError(f"feature_id={feature_id} outside decoder width={weight.shape[0]}.")
    return weight[int(feature_id)]


def _vp_observation(module: ModelWithSAEModule, feature_id: int, top_k: int) -> Dict[str, Any]:
    """Notebook formula: unembed(ln_final(sae.W_dec[feature]))."""
    model = module.model
    final_norm = getattr(getattr(model, "model", None), "norm", None)
    lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else getattr(model, "lm_head", None)
    if final_norm is None or lm_head is None:
        raise RuntimeError("Could not locate final norm and LM head for VocabProj.")
    vector = _decoder_row(module, feature_id).to(device=module.device)
    try:
        vector = vector.to(dtype=next(model.parameters()).dtype)
    except StopIteration:
        pass
    logits = lm_head(final_norm(vector.unsqueeze(0))).squeeze(0).float()
    count = min(int(top_k), int(logits.numel()))
    top_values, top_ids = torch.topk(logits, k=count)
    bottom_values, bottom_ids = torch.topk(logits, k=count, largest=False)

    def rows(ids: torch.Tensor, values: torch.Tensor) -> List[Dict[str, Any]]:
        id_list = [int(x) for x in ids.detach().cpu().tolist()]
        tokens = module.tokenizer.convert_ids_to_tokens(id_list)
        return [
            {"token_id": token_id, "token": str(token), "logit": float(value)}
            for token_id, token, value in zip(id_list, tokens, values.detach().cpu().tolist())
        ]

    return {
        "source": "vocab_projection",
        "formula": "lm_head(final_norm(sae.W_dec[feature_id]))",
        "top_tokens": rows(top_ids, top_values),
        "bottom_tokens": rows(bottom_ids, bottom_values),
    }


def _mean_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=logits.device, dtype=logits.dtype).unsqueeze(-1)
    return (logits * weights).sum(dim=(0, 1)) / weights.sum().clamp_min(1.0)


def _tc_rows(module: ModelWithSAEModule, values: torch.Tensor, ids: torch.Tensor, source: str, sign: float) -> List[Dict[str, Any]]:
    id_list = [int(x) for x in ids.detach().cpu().tolist()]
    tokens = module.tokenizer.convert_ids_to_tokens(id_list)
    return [
        {
            "token_id": token_id, "token": str(token), "delta": float(sign * value),
            "delta_abs": abs(float(value)), "raw_delta": float(value), "source": source,
        }
        for token_id, token, value in zip(id_list, tokens, values.detach().cpu().tolist())
    ]


def _output_centric_observation_sync(request: OutputCentricObservationRequest) -> Dict[str, Any]:
    sequences = [[int(token) for token in row] for row in request.token_id_batches]
    if not sequences or any(not row for row in sequences):
        raise ValueError("token_id_batches may not contain an empty sequence.")
    lengths = {len(row) for row in sequences}
    if len(lengths) != 1:
        raise ValueError("token_id_batches must be rectangular fixed-length Pile chunks.")
    module = state.module_for(layer_id=request.layer_id, feature_id=request.feature_id)
    input_ids = torch.tensor(sequences, dtype=torch.long, device=module.device)
    mask = torch.ones_like(input_ids, dtype=torch.long, device=module.device)
    clean = module.run_logits(input_ids=input_ids, attention_mask=mask)
    plus = module.run_logits_with_feature_intervention(
        input_ids=input_ids, attention_mask=mask, feature_index=int(request.feature_id),
        value=float(request.clamp_value), mode="clamp", intervention_scope="all_tokens",
    )
    minus = module.run_logits_with_feature_intervention(
        input_ids=input_ids, attention_mask=mask, feature_index=int(request.feature_id),
        value=-float(request.clamp_value), mode="clamp", intervention_scope="all_tokens",
    )
    plus_delta, minus_delta = _mean_logits(plus - clean, mask), _mean_logits(minus - clean, mask)
    count = min(int(request.token_change_top_k), int(plus_delta.numel()))
    plus_hi, plus_hi_ids = torch.topk(plus_delta, k=count)
    plus_lo, plus_lo_ids = torch.topk(plus_delta, k=count, largest=False)
    minus_hi, minus_hi_ids = torch.topk(minus_delta, k=count)
    minus_lo, minus_lo_ids = torch.topk(minus_delta, k=count, largest=False)

    # The released Notebook's `real`: + low, - high, + high, - low.
    legacy_ids = [
        *[int(x) for x in plus_lo_ids.detach().cpu().tolist()],
        *[int(x) for x in minus_hi_ids.detach().cpu().tolist()],
        *[int(x) for x in plus_hi_ids.detach().cpu().tolist()],
        *[int(x) for x in minus_lo_ids.detach().cpu().tolist()],
    ]
    token_change = {
        "source": "output_centric_token_change",
        "intervention_scope": "all_tokens",
        "intervention_direction": "fixed_plus_minus_clamp",
        "clamp_value": float(request.clamp_value),
        "token_change_formula": "mean_over_batch_and_positions(logits(clamp(+v))-logits(clean)); same for -v",
        # Both clamp directions give promotion evidence for increased feature
        # activation. The other two groups are suppression evidence.
        "topk_positive_tokens": _tc_rows(module, plus_hi, plus_hi_ids, "plus_clamp_increase", 1.0)
        + _tc_rows(module, minus_lo, minus_lo_ids, "minus_clamp_decrease", -1.0),
        "topk_negative_tokens": _tc_rows(module, plus_lo, plus_lo_ids, "plus_clamp_decrease", 1.0)
        + _tc_rows(module, minus_hi, minus_hi_ids, "minus_clamp_increase", -1.0),
        "legacy_token_list": [str(token) for token in module.tokenizer.convert_ids_to_tokens(legacy_ids)],
    }
    return {
        "vocab_projection": _vp_observation(module, int(request.feature_id), int(request.projection_top_k)),
        "token_change": token_change,
        "tc_input": {"batch_size": len(sequences), "sequence_length": next(iter(lengths))},
    }

def _require_semaphore() -> asyncio.Semaphore:
    if state.semaphore is None:
        raise HTTPException(status_code=503, detail="Inference server is not configured.")
    return state.semaphore


def _read_sae_paths_from_env() -> Dict[int, str]:
    raw = os.environ.get(ENV_SAE_PATHS)
    if not raw:
        return dict(DEFAULT_SAE_PATHS_16K)
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{ENV_SAE_PATHS} must be a JSON object mapping layer ids to paths.") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{ENV_SAE_PATHS} must be a JSON object mapping layer ids to paths.")
    sae_paths: Dict[int, str] = {}
    for layer, path in decoded.items():
        sae_paths[int(layer)] = str(path)
    return sae_paths


def _gpu_memory_snapshot(device: str) -> Dict[str, Any]:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return {"available": False}
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "available": True,
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 2),
            "free_mb": round(free_bytes / 1024**2, 2),
            "total_mb": round(total_bytes / 1024**2, 2),
        }
    except Exception as exc:
        return {"available": True, "error": str(exc)}


def _log_gpu_snapshot(stage: str, device: str) -> None:
    logging.info(
        "gpu_snapshot stage=%s pid=%s snapshot=%s",
        stage,
        os.getpid(),
        json.dumps(_gpu_memory_snapshot(device), sort_keys=True),
    )


def _clear_cuda_cache(device: str, stage: str) -> None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        _log_gpu_snapshot(f"after_cache_clear_{stage}", device)
    except Exception:
        logging.exception("cache_clear_failed stage=%s pid=%s", stage, os.getpid())


def _emit(message: str) -> None:
    formatted = f"[inference-server] {message}"
    print(formatted, flush=True)
    logging.info(formatted)


def _default_log_path() -> str:
    port = os.environ.get(ENV_PORT, "8008")
    return str(Path(__file__).resolve().parent / "logs" / f"inference_server_{port}.log")


def _setup_logging(log_path: str) -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    level_name = os.environ.get(ENV_LOG_LEVEL, "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s pid=%(process)d %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    logging.info("logging_ready path=%s", log_path)


def _parse_sae_layer_paths(
    items: Optional[List[str]],
    sae_root: str = DEFAULT_SAE_ROOT,
) -> Dict[int, str]:
    sae_paths = build_layer_sae_paths(
        layer_ids=[0, 6, 12, 18, 24],
        width="16k",
        sae_root=sae_root,
    )
    if not items:
        return sae_paths
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected --sae-layer-path in LAYER=PATH format, got: {item}")
        layer_text, path = item.split("=", 1)
        sae_paths[int(layer_text.strip())] = path.strip()
    return sae_paths


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve Gemma + Gemma-Scope SAE inference endpoints.")
    parser.add_argument("--host", default=os.environ.get(ENV_HOST, "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get(ENV_PORT, "8008")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SAE_INFERENCE_WORKERS", "1")))
    parser.add_argument("--model-checkpoint-path", default=os.environ.get(ENV_MODEL_PATH, DEFAULT_MODEL_CHECKPOINT_PATH))
    parser.add_argument("--sae-root", default=os.environ.get(ENV_SAE_ROOT, DEFAULT_SAE_ROOT))
    parser.add_argument("--device", default=os.environ.get(ENV_DEVICE, "cuda"))
    parser.add_argument("--max-gpu-jobs", type=int, default=int(os.environ.get(ENV_MAX_GPU_JOBS, "1")))
    parser.add_argument("--log-path", default=os.environ.get(ENV_LOG_PATH))
    parser.add_argument("--log-level", default=os.environ.get(ENV_LOG_LEVEL, "INFO"))
    parser.add_argument(
        "--timeout-worker-healthcheck",
        type=int,
        default=int(os.environ.get(ENV_WORKER_HEALTHCHECK_TIMEOUT, "60")),
        help="Seconds uvicorn waits for a worker healthcheck response before killing it.",
    )
    parser.add_argument(
        "--sae-layer-path",
        action="append",
        default=None,
        help="Override a default SAE path. Format: LAYER=/path/to/sae_dir. May be repeated.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    os.environ[ENV_HOST] = str(args.host)
    os.environ[ENV_PORT] = str(args.port)
    os.environ[ENV_MODEL_PATH] = str(args.model_checkpoint_path)
    os.environ[ENV_SAE_ROOT] = str(args.sae_root)
    os.environ[ENV_DEVICE] = str(args.device)
    os.environ[ENV_MAX_GPU_JOBS] = str(max(1, int(args.max_gpu_jobs)))
    os.environ[ENV_LOG_LEVEL] = str(args.log_level)
    os.environ[ENV_WORKER_HEALTHCHECK_TIMEOUT] = str(int(args.timeout_worker_healthcheck))
    if args.log_path:
        os.environ[ENV_LOG_PATH] = str(args.log_path)
    os.environ[ENV_SAE_PATHS] = json.dumps(_parse_sae_layer_paths(args.sae_layer_path, args.sae_root))

    uvicorn.run(
        "inference_server:app",
        host=str(args.host),
        port=int(args.port),
        workers=int(args.workers),
        timeout_worker_healthcheck=int(args.timeout_worker_healthcheck),
    )


if __name__ == "__main__":
    main()
