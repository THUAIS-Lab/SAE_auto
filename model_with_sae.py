from __future__ import annotations

import os
import re
import json
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

from sae_lens import SAE as SAELensSAE  # type: ignore
from sae_lens import HookedSAETransformer  # type: ignore

try:
    from safetensors.torch import load_file as load_safetensors_file
except Exception:
    load_safetensors_file = None

PSEUDO_ACTIVATION = False


def load_model(model_name: str, device: str, use_hooked_transformer: bool):
    try:
        is_cuda = str(device).startswith("cuda")
        if use_hooked_transformer:
            if HookedSAETransformer is not None:
                print(f"Loading HookedSAETransformer for model: {model_name}")

                model = HookedSAETransformer.from_pretrained_no_processing(
                    model_name=model_name,
                    device=str(device),
                    dtype=torch.bfloat16 if is_cuda else torch.float32,
                ).to(device).eval()

                return model

            raise RuntimeError("Neither SAELens nor TransformerLens is available")

        print(f"Loading AutoModelForCausalLM for model: {model_name}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if is_cuda else torch.float32,
            low_cpu_mem_usage=True,
        )

        model = model.to(device).eval()
        return model

    except Exception as e:
        print(f"Warning: Could not load model {model_name}: {e}")
        return None


def load_tokenizer(model_name: str):
    """
    Loads the tokenizer for the language model.
    """
    import os
    token = os.environ.get("HF_TOKEN")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)#, token=token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        print(f"Warning: Could not load tokenizer for {model_name}: {e}")
        return None


def _finalize_local_sae_dict(sae_data: Dict[str, Any], *, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if "W_enc" in sae_data and "encoder.weight" not in sae_data:
        sae_data["encoder.weight"] = sae_data["W_enc"].transpose(0, 1).contiguous()
    if "W_dec" in sae_data and "decoder.weight" not in sae_data:
        sae_data["decoder.weight"] = sae_data["W_dec"]
    if "b_enc" in sae_data and "encoder.bias" not in sae_data:
        sae_data["encoder.bias"] = sae_data["b_enc"]
    if "b_dec" in sae_data and "decoder.bias" not in sae_data:
        sae_data["decoder.bias"] = sae_data["b_dec"]

    if "encoder.weight" in sae_data and "W_enc" not in sae_data:
        sae_data["W_enc"] = sae_data["encoder.weight"]
    if "decoder.weight" in sae_data and "W_dec" not in sae_data:
        sae_data["W_dec"] = sae_data["decoder.weight"]
    if "encoder.bias" in sae_data and "b_enc" not in sae_data:
        sae_data["b_enc"] = sae_data["encoder.bias"]
    if "decoder.bias" in sae_data and "b_dec" not in sae_data:
        sae_data["b_dec"] = sae_data["decoder.bias"]

    if config is not None:
        sae_data["__config__"] = config
    return sae_data


def _load_local_safetensors_sae_dir(sae_dir: str, device: str) -> Dict[str, Any]:
    if load_safetensors_file is None:
        raise RuntimeError(
            "safetensors is required to load this SAE directory. "
            "Please install it with: pip install safetensors"
        )

    config: Dict[str, Any] = {}
    config_path = os.path.join(sae_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    index_path = os.path.join(sae_dir, "model.safetensors.index.json")
    single_path = os.path.join(sae_dir, "model.safetensors")

    sae_data: Dict[str, Any] = {}
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        shard_names = sorted(set(str(v) for v in weight_map.values()))
        for shard_name in shard_names:
            shard_path = os.path.join(sae_dir, shard_name)
            shard_tensors = load_safetensors_file(shard_path, device=str(device))
            for key, tensor in shard_tensors.items():
                sae_data[key] = tensor.to(device=device, dtype=torch.float32)
        return _finalize_local_sae_dict(sae_data, config=config)

    if os.path.exists(single_path):
        shard_tensors = load_safetensors_file(single_path, device=str(device))
        for key, tensor in shard_tensors.items():
            sae_data[key] = tensor.to(device=device, dtype=torch.float32)
        return _finalize_local_sae_dict(sae_data, config=config)

    raise FileNotFoundError(
        f"No params.npz, model.safetensors, or model.safetensors.index.json found under {sae_dir}"
    )


def load_sae(sae_path: str, device: str) -> Dict[str, Any]:
    try:
        # Support SAELens URI scheme: "sae-lens://release=...;sae_id=..."
        if isinstance(sae_path, str) and sae_path.startswith("sae-lens://"):
            spec = sae_path[len("sae-lens://"):]
            parts = [p.strip() for p in spec.split(";") if p.strip()]
            kv: Dict[str, str] = {}
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    kv[k.strip()] = v.strip()
            release = kv.get("release") or kv.get("repo") or kv.get("model")
            sae_id = kv.get("sae_id") or kv.get("path")
            if not release or not sae_id:
                print("Warning: Invalid sae-lens URI. Expected keys: release and sae_id")
                return {}
            print(f"Loading SAE from {release}/{sae_id} to device {device}")
            loaded = SAELensSAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device=str(device),
            )  # type: ignore
            sae_obj = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
            print(f"SAE loaded on {device}")
            return {"__sae_lens_obj__": sae_obj, "__source__": "sae-lens", "release": release, "sae_id": sae_id}

        if os.path.isdir(sae_path):
            npz_candidate = os.path.join(sae_path, "params.npz")
            if os.path.exists(npz_candidate):
                sae_path = npz_candidate
            else:
                return _load_local_safetensors_sae_dir(sae_path, device)

        if os.path.exists(sae_path):
            if sae_path.endswith(".npz"):
                npz_data = np.load(sae_path)
                sae_data: Dict[str, Any] = {}
                for key in npz_data.files:
                    arr = npz_data[key]
                    sae_data[key] = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
                return _finalize_local_sae_dict(sae_data)

            if sae_path.endswith(".safetensors"):
                if load_safetensors_file is None:
                    raise RuntimeError(
                        "safetensors is required to load this SAE file. "
                        "Please install it with: pip install safetensors"
                    )
                sae_data = {
                    key: value.to(device=device, dtype=torch.float32)
                    for key, value in load_safetensors_file(sae_path, device=str(device)).items()
                }
                return _finalize_local_sae_dict(sae_data)

            sae_data = torch.load(sae_path, map_location=device)
            if isinstance(sae_data, dict):
                return _finalize_local_sae_dict(sae_data)
            return sae_data
        print(f"Warning: SAE file not found at {sae_path}")
        return {}
    except Exception as e:
        print(f"Warning: Could not load SAE from {sae_path}: {e}")
        return {}


def infer_sae_layer_from_path(sae_path: str) -> Optional[int]:
    if not isinstance(sae_path, str) or not sae_path:
        return None

    candidates: List[str] = [sae_path]
    if sae_path.startswith("sae-lens://"):
        spec = sae_path[len("sae-lens://"):]
        parts = [p.strip() for p in spec.split(";") if p.strip()]
        kv: Dict[str, str] = {}
        for part in parts:
            if "=" in part:
                k, v = part.split("=", 1)
                kv[k.strip()] = v.strip()
        sae_id = kv.get("sae_id") or kv.get("path")
        if sae_id:
            candidates.insert(0, sae_id)

    for text in candidates:
        match = re.search(r"layer[_-]?(\d+)", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


class _BaseBackend:
    def __init__(self, owner: "ModelWithSAEModule"):
        self.owner = owner

    def get_layer_activations(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def run_logits(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def run_logits_with_intervention(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        residual_intervention: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_layer_activations_and_logits(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        raise NotImplementedError


class _SaeLensBackend(_BaseBackend):
    def _require_hook_name(self) -> str:
        hook_name = self.owner.hook_name
        if not isinstance(hook_name, str) or not hook_name:
            raise RuntimeError("hook_name not available for hooked-model operations.")
        return hook_name

    def get_layer_activations(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        hook_name = self._require_hook_name()
        _, cache = self.owner.model.run_with_cache(
            input_ids,
            names_filter=[hook_name],
            attention_mask=attention_mask,
        )
        return cache[hook_name]

    def get_layer_activations_and_logits(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        hook_name = self._require_hook_name()
        try:
            logits, cache = self.owner.model.run_with_cache(
                input_ids,
                names_filter=[hook_name],
                attention_mask=attention_mask,
                return_type="logits",
            )
        except TypeError:
            logits, cache = self.owner.model.run_with_cache(
                input_ids,
                names_filter=[hook_name],
                attention_mask=attention_mask,
            )
        return cache[hook_name], logits

    def run_logits(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Keep semantics aligned with HF backend: this should be clean model logits
        # without automatically inserting SAE transforms.
        return self.owner.model(
            input_ids,
            return_type="logits",
            attention_mask=attention_mask,
        )

    def run_logits_with_intervention(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        residual_intervention: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        hook_name = self._require_hook_name()

        def _hook_fn(clean_act, hook):
            _ = hook
            return residual_intervention(clean_act)

        return self.owner.model.run_with_hooks(
            input_ids,
            return_type="logits",
            fwd_hooks=[(hook_name, _hook_fn)],
            attention_mask=attention_mask,
        )


class _HFBackend(_BaseBackend):
    def get_layer_activations(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        outputs = self.owner.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        assert isinstance(hidden_states, (list, tuple))

        layer_idx = self.owner._resolve_hidden_state_index(len(hidden_states))
        if layer_idx is None:
            return None
        return hidden_states[layer_idx]

    def get_layer_activations_and_logits(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        outputs = self.owner.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        assert isinstance(hidden_states, (list, tuple))

        layer_idx = self.owner._resolve_hidden_state_index(len(hidden_states))
        layer_activations = hidden_states[layer_idx] if layer_idx is not None else None
        if not hasattr(outputs, "logits") or outputs.logits is None:
            raise RuntimeError("Model output does not contain logits.")
        return layer_activations, outputs.logits

    def run_logits(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.owner.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        if not hasattr(outputs, "logits") or outputs.logits is None:
            raise RuntimeError("Model output does not contain logits.")
        return outputs.logits

    def run_logits_with_intervention(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        residual_intervention: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        target_module = self.owner._resolve_local_intervention_module()

        def _local_hook(module, hook_inputs, hook_output):  # noqa: ARG001
            _ = hook_inputs
            hidden = hook_output[0] if isinstance(hook_output, tuple) else hook_output
            steered_hidden = residual_intervention(hidden)
            if isinstance(hook_output, tuple):
                return (steered_hidden, *hook_output[1:])
            return steered_hidden

        hook_handle = target_module.register_forward_hook(_local_hook)
        try:
            return self.run_logits(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            hook_handle.remove()


class ModelWithSAEModule:

    def __init__(
        self,
        llm_name: str,
        sae_path: str,
        sae_layer: Optional[int] = None,
        feature_index: int = -1,
        device: str = "cpu",
        context_size: int = 128,
        debug: bool = False,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        sae: Optional[Dict[str, Any]] = None,
    ):
        inferred_layer = infer_sae_layer_from_path(sae_path)
        if sae_layer is None:
            sae_layer = inferred_layer
        elif inferred_layer is not None and int(sae_layer) != inferred_layer:
            print(
                f"Warning: explicit sae_layer={sae_layer} differs from sae_path inferred layer={inferred_layer}."
            )

        self.layer = int(sae_layer) if sae_layer is not None else None
        self.debug = debug  # Store debug flag
        self.device = device
        
        self.feature_index = feature_index
            
        self.model_name = llm_name
        self.sae_path = sae_path

        # Determine if we're using SAELens (which requires HookedTransformer)
        self.use_hooked_transformer = sae_path.startswith("sae-lens://") if isinstance(sae_path, str) else False

        self.sae = sae if sae is not None else load_sae(sae_path, self.device)  # Load SAE first to check compatibility
        if not self.sae:
            raise RuntimeError(
                f"Failed to load SAE from path: {sae_path}. "
                "Activation tracing/intervention requires a valid SAE."
            )
        self.model = model if model is not None else load_model(llm_name, self.device, self.use_hooked_transformer)
        if self.model is None and self.use_hooked_transformer:
            print(
                "Warning: failed to load HookedSAETransformer model; "
                "falling back to AutoModelForCausalLM with local forward-hook intervention."
            )
            self.use_hooked_transformer = False
            self.model = load_model(llm_name, self.device, use_hooked_transformer=False)
        self.tokenizer = tokenizer if tokenizer is not None else load_tokenizer(llm_name)
        if self.model is None:
            raise RuntimeError(
                f"Failed to load model: {llm_name}. "
                "Check model path/name and environment dependencies."
            )
        if self.use_hooked_transformer and "__sae_lens_obj__" not in self.sae:
            print(
                "Warning: SAE-Lens object missing while hooked mode was requested; "
                "falling back to local forward-hook intervention."
            )
            self.use_hooked_transformer = False
            # Important: when disabling hooked mode, ensure model API also switches
            # from TransformerLens-style forward(...) to HF-style forward(input_ids=...).
            if isinstance(self.model, HookedTransformer):
                self.model = load_model(llm_name, self.device, use_hooked_transformer=False)
                if self.model is None:
                    raise RuntimeError(
                        "Failed to reload HuggingFace model after disabling hooked mode. "
                        "Please verify llm_name and environment dependencies."
                    )
        if self.layer is None and not self.use_hooked_transformer:
            raise ValueError(
                "sae_layer is required for non-hooked models when it cannot be inferred from sae_path."
            )
        
        self.context_size = context_size
        
        self.hook_name = None
        if isinstance(self.sae, dict) and "__sae_lens_obj__" in self.sae:
            sae_obj = self.sae["__sae_lens_obj__"]
            cfg = getattr(sae_obj, "cfg", None)
            
            # 1. 尝试直接从 cfg 获取 (支持较新版本的 sae-lens)
            if hasattr(cfg, "hook_name") and cfg.hook_name:
                self.hook_name = cfg.hook_name
            else:
                # 2. 回退到从 metadata 获取 (兼容老版本)
                metadata = getattr(cfg, "metadata", None)
                if isinstance(metadata, dict):
                    self.hook_name = metadata.get("hook_name")
                    
            # 3. 如果还是没拿到，根据层数手动推断 (TransformerLens 默认格式)
            if not self.hook_name and self.layer is not None:
                self.hook_name = f"blocks.{self.layer}.hook_resid_post"

        self._backend: _BaseBackend = self._build_backend()

    def _build_backend(self) -> _BaseBackend:
        if self.use_hooked_transformer:
            return _SaeLensBackend(self)
        return _HFBackend(self)

    def _resolve_hidden_state_index(self, num_hidden_states: int) -> Optional[int]:
        # HF hidden_states[0] is embeddings; block layer k corresponds to index k+1.
        layer_idx = int(self.layer) + 1 if self.layer is not None else 1
        if layer_idx < 0 or layer_idx >= num_hidden_states:
            return None
        return layer_idx

    def _new_trace_template(self) -> Dict[str, Any]:
        return {
            "tokens": [],
            "token_ids": [],
            "per_token_activation": [],
            "summary_activation": 0.0,
            "summary_activation_mean": 0.0,
            "summary_activation_sum": 0.0,
            "max_token_index": 0,
            "layer_index": self.layer,
            "shapes": {},
            "raw_stats": {},
        }

    def _build_pseudo_trace(self, text: str) -> Dict[str, Any]:
        trace = self._new_trace_template()
        ids = [ord(c) % 256 for c in text]
        fallback_activations = [0.5] * len(ids)
        summary_max = max(fallback_activations) if fallback_activations else 0.0
        summary_mean = (
            sum(fallback_activations) / len(fallback_activations) if fallback_activations else 0.0
        )
        summary_sum = sum(fallback_activations)
        trace.update(
            {
                "tokens": list(text),
                "token_ids": ids,
                "per_token_activation": fallback_activations,
                "summary_activation": summary_max,
                "summary_activation_mean": summary_mean,
                "summary_activation_sum": summary_sum,
                "max_token_index": 0,
            }
        )
        return trace

    def _build_trace_from_feature_row(
        self,
        *,
        input_id_row: torch.Tensor,
        feature_row: torch.Tensor,
        attention_mask_row: Optional[torch.Tensor] = None,
        selected_position: Optional[int] = None,
        selected_base_value: Optional[float] = None,
        selected_target_value: Optional[float] = None,
    ) -> Dict[str, Any]:
        trace = self._new_trace_template()

        if attention_mask_row is None:
            valid_positions = torch.arange(
                int(input_id_row.shape[0]),
                device=input_id_row.device,
                dtype=torch.long,
            )
        else:
            mask = attention_mask_row.to(device=input_id_row.device, dtype=torch.bool)
            if mask.shape != input_id_row.shape:
                raise ValueError(
                    f"attention_mask row shape {tuple(mask.shape)} must match input_id row shape {tuple(input_id_row.shape)}."
                )
            valid_positions = mask.nonzero(as_tuple=False).flatten()
        valid_length = int(valid_positions.numel())

        input_ids_cpu = input_id_row.index_select(0, valid_positions).detach().cpu().tolist()
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids_cpu)
            except Exception:
                tokens = [str(x) for x in input_ids_cpu]
        else:
            tokens = [str(x) for x in input_ids_cpu]

        trace["tokens"] = tokens
        trace["token_ids"] = input_ids_cpu

        if valid_length <= 0:
            trace["intervention_token_index"] = 0
            trace["intervention_token"] = ""
            trace["intervention_base_activation"] = 0.0
            trace["intervention_target_activation"] = 0.0
            return trace

        valid_feature_row = feature_row.index_select(0, valid_positions)
        per_token_list = [float(x) for x in valid_feature_row.detach().cpu().tolist()]
        summary_activation = float(valid_feature_row.max().item())
        summary_activation_mean = float(valid_feature_row.mean().item())
        summary_activation_sum = float(valid_feature_row.sum().item())
        max_token_index = int(valid_feature_row.argmax().item())
        per_token_act_no_bos = valid_feature_row[1:] if valid_feature_row.shape[0] > 1 else valid_feature_row
        summary_activation_no_bos = float(per_token_act_no_bos.max().item())

        selected_padded_position: Optional[int] = None
        if selected_position is None:
            selected_compact_position = max_token_index
        else:
            selected_padded_position = int(max(0, min(int(selected_position), int(input_id_row.shape[0]) - 1)))
            matches = (valid_positions == selected_padded_position).nonzero(as_tuple=False).flatten()
            selected_compact_position = int(matches[0].item()) if matches.numel() > 0 else max_token_index
        selected_compact_position = int(max(0, min(int(selected_compact_position), valid_length - 1)))
        selected_token = tokens[selected_compact_position] if selected_compact_position < len(tokens) else ""
        if selected_base_value is None:
            selected_base_value = float(valid_feature_row[selected_compact_position].item())
        if selected_target_value is None:
            selected_target_value = float(selected_base_value)

        trace["per_token_activation"] = per_token_list
        trace["summary_activation"] = float(round(summary_activation, 4))
        trace["summary_activation_no_bos"] = float(round(summary_activation_no_bos, 4))
        trace["summary_activation_mean"] = float(round(summary_activation_mean, 4))
        trace["summary_activation_sum"] = float(round(summary_activation_sum, 4))
        trace["max_token_index"] = max_token_index
        trace["intervention_token_index"] = selected_compact_position
        if selected_padded_position is not None:
            trace["intervention_padded_token_index"] = selected_padded_position
        trace["intervention_token"] = str(selected_token)
        trace["intervention_base_activation"] = float(round(float(selected_base_value), 4))
        trace["intervention_target_activation"] = float(round(float(selected_target_value), 4))

        mean_val = sum(per_token_list) / len(per_token_list)
        variance = sum((x - mean_val) ** 2 for x in per_token_list) / len(per_token_list)
        std_val = variance ** 0.5
        trace["raw_stats"] = {
            "min": float(min(per_token_list)),
            "max": float(max(per_token_list)),
            "mean": float(mean_val),
            "sum": float(sum(per_token_list)),
            "std": float(std_val),
            "count": len(per_token_list),
        }
        return trace

    @torch.no_grad()
    def analyze_feature_batch_from_tensors(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        feature_index: Optional[int] = None,
        max_activation_scale: float = 2.0,
    ) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if not self.sae:
            raise RuntimeError("No SAE loaded. Activation trace requires a valid SAE.")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be shape [batch, seq].")

        feature_idx = self.feature_index if feature_index is None else int(feature_index)
        input_ids = input_ids.to(self.device)
        attention_mask = self._coerce_attention_mask(input_ids, attention_mask)

        layer_activations, clean_logits = self._backend.get_layer_activations_and_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if layer_activations is None:
            raise RuntimeError("Failed to capture layer activations for batch analysis.")

        sae_features = self._encode_with_sae(layer_activations)
        if sae_features is None or sae_features.ndim != 3:
            raise RuntimeError("SAE encode returned invalid feature tensor; expected shape [batch, seq, d_sae].")
        if feature_idx < 0 or feature_idx >= sae_features.shape[-1]:
            raise ValueError(
                f"Feature index {feature_idx} out of range for SAE feature dim {sae_features.shape[-1]}."
            )

        feature_values = sae_features[..., feature_idx]
        target_info = self._resolve_max_activation_token_targets(
            feature_values,
            attention_mask=attention_mask,
            scale=max_activation_scale,
        )

        traces: List[Dict[str, Any]] = []
        for batch_index in range(int(input_ids.shape[0])):
            trace = self._build_trace_from_feature_row(
                input_id_row=input_ids[batch_index],
                feature_row=feature_values[batch_index],
                attention_mask_row=attention_mask[batch_index],
                selected_position=int(target_info["positions"][batch_index].item()),
                selected_base_value=float(target_info["base_values"][batch_index].item()),
                selected_target_value=float(target_info["target_values"][batch_index].item()),
            )
            trace["shapes"] = {
                "layer_activations": list(layer_activations.shape),
                "sae_features": list(sae_features.shape),
            }
            traces.append(trace)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "clean_logits": clean_logits,
            "layer_activations": layer_activations,
            "sae_features": sae_features,
            "feature_values": feature_values,
            "selected_positions": target_info["positions"],
            "selected_base_values": target_info["base_values"],
            "selected_target_values": target_info["target_values"],
            "traces": traces,
        }

    def _compute_activation_trace_from_tensors(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        batch_analysis = self.analyze_feature_batch_from_tensors(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        traces = batch_analysis.get("traces", [])
        if not traces:
            return self._new_trace_template()
        return traces[0]

    def get_activation_trace(self, text: str) -> Dict[str, Any]:
        if PSEUDO_ACTIVATION or self.model is None or self.tokenizer is None:
            print("Warning: PyTorch, model, or tokenizer not available.")
            return self._build_pseudo_trace(text)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
        )
        input_ids: torch.Tensor = inputs["input_ids"]  # type: ignore
        attention_mask = inputs.get("attention_mask")
        return self._compute_activation_trace_from_tensors(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def get_activation_trace_from_tensors(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        return self._compute_activation_trace_from_tensors(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    @torch.no_grad()
    def get_max_activation_intervention_details(
        self,
        input_ids: torch.Tensor,
        feature_index: int,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        scale: float = 2.0,
    ) -> Dict[str, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be shape [batch, seq].")

        input_ids = input_ids.to(self.device)
        attention_mask = self._coerce_attention_mask(input_ids, attention_mask)
        layer_activations = self._backend.get_layer_activations(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        feature_values = self._encode_with_sae(layer_activations)[..., feature_index]
        return self._resolve_max_activation_token_targets(
            feature_values,
            attention_mask=attention_mask,
            scale=scale,
        )

    def _get_transformer_blocks(self):
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        for path in (
            "model.layers",
            "model.model.layers",
            "transformer.h",
            "gpt_neox.layers",
        ):
            current = self.model
            ok = True
            for part in path.split("."):
                if not hasattr(current, part):
                    ok = False
                    break
                current = getattr(current, part)
            if ok and isinstance(current, (torch.nn.ModuleList, list, tuple)):
                return current
        raise RuntimeError("Could not locate transformer blocks for forward hooking.")

    def _resolve_local_intervention_module(self):
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if self.layer is None:
            raise RuntimeError("layer must be set for local intervention.")
        if self.layer < 0:
            raise RuntimeError(
                f"Requested SAE layer {self.layer}. Layer must be >= 0 and refer to transformer block index."
            )

        blocks = self._get_transformer_blocks()
        block_index = int(self.layer)
        if block_index >= len(blocks):
            raise RuntimeError(
                f"Requested SAE layer {self.layer}, but model has {len(blocks)} blocks."
            )
        return blocks[block_index]

    def _coerce_attention_mask(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if attention_mask is not None:
            return attention_mask.to(device=input_ids.device, dtype=torch.long)
        if self.tokenizer is None:
            return torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            return torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
        return (input_ids != pad_id).to(dtype=torch.long, device=input_ids.device)

    def _encode_with_sae(self, residual: torch.Tensor, *, return_pre: bool = False) -> torch.Tensor:
        if not self.sae:
            raise RuntimeError("No SAE loaded.")

        if "__sae_lens_obj__" in self.sae:
            if return_pre:
                raise RuntimeError("return_pre=True is not supported for SAE-Lens object encode.")
            sae_obj = self.sae["__sae_lens_obj__"]
            return sae_obj.encode(residual)

        cfg = self.sae.get("__config__", {}) if isinstance(self.sae, dict) else {}

        w_enc = self.sae.get("W_enc")
        if w_enc is None:
            w_enc = self.sae.get("encoder.weight")
        if w_enc is None:
            raise RuntimeError("Local SAE is missing encoder weights.")

        b_dec = self.sae.get("b_dec")
        if b_dec is None:
            b_dec = self.sae.get("decoder.bias")
        b_enc = self.sae.get("b_enc")
        if b_enc is None:
            b_enc = self.sae.get("encoder.bias")
        threshold = self.sae.get("threshold")

        residual_dtype = residual.dtype
        hidden = residual
        if bool(cfg.get("input_normalize", False)):
            eps = float(cfg.get("input_normalize_eps", 1e-5))
            hidden_mean = hidden.mean(dim=-1, keepdim=True)
            hidden_std = hidden.std(dim=-1, keepdim=True, unbiased=False)
            hidden = (hidden - hidden_mean) / (hidden_std + eps)

        w_enc = w_enc.to(device=residual.device, dtype=residual_dtype)
        activation_name = " ".join(
            str(cfg.get(key, ""))
            for key in ("activation", "activation_fn", "architecture")
        ).lower()
        is_jumprelu = threshold is not None and (
            "jump" in activation_name or not activation_name.strip()
        )

        apply_b_dec_default = not is_jumprelu
        apply_b_dec = bool(cfg.get("apply_b_dec_to_input", apply_b_dec_default))
        if b_dec is not None and apply_b_dec:
            hidden = hidden - b_dec.to(device=residual.device, dtype=residual_dtype)

        if w_enc.ndim != 2:
            raise RuntimeError("Encoder weight must be 2D.")
        if w_enc.shape[0] == residual.shape[-1]:
            pre = torch.matmul(hidden, w_enc)
        elif w_enc.shape[1] == residual.shape[-1]:
            pre = torch.matmul(hidden, w_enc.transpose(0, 1))
        else:
            raise RuntimeError(
                f"Incompatible encoder shape {tuple(w_enc.shape)} for residual dim {residual.shape[-1]}."
            )

        if b_enc is not None:
            pre = pre + b_enc.to(device=pre.device, dtype=pre.dtype)

        if is_jumprelu:
            if return_pre:
                return pre
            threshold_t = threshold.to(device=pre.device, dtype=pre.dtype)
            return F.relu(pre) * (pre > threshold_t).to(dtype=pre.dtype)

        if threshold is not None:
            pre = pre - threshold.to(device=pre.device, dtype=pre.dtype)
        if return_pre:
            return pre

        act = F.relu(pre)
        activation_name = str(cfg.get("activation", "")).lower()
        if activation_name == "topk":
            k = int(cfg.get("k", 0) or 0)
            if 0 < k < act.shape[-1]:
                topk_values, topk_indices = torch.topk(act, k=k, dim=-1)
                sparse_act = torch.zeros_like(act)
                sparse_act.scatter_(-1, topk_indices, topk_values)
                act = sparse_act

        return act

    def _encode_with_sae_legacy(self, residual: torch.Tensor) -> torch.Tensor:
        if not self.sae:
            raise RuntimeError("No SAE loaded.")

        if "__sae_lens_obj__" in self.sae:
            sae_obj = self.sae["__sae_lens_obj__"]
            return sae_obj.encode(residual)

        cfg = self.sae.get("__config__", {}) if isinstance(self.sae, dict) else {}

        w_enc = self.sae.get("W_enc")
        if w_enc is None:
            w_enc = self.sae.get("encoder.weight")
        if w_enc is None:
            raise RuntimeError("Local SAE is missing encoder weights.")

        b_dec = self.sae.get("b_dec")
        if b_dec is None:
            b_dec = self.sae.get("decoder.bias")
        b_enc = self.sae.get("b_enc")
        if b_enc is None:
            b_enc = self.sae.get("encoder.bias")
        threshold = self.sae.get("threshold")

        residual_dtype = residual.dtype
        hidden = residual
        if bool(cfg.get("input_normalize", False)):
            eps = float(cfg.get("input_normalize_eps", 1e-5))
            hidden_mean = hidden.mean(dim=-1, keepdim=True)
            hidden_std = hidden.std(dim=-1, keepdim=True, unbiased=False)
            hidden = (hidden - hidden_mean) / (hidden_std + eps)

        w_enc = w_enc.to(device=residual.device, dtype=residual_dtype)
        centered = hidden
        if b_dec is not None:
            centered = centered - b_dec.to(device=residual.device, dtype=residual_dtype)

        if w_enc.ndim != 2:
            raise RuntimeError("Encoder weight must be 2D.")
        if w_enc.shape[0] == residual.shape[-1]:
            pre = torch.matmul(centered, w_enc)
        elif w_enc.shape[1] == residual.shape[-1]:
            pre = torch.matmul(centered, w_enc.transpose(0, 1))
        else:
            raise RuntimeError(
                f"Incompatible encoder shape {tuple(w_enc.shape)} for residual dim {residual.shape[-1]}."
            )

        if b_enc is not None:
            pre = pre + b_enc.to(device=pre.device, dtype=pre.dtype)
        if threshold is not None:
            pre = pre - threshold.to(device=pre.device, dtype=pre.dtype)

        act = F.relu(pre)
        activation_name = str(cfg.get("activation", "")).lower()
        if activation_name == "topk":
            k = int(cfg.get("k", 0) or 0)
            if 0 < k < act.shape[-1]:
                topk_values, topk_indices = torch.topk(act, k=k, dim=-1)
                sparse_act = torch.zeros_like(act)
                sparse_act.scatter_(-1, topk_indices, topk_values)
                act = sparse_act

        return act

    def _decode_with_sae(self, features: torch.Tensor) -> torch.Tensor:
        if not self.sae:
            raise RuntimeError("No SAE loaded.")

        if "__sae_lens_obj__" in self.sae:
            # print('decoding with sae-lens')
            sae_obj = self.sae["__sae_lens_obj__"]
            return sae_obj.decode(features)

        # print('decoding with local sae')
        w_dec = self.sae.get("W_dec")
        if w_dec is None:
            w_dec = self.sae.get("decoder.weight")
        if w_dec is None:
            raise RuntimeError("Local SAE is missing decoder weights.")

        b_dec = self.sae.get("b_dec")
        if b_dec is None:
            b_dec = self.sae.get("decoder.bias")

        feature_dtype = features.dtype
        w_dec = w_dec.to(device=features.device, dtype=feature_dtype)
        if w_dec.ndim != 2:
            raise RuntimeError("Decoder weight must be 2D.")
        if w_dec.shape[0] == features.shape[-1]:
            recon = torch.matmul(features, w_dec)
        elif w_dec.shape[1] == features.shape[-1]:
            recon = torch.matmul(features, w_dec.transpose(0, 1))
        else:
            raise RuntimeError(
                f"Incom {features.shape[-1]}."
            )

        if b_dec is not None:
            recon = recon + b_dec.to(device=recon.device, dtype=recon.dtype)
        return recon

    def _resolve_max_activation_token_targets(
        self,
        feature_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        scale: float = 2.0,
    ) -> Dict[str, torch.Tensor]:
        if feature_values.ndim != 2:
            raise ValueError("feature_values must be shape [batch, seq].")

        if attention_mask is None:
            valid_mask = torch.ones_like(feature_values, dtype=torch.bool, device=feature_values.device)
        else:
            if attention_mask.shape != feature_values.shape:
                raise ValueError(
                    f"attention_mask shape {tuple(attention_mask.shape)} must match feature_values shape {tuple(feature_values.shape)}."
                )
            valid_mask = attention_mask.to(device=feature_values.device, dtype=torch.bool)

        # Prefer content tokens over BOS when the sequence has non-BOS content.
        # The BOS token is the first valid token, which is not necessarily column 0
        # under left padding.
        if feature_values.shape[1] > 1:
            non_bos_mask = valid_mask.clone()
            valid_counts = valid_mask.to(dtype=torch.long).sum(dim=1)
            has_non_bos = valid_counts > 1
            first_valid_positions = valid_mask.to(dtype=torch.long).argmax(dim=1)
            row_indices = torch.arange(feature_values.shape[0], device=feature_values.device)
            non_bos_mask[row_indices[has_non_bos], first_valid_positions[has_non_bos]] = False
            valid_mask = torch.where(has_non_bos[:, None], non_bos_mask, valid_mask)

        masked_values = feature_values.masked_fill(~valid_mask, float("-inf"))
        has_valid = valid_mask.any(dim=1)
        max_positions = masked_values.argmax(dim=1)
        fallback_positions = valid_mask.to(dtype=torch.long).sum(dim=1).sub(1).clamp(min=0)
        max_positions = torch.where(has_valid, max_positions, fallback_positions)

        base_values = feature_values.gather(1, max_positions[:, None]).squeeze(1)
        target_values = base_values * float(scale)
        return {
            "positions": max_positions,
            "base_values": base_values,
            "target_values": target_values,
        }

    def _apply_feature_intervention(
        self,
        features: torch.Tensor,
        feature_index: int,
        value: Optional[Union[float, torch.Tensor]],
        mode: str,
        intervention_scope: str = "all_tokens",
        attention_mask: Optional[torch.Tensor] = None,
        max_activation_scale: float = 2.0,
    ) -> torch.Tensor:
        if feature_index < 0 or feature_index >= features.shape[-1]:
            raise ValueError(f"Feature {feature_index} out of range for {features.shape[-1]} features.")
        if mode not in {"clamp", "add"}:
            raise ValueError("mode must be one of: clamp, add")
        if intervention_scope not in {"all_tokens", "last_token_only", "max_activation_token"}:
            raise ValueError("intervention_scope must be one of: all_tokens, last_token_only, max_activation_token")

        steered = features.clone()
        clean_target = features[..., feature_index]
        if intervention_scope == "max_activation_token":
            target_info = self._resolve_max_activation_token_targets(
                clean_target,
                attention_mask=attention_mask,
                scale=max_activation_scale,
            )
            batch_index = torch.arange(steered.shape[0], device=steered.device)
            token_index = target_info["positions"]
            target = steered[..., feature_index]

            if value is None:
                base_values = target_info["base_values"].to(device=target.device, dtype=target.dtype)
                target_values = target_info["target_values"].to(device=target.device, dtype=target.dtype)
                if mode == "clamp":
                    target[batch_index, token_index] = target_values
                else:
                    target[batch_index, token_index] += target_values - base_values
                return steered

            if torch.is_tensor(value):
                v = value.to(device=target.device, dtype=target.dtype)
                if v.ndim == 0:
                    scalar = float(v.item())
                    if mode == "clamp":
                        target[batch_index, token_index] = scalar
                    else:
                        target[batch_index, token_index] += scalar
                else:
                    if v.ndim == 2 and v.shape[1] == 1:
                        v = v[:, 0]
                    if v.ndim != 1 or v.shape[0] != batch_index.shape[0]:
                        raise ValueError(
                            f"Intervention tensor shape {tuple(v.shape)} does not match target shape {(batch_index.shape[0],)}."
                        )
                    if mode == "clamp":
                        target[batch_index, token_index] = v
                    else:
                        target[batch_index, token_index] += v
            else:
                scalar = float(value)
                if mode == "clamp":
                    target[batch_index, token_index] = scalar
                else:
                    target[batch_index, token_index] += scalar
            return steered

        if value is None:
            raise ValueError("value is required unless intervention_scope is max_activation_token.")
        target = steered[..., feature_index]
        if intervention_scope == "last_token_only":
            target = target[:, -2:-1]
        if torch.is_tensor(value):
            v = value.to(device=target.device, dtype=target.dtype)
            if v.ndim == 0:
                if mode == "clamp":
                    target.fill_(float(v.item()))
                else:
                    target.add_(float(v.item()))
            else:
                if v.ndim == 1 and v.shape[0] == target.shape[0]:
                    v = v[:, None]
                if v.shape != target.shape:
                    raise ValueError(
                        f"Intervention tensor shape {tuple(v.shape)} does not match target shape {tuple(target.shape)}."
                    )
                if mode == "clamp":
                    target.copy_(v)
                else:
                    target.add_(v)
        else:
            scalar = float(value)
            if mode == "clamp":
                target.fill_(scalar)
            else:
                target.add_(scalar)
        return steered

    @torch.no_grad()
    def run_logits(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be shape [batch, seq].")

        input_ids = input_ids.to(self.device)
        attention_mask = self._coerce_attention_mask(input_ids, attention_mask)

        return self._backend.run_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    @torch.no_grad()
    def run_logits_with_feature_intervention(
        self,
        input_ids: torch.Tensor,
        feature_index: int,
        value: Optional[Union[float, torch.Tensor]] = None,
        mode: str = "add",
        attention_mask: Optional[torch.Tensor] = None,
        intervention_scope: str = "all_tokens",
        max_activation_scale: float = 2.0,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be shape [batch, seq].")
        if mode not in {"clamp", "add"}:
            raise ValueError("mode must be one of: clamp, add")
        if intervention_scope not in {"all_tokens", "last_token_only", "max_activation_token"}:
            raise ValueError("intervention_scope must be one of: all_tokens, last_token_only, max_activation_token")
        if value is None and intervention_scope != "max_activation_token":
            raise ValueError("value is required unless intervention_scope is max_activation_token.")

        input_ids = input_ids.to(self.device)
        attention_mask = self._coerce_attention_mask(input_ids, attention_mask)

        def _residual_intervention(hidden: torch.Tensor) -> torch.Tensor:
            clean_features = self._encode_with_sae(hidden)
            clean_recon = self._decode_with_sae(clean_features)
            steered_features = self._apply_feature_intervention(
                clean_features,
                feature_index=feature_index,
                value=value,
                mode=mode,
                intervention_scope=intervention_scope,
                attention_mask=attention_mask,
                max_activation_scale=max_activation_scale,
            )
            steered_recon = self._decode_with_sae(steered_features)
            return steered_recon + (hidden - clean_recon)

        return self._backend.run_logits_with_intervention(
            input_ids=input_ids,
            attention_mask=attention_mask,
            residual_intervention=_residual_intervention,
        )

    def _gen_hook(
        self,
        resid: torch.Tensor,
        hook: Any,
        *,
        feature: int,
        value: Union[float, torch.Tensor],
        sae: Any,
    ) -> torch.Tensor:
        _ = hook
        clean_features = sae.encode(resid)
        clean_recon = sae.decode(clean_features)

        steered_features = clean_features.clone()
        target = steered_features[..., int(feature)]
        if target.ndim == 2:
            target = target[:, -1:]

        if torch.is_tensor(value):
            v = value.to(device=target.device, dtype=target.dtype)
            if v.ndim == 0:
                target.fill_(float(v.item()))
            else:
                if v.ndim == 1 and v.shape[0] == target.shape[0]:
                    v = v[:, None]
                if v.shape != target.shape:
                    raise ValueError(
                        f"Intervention tensor shape {tuple(v.shape)} does not match target shape {tuple(target.shape)}."
                    )
                target.copy_(v)
        else:
            target.fill_(float(value))

        steered_recon = sae.decode(steered_features)
        return steered_recon + (resid - clean_recon)

    @torch.no_grad()
    def _find_clamp_values_for_kl(
        self,
        prompts_tokens: torch.Tensor,
        feature_index: int,
        sae_obj: Any,
        *,
        target_kl: float,
        tolerance: float = 0.1,
        max_steps: int = 12,
        intervention_scope: str = "last_token_only",
        attention_mask: Optional[torch.Tensor] = None,
        clean_logits: Optional[torch.Tensor] = None,
    ) -> tuple[list[float], list[float]]:
        if prompts_tokens.ndim != 2:
            raise ValueError("prompts_tokens must be shape [batch, seq].")
        if intervention_scope == "max_activation_token":
            raise ValueError("intervention_scope=max_activation_token does not support KL-based clamp search.")

        if target_kl == 0:
            return [0.0], [0.0]

        sign = 1.0 if target_kl >= 0 else -1.0
        target = abs(float(target_kl))

        prompts_tokens = prompts_tokens.to(self.device)
        attention_mask = self._coerce_attention_mask(prompts_tokens, attention_mask)
        if clean_logits is None:
            clean_logits = self.run_logits(input_ids=prompts_tokens, attention_mask=attention_mask)
        else:
            clean_logits = clean_logits.to(self.device)

        def _mean_kl(steered_logits: torch.Tensor) -> float:
            clean_logprob = F.log_softmax(clean_logits, dim=-1)
            steered_logprob = F.log_softmax(steered_logits, dim=-1)
            clean_prob = clean_logprob.exp()
            kl = (clean_prob * (clean_logprob - steered_logprob)).sum(dim=-1)
            return float(kl.mean().item())

        def _eval_clamp(val: float) -> float:
            steered_logits = self.run_logits_with_feature_intervention(
                input_ids=prompts_tokens,
                feature_index=int(feature_index),
                value=float(val),
                mode="clamp",
                attention_mask=attention_mask,
                intervention_scope=intervention_scope,
            )
            return _mean_kl(steered_logits)

        low = 0.0
        high = 1.0
        kl_high = _eval_clamp(sign * high)

        expand_steps = 0
        while kl_high < target and expand_steps < 10:
            high *= 2.0
            kl_high = _eval_clamp(sign * high)
            expand_steps += 1

        best_val = sign * high
        best_kl = kl_high

        for _ in range(max(1, int(max_steps))):
            mid = (low + high) / 2.0
            kl_mid = _eval_clamp(sign * mid)
            best_val = sign * mid
            best_kl = kl_mid

            if abs(kl_mid - target) <= float(tolerance):
                break
            if kl_mid < target:
                low = mid
            else:
                high = mid

        return [float(best_val)], [float(best_kl)]


__all__ = [
    "ModelWithSAEModule",
    "load_model",
    "load_tokenizer",
    "load_sae",
]


if __name__ == "__main__":
    llm_name = "google/gemma-2-2b"
    
    # 确保 sae_path 中的 layer_6 和下方的 sae_layer=6 保持一致
    sae_path = "sae-lens://release=gemma-scope-2b-pt-res;sae_id=layer_6/width_16k/average_l0_70"
    test_feature_index = 0 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"⏳ 正在初始化 ModelWithSAEModule (设备: {device})...")
    module = ModelWithSAEModule(
        llm_name=llm_name,
        sae_path=sae_path,
        sae_layer=6,
        feature_index=test_feature_index,
        device=device,
        debug=True
    )

    prompt = "Hello, world! This is a simple prompt to test SAE forward pass."
    print(f"\n🚀 开始执行前向传播并提取特征激活...\nPrompt: '{prompt}'")

    # 【修改点2】去掉了 try...except 捕获，让系统原生的报错 Traceback 直接暴露出来
    trace_result = module.get_activation_trace(prompt)
    
    print("\n" + "="*40)
    print("🎯 激活追踪结果 (Activation Trace)")
    print("="*40)
    
    tokens = trace_result.get("tokens", [])
    activations = trace_result.get("per_token_activation", [])
    
    print(f"监测特征 ID: {test_feature_index}")
    print(f"最大激活值 (Max): {trace_result.get('summary_activation')}")
    print(f"平均激活值 (Mean): {trace_result.get('summary_activation_mean')}")
    print(f"最大激活对应的 Token 索引: {trace_result.get('max_token_index')}")
    print("\n📊 逐 Token 激活详情:")
    
    if tokens and activations and len(tokens) == len(activations):
        for i, (tok, act) in enumerate(zip(tokens, activations)):
            marker = " <--- MAX" if i == trace_result.get("max_token_index") else ""
            print(f"  [{i:02d}] {tok:>15} : {act:.4f}{marker}")
    else:
        print("⚠️ 未能正确获取 tokens 或激活值列表。")
