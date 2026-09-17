"""Windows-only compatibility for importing the Linux runtime modules in tests."""

from __future__ import annotations

import importlib.util
import os
import sys
import types


if os.name == "nt" and importlib.util.find_spec("fcntl") is None:
    fcntl = types.ModuleType("fcntl")
    fcntl.LOCK_SH = 1
    fcntl.LOCK_EX = 2
    fcntl.LOCK_NB = 4
    fcntl.LOCK_UN = 8
    fcntl.flock = lambda *_args, **_kwargs: None
    sys.modules["fcntl"] = fcntl


# The local Windows test environment has an intentionally mismatched
# transformers/huggingface-hub pair. Tests in this directory exercise scan
# orchestration with fake modules, so keep the production import isolated.
if os.name == "nt" and "model_with_sae" not in sys.modules:
    model_with_sae = types.ModuleType("model_with_sae")

    class ModelWithSAEModule:  # pragma: no cover - construction is not used locally
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("real model loading is unavailable in Windows unit tests")

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError("real model loading is unavailable in Windows unit tests")

    model_with_sae.ModelWithSAEModule = ModelWithSAEModule
    model_with_sae.load_model = _unavailable
    model_with_sae.load_sae = _unavailable
    model_with_sae.load_tokenizer = _unavailable
    sys.modules["model_with_sae"] = model_with_sae
