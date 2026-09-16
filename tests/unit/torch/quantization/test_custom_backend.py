# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for registering and using a custom quantization backend."""

import torch

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import (
    TensorQuantizer,
    TensorQuantizerCache,
    quant_backend_caches_reconstruction,
    register_quant_backend,
    unregister_quant_backend,
)


def test_backend_cache_capability_freezes_after_calibration():
    backend_name = "reconstruction_cache_backend"
    register_quant_backend(
        backend_name, lambda inputs, _quantizer: inputs, caches_reconstruction=True
    )
    model = torch.nn.Linear(16, 16, bias=False)
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": 8, "axis": None, "backend": backend_name},
                "enable": True,
            },
        ],
        "algorithm": "max",
    }

    try:
        assert quant_backend_caches_reconstruction(backend_name)
        mtq.quantize(model, config)
        assert model.weight_quantizer._quantizer_cache == {"_modelopt_cache_frozen": True}
    finally:
        unregister_quant_backend(backend_name)

    assert not quant_backend_caches_reconstruction(backend_name)


def test_backend_cache_capability_freezes_after_all_restore_hooks(monkeypatch):
    backend_name = "restore_reconstruction_cache_backend"
    register_quant_backend(
        backend_name, lambda inputs, _quantizer: inputs, caches_reconstruction=True
    )
    model = torch.nn.Linear(16, 16, bias=False)
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": 8, "axis": None, "backend": backend_name},
                "enable": True,
            },
        ],
        "algorithm": "max",
    }

    try:
        mtq.quantize(model, config)
        modelopt_state = mto.modelopt_state(model)

        # Emulate a weight-bearing plugin hook that resets state and does not call the base hook.
        quant_module_type = type(model)

        def reset_cache_in_restore_hook(self, _prefix=""):
            self.weight_quantizer.unfreeze_quantizer_cache()

        monkeypatch.setattr(quant_module_type, "modelopt_post_restore", reset_cache_in_restore_hook)

        restored = torch.nn.Linear(16, 16, bias=False)
        mto.restore_from_modelopt_state(restored, modelopt_state)
        assert restored.weight_quantizer._quantizer_cache == {"_modelopt_cache_frozen": True}
    finally:
        unregister_quant_backend(backend_name)


def test_custom_backend_via_quantize():
    # Define and register a simple dummy backend that adds a constant to inputs
    def dummy_backend(inputs: torch.Tensor, tq) -> torch.Tensor:
        extra = getattr(tq, "backend_extra_args", None) or {}
        offset = extra.get("offset", 1.0)
        return inputs + offset

    register_quant_backend("dummy_backend", dummy_backend)

    model = torch.nn.Linear(16, 16, bias=False)

    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {
                    "num_bits": 8,
                    "axis": None,
                    "backend": "dummy_backend",
                    "backend_extra_args": {"offset": 2.5},
                },
                "enable": True,
            },
        ],
        "algorithm": "max",
    }

    inputs = torch.randn(1, 16)

    def forward_loop(m):
        m(inputs)

    mtq.quantize(model, cfg, forward_loop=forward_loop)
    output_test = model(inputs)

    assert torch.allclose(output_test, inputs @ (model.weight.T + 2.5))

    # Unregister the backend to avoid impacting other tests
    unregister_quant_backend("dummy_backend")


def test_custom_backend_with_quantizer_cache():
    """Test that TensorQuantizerCache is initialized on first forward and not saved/restored."""

    # Simple cache class
    class DummyCache(TensorQuantizerCache):
        def __init__(self):
            super().__init__()
            self.value = 3.0

    # Backend that creates cache on first call
    def cached_backend(inputs: torch.Tensor, tq: TensorQuantizer) -> torch.Tensor:
        if tq._quantizer_cache is None:
            tq._quantizer_cache = DummyCache()
        return inputs + tq._quantizer_cache.value

    register_quant_backend("cached_backend", cached_backend)

    model = torch.nn.Linear(16, 16, bias=False)
    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"backend": "cached_backend"},
                "enable": True,
            },
        ],
        "algorithm": "max",
    }
    inputs = torch.randn(1, 16)

    mtq.quantize(model, cfg, forward_loop=lambda m: m(inputs))

    # Cache is None before first forward
    assert model.weight_quantizer._quantizer_cache is None

    # Cache gets initialized on forward
    model(inputs)
    assert isinstance(model.weight_quantizer._quantizer_cache, DummyCache)

    # Save modelopt state
    modelopt_state = mto.modelopt_state(model)

    # Cache is not saved in modelopt_state
    assert not any("_quantizer_cache" in key for key in modelopt_state)

    # Restore to a new model
    model_restored = torch.nn.Linear(16, 16, bias=False)
    mto.restore_from_modelopt_state(model_restored, modelopt_state)
    model_restored.load_state_dict(model.state_dict())

    # Cache is None after restore
    assert model_restored.weight_quantizer._quantizer_cache is None

    # Cache gets re-initialized on forward
    model_restored(inputs)
    assert isinstance(model_restored.weight_quantizer._quantizer_cache, DummyCache)

    unregister_quant_backend("cached_backend")
