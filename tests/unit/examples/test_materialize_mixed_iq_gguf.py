# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for the mixed-IQ GGUF validation bridge."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "megatron_bridge"
    / "materialize_mixed_iq_gguf.py"
)
_SPEC = importlib.util.spec_from_file_location("materialize_mixed_iq_gguf", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
bridge = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bridge
_SPEC.loader.exec_module(bridge)


def test_packed_rows_preserves_bytes_and_recovers_logical_shape():
    tensor = torch.arange(2 * 3 * 74, dtype=torch.int64).remainder(256).to(torch.uint8)
    tensor = tensor.reshape(2, 3, 74)

    raw, logical_shape = bridge.packed_rows(tensor)

    assert raw.shape == (2, 3 * 74)
    assert raw.tobytes() == tensor.numpy().tobytes()
    assert logical_shape == [2, 3 * 256]


@pytest.mark.parametrize(
    "tensor",
    [
        torch.zeros((2, 3, 74), dtype=torch.int8),
        torch.zeros((2, 3, 73), dtype=torch.uint8),
        torch.zeros((2, 74), dtype=torch.uint8),
    ],
)
def test_packed_rows_rejects_noncanonical_payloads(tensor):
    with pytest.raises(ValueError, match="IQ2_XS payload"):
        bridge.packed_rows(tensor)


def test_iq2_layer_matching_normalizes_model_prefix(tmp_path):
    (tmp_path / "hf_quant_config.json").write_text(
        """{
          "quantization": {
            "quantized_layers": {
              "model.layers.1.mixer.experts": {"quant_algo": "IQ2_XS"},
              "model.layers.2.mixer.in_proj": {"quant_algo": "NVFP4"}
            }
          }
        }"""
    )

    layers = bridge._load_iq2_layers(tmp_path)

    assert layers == {"backbone.layers.1.mixer.experts"}
    assert bridge._matches_iq2_layer("backbone.layers.1.mixer.experts.7.up_proj.weight", layers)
    assert not bridge._matches_iq2_layer("backbone.layers.2.mixer.in_proj.weight", layers)


def test_materialize_declares_model_arch_for_pinned_converter(monkeypatch, tmp_path):
    registered = {}

    class FakeModelBase:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            if "model_arch" not in cls.__dict__:
                raise TypeError(f"Missing property 'model_arch' for {cls.__name__!r}")

        @classmethod
        def register(cls, architecture):
            def decorator(model_cls):
                registered[architecture] = model_cls
                return model_cls

            return decorator

    class FakeNemotronHModel(FakeModelBase):
        model_arch = "nemotron_h"

    conversion = ModuleType("conversion")
    conversion.__path__ = []
    base = ModuleType("conversion.base")
    base.LazyTorchTensor = object
    base.ModelBase = FakeModelBase
    nemotron = ModuleType("conversion.nemotron")
    nemotron.NemotronHModel = FakeNemotronHModel

    monkeypatch.setitem(sys.modules, "gguf", ModuleType("gguf"))
    monkeypatch.setitem(sys.modules, "conversion", conversion)
    monkeypatch.setitem(sys.modules, "conversion.base", base)
    monkeypatch.setitem(sys.modules, "conversion.nemotron", nemotron)
    monkeypatch.setattr(bridge.runpy, "run_path", lambda *args, **kwargs: None)
    monkeypatch.setattr(bridge, "_git_commit", lambda source: "pinned-commit")

    expected, commit = bridge.materialize(tmp_path, tmp_path, tmp_path / "model.gguf")

    assert expected == {}
    assert commit == "pinned-commit"
    assert registered["NemotronHForCausalLM"].model_arch == "nemotron_h"
