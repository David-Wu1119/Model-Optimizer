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


from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.export.utils import ToyModel, partial_fp8_config, partial_w4a8_config

import modelopt.torch.export.quant_utils as quant_utils
import modelopt.torch.quantization as mtq
from modelopt.torch.export.quant_utils import (
    _pack_iq_weight,
    _validate_iq_export_weight_shapes,
    postprocess_state_dict,
)
from modelopt.torch.export.unified_export_hf import (
    _export_quantized_weight,
    _export_transformers_checkpoint,
    _process_quantized_modules,
    export_hf_checkpoint,
)
from modelopt.torch.export.unified_export_hf_streaming import (
    _export_transformers_checkpoint_streaming,
)
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import GroupedQuantizer, TensorQuantizer
from modelopt.torch.quantization.utils import quantizer_attr_names


@pytest.mark.parametrize(
    "weight_name",
    ["weight", "weight_2", "some_other_w"],
)
def test_quantizer_attr_names(weight_name):
    quantizer_attrs = quantizer_attr_names(weight_name)
    if weight_name == "weight":
        assert quantizer_attrs.weight_scale == "weight_scale"
        assert quantizer_attrs.input_scale == "input_scale"
        assert quantizer_attrs.weight_scale_2 == "weight_scale_2"
        assert quantizer_attrs.weight_quantizer == "weight_quantizer"
        assert quantizer_attrs.input_quantizer == "input_quantizer"
        assert quantizer_attrs.output_quantizer == "output_quantizer"
        assert quantizer_attrs.output_scale == "output_scale"
    else:
        assert quantizer_attrs.weight_scale == f"{weight_name}_weight_scale"
        assert quantizer_attrs.input_scale == f"{weight_name}_input_scale"
        assert quantizer_attrs.weight_scale_2 == f"{weight_name}_weight_scale_2"
        assert quantizer_attrs.weight_quantizer == f"{weight_name}_weight_quantizer"
        assert quantizer_attrs.input_quantizer == f"{weight_name}_input_quantizer"
        assert quantizer_attrs.output_quantizer == f"{weight_name}_output_quantizer"
        assert quantizer_attrs.output_scale == f"{weight_name}_output_scale"


def test_export_per_tensor_quantized_weight():
    model = ToyModel(dims=[32, 256, 32, 128])

    mtq.quantize(model, partial_fp8_config, lambda x: x(torch.randn(1, 4, 32)))

    orig_dtype = model.linears[0].weight.dtype
    quantizer_attrs = quantizer_attr_names("weight")
    _export_quantized_weight(model.linears[0], torch.float32, "weight")
    assert model.linears[0].weight.dtype == orig_dtype
    assert hasattr(model.linears[0], quantizer_attrs.weight_quantizer)
    assert not getattr(model.linears[0], quantizer_attrs.weight_quantizer).is_enabled
    assert not hasattr(model.linears[0], quantizer_attrs.weight_scale)
    assert not hasattr(model.linears[0], quantizer_attrs.weight_scale_2)
    assert not hasattr(model.linears[0], quantizer_attrs.input_scale)
    assert hasattr(model.linears[0], quantizer_attrs.input_quantizer)
    assert not getattr(model.linears[0], quantizer_attrs.input_quantizer).is_enabled
    assert hasattr(model.linears[0], quantizer_attrs.output_quantizer)
    assert not getattr(model.linears[0], quantizer_attrs.output_quantizer).is_enabled
    assert not hasattr(model.linears[0], quantizer_attrs.output_scale)

    _export_quantized_weight(model.linears[1], torch.float32, "weight")
    assert model.linears[1].weight.dtype == torch.float8_e4m3fn
    assert hasattr(model.linears[1], quantizer_attrs.weight_quantizer)
    assert hasattr(model.linears[1], quantizer_attrs.weight_scale)
    assert not hasattr(model.linears[1], quantizer_attrs.weight_scale_2)
    assert hasattr(model.linears[1], quantizer_attrs.input_quantizer)
    assert hasattr(model.linears[1], quantizer_attrs.input_scale)
    assert hasattr(model.linears[1], quantizer_attrs.output_quantizer)
    assert not getattr(model.linears[1], quantizer_attrs.output_quantizer).is_enabled
    assert not hasattr(model.linears[1], quantizer_attrs.output_scale)


def test_export_per_block_quantized_weight():
    model = ToyModel(dims=[32, 256, 256, 32])

    mtq.quantize(model, partial_w4a8_config, lambda x: x(torch.randn(1, 4, 32)))

    quantizer_attrs = quantizer_attr_names("weight")
    _export_quantized_weight(model.linears[2], torch.float32, "weight")
    assert model.linears[2].weight.dtype == torch.uint8
    assert hasattr(model.linears[2], quantizer_attrs.weight_quantizer)
    assert hasattr(model.linears[2], quantizer_attrs.weight_scale)
    assert hasattr(model.linears[2], quantizer_attrs.weight_scale_2)
    assert hasattr(model.linears[2], quantizer_attrs.input_scale)
    assert hasattr(model.linears[2], quantizer_attrs.input_quantizer)

    assert hasattr(model.linears[2], quantizer_attrs.output_quantizer)
    assert not getattr(model.linears[2], quantizer_attrs.output_quantizer).is_enabled
    assert not hasattr(model.linears[2], quantizer_attrs.output_scale)


@pytest.mark.parametrize(("num_bits", "payload_bytes"), [("iq1_s", 50), ("iq2_xs", 74)])
def test_export_iq_payload_as_weight(num_bits, payload_bytes):
    linear = nn.Linear(256, 4, bias=False, dtype=torch.bfloat16)
    linear.weight_quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=num_bits,
            block_sizes={-1: 256},
            backend="ggml",
            backend_extra_args={"search_impl": "auto"},
        )
    )

    _export_quantized_weight(linear, torch.bfloat16)
    state_dict = postprocess_state_dict(linear.state_dict(), maxbound=448, quantization=None)

    assert state_dict["weight"].shape == (4, 1, payload_bytes)
    assert state_dict["weight"].dtype == torch.uint8
    assert isinstance(linear.weight, nn.Parameter)
    assert "packed_weights" not in state_dict
    assert "weight_shape" not in state_dict


@pytest.mark.parametrize(("num_bits", "payload_bytes"), [("iq1_s", 50), ("iq2_xs", 74)])
def test_iq_packer_preserves_leading_dimensions(num_bits, payload_bytes):
    weight = torch.randn(2, 3, 512, dtype=torch.bfloat16)

    packed = _pack_iq_weight(weight, num_bits)

    assert packed.shape == (2, 3, 2, payload_bytes)


def test_postprocess_state_dict_drops_legacy_weight_shape():
    state_dict = {
        "layer.weight": torch.ones(2, 2),
        "layer.weight_shape": torch.tensor([2, 2]),
    }

    processed = postprocess_state_dict(state_dict, maxbound=448, quantization=None)

    assert set(processed) == {"layer.weight"}


@pytest.mark.parametrize("num_bits", ["iq1_s", "iq2_xs"])
def test_export_iq_divisibility_error_identifies_weight(num_bits):
    linear = nn.Linear(192, 4, bias=False, dtype=torch.bfloat16)
    linear.weight_quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=num_bits,
            block_sizes={-1: 256},
            backend="ggml",
            backend_extra_args={"search_impl": "auto"},
        )
    )

    with pytest.raises(
        ValueError,
        match=rf"Failed to pack {num_bits.upper()} weight 'Linear.weight' with shape \(4, 192\)",
    ):
        _export_quantized_weight(linear, torch.bfloat16)


@pytest.mark.parametrize("num_bits", ["iq1_s", "iq2_xs"])
def test_export_iq_dtype_error_identifies_weight(num_bits):
    weight = torch.ones((4, 256), dtype=torch.uint8)

    with pytest.raises(
        TypeError,
        match=rf"Failed to pack {num_bits.upper()} weight 'model.layers.0.weight'",
    ):
        _pack_iq_weight(weight, num_bits, describe_as="model.layers.0.weight")


def test_export_iq_payload_shape_error_identifies_weight(monkeypatch):
    weight = torch.ones((4, 256), dtype=torch.bfloat16)

    def pack_with_wrong_shape(weight):
        return torch.zeros((4, 2, 74), dtype=torch.uint8), None

    monkeypatch.setattr(quant_utils, "quantize_iq2_xs", pack_with_wrong_shape)

    with pytest.raises(
        RuntimeError,
        match=r"packing for weight 'model\.layers\.0\.weight'.*expected \(4, 1, 74\)",
    ):
        _pack_iq_weight(weight, "iq2_xs", describe_as="model.layers.0.weight")


def test_iq_packer_rejects_non_iq_format_before_inspecting_module():
    linear = nn.Linear(256, 4, bias=False, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="Unsupported IQ quantization format: fp8"):
        _pack_iq_weight(linear.weight, "fp8", module=linear)


def test_export_iq_shape_preflight_reports_every_incompatible_weight():
    model = nn.Sequential(
        nn.Linear(192, 4, bias=False, dtype=torch.bfloat16),
        nn.Linear(256, 4, bias=False, dtype=torch.bfloat16),
        nn.Linear(320, 4, bias=False, dtype=torch.bfloat16),
    )
    for linear in model:
        linear.weight_quantizer = TensorQuantizer(
            QuantizerAttributeConfig(
                num_bits="iq2_xs",
                block_sizes={-1: 256},
                backend="ggml",
            )
        )

    with pytest.raises(ValueError) as exc_info:
        _validate_iq_export_weight_shapes(model)

    message = str(exc_info.value)
    assert "0.weight: (4, 192)" in message
    assert "2.weight: (4, 320)" in message
    assert "1.weight" not in message


def test_export_iq_shape_preflight_skips_detailed_scan_without_iq(monkeypatch):
    model = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
    monkeypatch.setattr(
        quant_utils,
        "_iq_export_weight_shape_errors",
        lambda _model: pytest.fail("non-IQ export must skip the detailed weight scan"),
    )

    _validate_iq_export_weight_shapes(model)


def test_export_iq_shape_preflight_reports_grouped_weights():
    class GroupedLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_gemms = 2
            self.weight0 = nn.Parameter(torch.ones(4, 256, dtype=torch.bfloat16))
            self.weight1 = nn.Parameter(torch.ones(4, 192, dtype=torch.bfloat16))
            quantizer_config = QuantizerAttributeConfig(
                num_bits="iq2_xs",
                block_sizes={-1: 256},
                backend="ggml",
            )
            self.weight_quantizer = GroupedQuantizer(
                TensorQuantizer(quantizer_config),
                TensorQuantizer(quantizer_config),
            )

    model = nn.Module()
    model.experts = GroupedLinear()

    with pytest.raises(ValueError) as exc_info:
        _validate_iq_export_weight_shapes(model)

    message = str(exc_info.value)
    assert "experts.weight1: (4, 192)" in message
    assert "experts.weight0" not in message


def test_export_iq_shape_preflight_reports_grouped_quantizers_beyond_num_gemms():
    class GroupedLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_gemms = 1
            self.weight0 = nn.Parameter(torch.ones(4, 256, dtype=torch.bfloat16))
            self.weight1 = nn.Parameter(torch.ones(4, 192, dtype=torch.bfloat16))
            quantizer_config = QuantizerAttributeConfig(
                num_bits="iq2_xs",
                block_sizes={-1: 256},
                backend="ggml",
            )
            self.weight_quantizer = GroupedQuantizer(
                TensorQuantizer(quantizer_config),
                TensorQuantizer(quantizer_config),
            )

    model = nn.Module()
    model.experts = GroupedLinear()

    with pytest.raises(ValueError, match=r"experts\.weight1: \(4, 192\)"):
        _validate_iq_export_weight_shapes(model)


def test_export_iq_preflight_rejects_nonstandard_weight_before_mutation(tmp_path):
    class CustomWeightModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Parameter(torch.ones(4, 256, dtype=torch.bfloat16))
            self.proj_weight_quantizer = TensorQuantizer(
                QuantizerAttributeConfig(
                    num_bits="iq2_xs",
                    block_sizes={-1: 256},
                    backend="ggml",
                )
            )

    model = nn.Module()
    model.block = CustomWeightModule()
    model.config = SimpleNamespace(torch_dtype=torch.bfloat16)
    original_weight = model.block.proj.detach().clone()

    with pytest.raises(ValueError, match=r"block\.proj: nonstandard weight"):
        export_hf_checkpoint(model, export_dir=tmp_path)

    assert model.block.proj.dtype == torch.bfloat16
    assert torch.equal(model.block.proj, original_weight)

    with pytest.raises(NotImplementedError, match=r"got 'model\.block\.proj'"):
        _export_quantized_weight(
            model.block,
            torch.bfloat16,
            weight_name="proj",
            describe_as="model.block.proj",
        )


def test_export_hf_checkpoint_runs_iq_shape_preflight_before_mutation(tmp_path):
    linear = nn.Linear(192, 4, bias=False, dtype=torch.bfloat16)
    linear.weight_quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits="iq2_xs",
            block_sizes={-1: 256},
            backend="ggml",
        )
    )
    linear.config = SimpleNamespace(torch_dtype=torch.bfloat16)
    original_weight = linear.weight.detach().clone()

    with pytest.raises(ValueError, match=r"weight: \(4, 192\)"):
        export_hf_checkpoint(linear, export_dir=tmp_path)

    assert linear.weight.dtype == torch.bfloat16
    assert torch.equal(linear.weight, original_weight)


def test_export_transformers_checkpoint_runs_iq_shape_preflight_before_mutation():
    model = nn.Sequential(
        nn.Linear(256, 4, bias=False, dtype=torch.bfloat16),
        nn.Linear(192, 4, bias=False, dtype=torch.bfloat16),
    )
    model.config = SimpleNamespace(torch_dtype=torch.bfloat16)
    for linear in model:
        linear.weight_quantizer = TensorQuantizer(
            QuantizerAttributeConfig(
                num_bits="iq2_xs",
                block_sizes={-1: 256},
                backend="ggml",
            )
        )
    original_weights = [linear.weight.detach().clone() for linear in model]

    with pytest.raises(ValueError, match=r"1\.weight: \(4, 192\)"):
        _export_transformers_checkpoint(model)

    for linear, original_weight in zip(model, original_weights):
        assert linear.weight.dtype == torch.bfloat16
        assert torch.equal(linear.weight, original_weight)


def test_export_transformers_checkpoint_streaming_runs_iq_shape_preflight_before_mutation(
    tmp_path,
):
    linear = nn.Linear(192, 4, bias=False, dtype=torch.bfloat16)
    linear.config = SimpleNamespace(torch_dtype=torch.bfloat16)
    linear.weight_quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits="iq2_xs",
            block_sizes={-1: 256},
            backend="ggml",
        )
    )
    original_weight = linear.weight.detach().clone()

    with pytest.raises(ValueError, match=r"weight: \(4, 192\)"):
        _export_transformers_checkpoint_streaming(linear, export_dir=tmp_path)

    assert linear.weight.dtype == torch.bfloat16
    assert torch.equal(linear.weight, original_weight)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("num_bits", ["iq1_s", "iq2_xs"])
def test_export_iq_rejects_an_unhandled_search_impl(num_bits):
    linear = nn.Linear(256, 4, bias=False, dtype=torch.bfloat16)
    linear.weight_quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=num_bits,
            block_sizes={-1: 256},
            backend="ggml",
            backend_extra_args={"search_impl": "reference"},
        )
    )

    with pytest.raises(NotImplementedError, match="only supports search_impl='auto'"):
        _export_quantized_weight(linear, torch.bfloat16)


@pytest.mark.parametrize(
    ("quantizer_name", "mode"),
    [
        ("input_quantizer", "enabled"),
        ("input_quantizer", "pre_quant_scale"),
        ("output_quantizer", "enabled"),
    ],
)
@pytest.mark.parametrize("num_bits", ["iq1_s", "iq2_xs"])
def test_export_iq_rejects_non_weight_only_config(num_bits, quantizer_name, mode):
    linear = nn.Linear(256, 4, bias=False, dtype=torch.bfloat16)
    linear.weight_quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=num_bits,
            block_sizes={-1: 256},
            backend="ggml",
            backend_extra_args={"search_impl": "auto"},
        )
    )
    activation_quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=8, axis=None, enable=mode == "enabled")
    )
    setattr(linear, quantizer_name, activation_quantizer)
    if mode == "pre_quant_scale":
        activation_quantizer.pre_quant_scale = torch.ones(256)

    with pytest.raises(NotImplementedError, match="export is weight-only"):
        _export_quantized_weight(linear, torch.bfloat16)


def test_iq_config_error_identifies_weight():
    linear = nn.Linear(256, 4, bias=False, dtype=torch.bfloat16)
    linear.weight_quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits="iq2_xs",
            block_sizes={-1: 256},
            backend="ggml",
        )
    )
    linear.input_quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, axis=None))

    with pytest.raises(NotImplementedError, match=r"model\.layers\.3\.mlp\.down_proj\.weight"):
        _pack_iq_weight(
            linear.weight,
            "iq2_xs",
            module=linear,
            describe_as="model.layers.3.mlp.down_proj.weight",
        )


class QuantMoELinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(8, 8, bias=False) for _ in range(2)])

    def forward(self, x):
        return self.experts[0](x)


class _SingleRoutedExpertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = QuantMoELinear()

    def forward(self, x):
        return self.moe(x)


def test_process_quantized_modules_fills_step3p5_moe_input_scale_for_unrouted_experts():
    model = _SingleRoutedExpertModel()
    quant_cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": None}},
            {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
        ],
        "algorithm": "max",
    }

    mtq.quantize(model, quant_cfg, lambda m: m(torch.randn(2, 4, 8)))

    assert model.moe.experts[0].input_quantizer.amax is not None
    assert model.moe.experts[1].input_quantizer.amax is None

    _process_quantized_modules(model, torch.float32)

    assert hasattr(model.moe.experts[0], "input_scale")
    assert hasattr(model.moe.experts[1], "input_scale")
