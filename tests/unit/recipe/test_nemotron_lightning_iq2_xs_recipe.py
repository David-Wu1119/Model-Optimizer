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

"""Quantizer-placement tests for the Nemotron 3.5 Lightning mixed IQ2_XS recipe."""

import json

import pytest
import torch
import transformers
from _test_utils.torch.transformers_models import get_tiny_nemotron_h
from safetensors.torch import load_file

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
from modelopt.torch.export.quant_utils import get_quant_config
from modelopt.torch.quantization.config import need_calibration
from modelopt.torch.quantization.conversion import set_quantizer_by_cfg
from modelopt.torch.quantization.nn import TensorQuantizer

_RECIPE = "models/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/ptq/iq2_xs_experts-nvfp4_mamba"


def _linear_with_weight_quantizer(*, per_expert: bool = False):
    linear = torch.nn.Module()
    quantizer = TensorQuantizer()
    linear.weight_quantizer = torch.nn.ModuleList([quantizer]) if per_expert else quantizer
    return linear


def test_nemotron_lightning_iq2_xs_recipe_matches_megatron_core_names():
    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([torch.nn.Module()])
    layer = model.decoder.layers[0]
    layer.mlp = torch.nn.Module()
    layer.mlp.experts = torch.nn.Module()
    layer.mlp.experts.linear_fc1 = _linear_with_weight_quantizer(per_expert=True)
    layer.mlp.shared_experts = torch.nn.Module()
    layer.mlp.shared_experts.linear_fc1 = _linear_with_weight_quantizer()
    model.mtp = torch.nn.Module()
    model.mtp.mlp = torch.nn.Module()
    model.mtp.mlp.experts = torch.nn.Module()
    model.mtp.mlp.experts.linear_fc1 = _linear_with_weight_quantizer(per_expert=True)

    config = load_recipe(_RECIPE).quantize.model_dump(exclude_unset=True)
    set_quantizer_by_cfg(model, config["quant_cfg"])

    routed = layer.mlp.experts.linear_fc1.weight_quantizer[0]
    shared = layer.mlp.shared_experts.linear_fc1.weight_quantizer
    mtp = model.mtp.mlp.experts.linear_fc1.weight_quantizer[0]
    assert routed.is_enabled and routed.num_bits == "iq2_xs"
    assert shared.is_enabled and shared.num_bits == "iq2_xs"
    assert not mtp.is_enabled


@pytest.mark.skipif(
    not hasattr(transformers, "NemotronHConfig"),
    reason="NemotronH is not supported by this Transformers version",
)
def test_nemotron_lightning_iq2_xs_recipe_quantizer_placement():
    model = get_tiny_nemotron_h()
    model.mtp = torch.nn.Module()
    model.mtp.mixer = torch.nn.Module()
    model.mtp.mixer.experts = torch.nn.Linear(256, 256, bias=False)
    config = load_recipe(_RECIPE).quantize.model_dump(exclude_unset=True)

    assert not need_calibration(config)
    mtq.quantize(model, config)

    quantizers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, TensorQuantizer)
    }
    iq2_xs = {
        name: module
        for name, module in quantizers.items()
        if module.is_enabled and module.num_bits == "iq2_xs"
    }
    nvfp4 = {
        name: module
        for name, module in quantizers.items()
        if module.is_enabled and module.num_bits == (2, 1)
    }

    assert iq2_xs
    assert all(".mixer.experts." in name or ".mixer.shared_experts." in name for name in iq2_xs)
    assert any(".mixer.experts." in name for name in iq2_xs)
    assert any(".mixer.shared_experts." in name for name in iq2_xs)
    assert all(module.block_sizes == {-1: 256} for module in iq2_xs.values())

    assert set(nvfp4) == {
        "model.layers.0.mixer.in_proj.weight_quantizer",
        "model.layers.0.mixer.out_proj.weight_quantizer",
    }
    assert all(module.block_sizes[-1] == 16 for module in nvfp4.values())
    assert all(module.block_sizes["type"] == "dynamic" for module in nvfp4.values())

    enabled = {name for name, module in quantizers.items() if module.is_enabled}
    assert enabled == set(iq2_xs) | set(nvfp4)
    assert not quantizers["mtp.mixer.experts.weight_quantizer"].is_enabled

    hf_quant_config = get_quant_config(model)
    quantization = hf_quant_config["quantization"]
    assert quantization["quant_algo"] == "MIXED_PRECISION"
    groups = convert_hf_quant_config_format(hf_quant_config)["config_groups"]
    weights = [group["weights"] for group in groups.values()]
    assert any(weight.get("packing") == "ggml" and weight["num_bits"] == 2 for weight in weights)
    assert any(weight["num_bits"] == 4 and weight["group_size"] == 16 for weight in weights)


@pytest.mark.skipif(
    not hasattr(transformers, "NemotronHConfig"),
    reason="NemotronH is not supported by this Transformers version",
)
def test_nemotron_lightning_iq2_xs_recipe_direct_export(tmp_path):
    model = get_tiny_nemotron_h(
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=2,
        hybrid_override_pattern="ME",
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        mamba_num_heads=8,
        mamba_head_dim=32,
        n_routed_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=256,
        n_shared_experts=1,
        moe_shared_expert_intermediate_size=256,
    )
    model.config.architectures = [type(model).__name__]
    config = load_recipe(_RECIPE).quantize.model_dump(exclude_unset=True)
    mtq.quantize(model, config)

    export_dir = tmp_path / "checkpoint"
    export_hf_checkpoint(model, export_dir=export_dir)

    state = load_file(export_dir / "model.safetensors")
    iq2_xs_weights = {
        name: tensor
        for name, tensor in state.items()
        if tensor.dtype == torch.uint8 and tensor.shape[-1] == 74
    }
    assert iq2_xs_weights
    assert all(
        ".mixer.experts." in name or ".mixer.shared_experts." in name for name in iq2_xs_weights
    )
    assert any(".mixer.experts." in name for name in iq2_xs_weights)
    assert any(".mixer.shared_experts." in name for name in iq2_xs_weights)
    for name, weight in iq2_xs_weights.items():
        base = name.removesuffix(".weight")
        logical_shape = state[base + ".weight_logical_shape"]
        padded_shape = state[base + ".weight_padded_shape"]
        assert padded_shape[-1] == weight.shape[-2] * 256
        assert torch.all(padded_shape >= logical_shape)

    with open(export_dir / "config.json") as file:
        quantization_config = json.load(file)["quantization_config"]
    assert quantization_config["quant_algo"] == "MIXED_PRECISION"
    weights = [group["weights"] for group in quantization_config["config_groups"].values()]
    assert any(weight.get("packing") == "ggml" and weight["num_bits"] == 2 for weight in weights)
    assert any(weight.get("row_padding") == "right" for weight in weights)
    assert any(weight["num_bits"] == 4 and weight["group_size"] == 16 for weight in weights)
