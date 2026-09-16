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

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

import modelopt.torch.quantization.ggml.iq2_xs as iq2_xs_module
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.ggml.iq2_xs import (
    IQ2_XS_BLOCK_BYTES,
    dequantize_iq2_xs,
    iq2_xs_fake_quant,
    iq2_xs_grid,
    quantize_iq2_xs,
)
from modelopt.torch.quantization.nn import TensorQuantizer


def test_iq2_xs_canonical_grid():
    grid = iq2_xs_grid()

    assert grid.shape == (512, 8)
    assert grid.dtype == torch.float32
    assert set(grid.unique().tolist()) == {8.0, 25.0, 43.0}
    assert grid[0].tolist() == [8.0] * 8
    assert grid[-1].tolist() == [43.0] * 8
    assert grid.unique(dim=0).shape[0] == grid.shape[0]


def test_iq2_xs_fake_grid_does_not_poison_real_grid_cache():
    iq2_xs_module._GRID_CACHE.clear()
    with FakeTensorMode():
        fake_grid = iq2_xs_grid()

    assert isinstance(fake_grid, FakeTensor)
    assert not iq2_xs_module._GRID_CACHE
    assert not isinstance(iq2_xs_grid(), FakeTensor)


def test_iq2_xs_zero_block_has_canonical_zero_encoding():
    weight = torch.zeros((2, 256), dtype=torch.bfloat16)

    packed, shape = quantize_iq2_xs(weight)

    assert packed.shape == (2, 1, IQ2_XS_BLOCK_BYTES)
    assert packed.dtype == torch.uint8
    assert not packed.any()
    assert shape.tolist() == [2, 256]
    assert torch.equal(dequantize_iq2_xs(packed, shape), weight)


def test_iq2_xs_round_trip_and_payload_fields():
    generator = torch.Generator().manual_seed(1234)
    weight = torch.randn((2, 512), generator=generator, dtype=torch.bfloat16)

    packed, shape = quantize_iq2_xs(weight, block_chunk_size=2)
    reconstructed = dequantize_iq2_xs(packed, shape)

    assert packed.shape == (2, 2, 74)
    assert reconstructed.shape == weight.shape
    assert reconstructed.dtype == torch.bfloat16
    normalized_mse = (
        reconstructed.float() - weight.float()
    ).square().mean() / weight.float().square().mean()
    assert normalized_mse < 0.1

    blocks = packed.reshape(-1, 74)
    codes = blocks[:, 2:66:2].to(torch.int64) | (blocks[:, 3:66:2].to(torch.int64) << 8)
    assert torch.all((codes & 0x1FF) < 512)
    assert torch.all((codes >> 9) < 128)


def test_iq2_xs_encoding_is_independent_of_default_dtype():
    weight = torch.randn((1, 256), generator=torch.Generator().manual_seed(4321))
    expected, _ = quantize_iq2_xs(weight)
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        actual, _ = quantize_iq2_xs(weight)
    finally:
        torch.set_default_dtype(original_dtype)

    assert torch.equal(actual, expected)


def test_iq2_xs_requires_complete_last_dimension_blocks():
    with pytest.raises(ValueError, match="last weight dimension"):
        quantize_iq2_xs(torch.ones(2, 257))


def test_iq2_xs_fake_quant_has_pass_through_gradient():
    class Quantizer:
        num_bits = "iq2_xs"
        backend_extra_args = {"search_impl": "auto"}
        _quantizer_cache = None

    weight = torch.randn(1, 256, requires_grad=True)
    output = iq2_xs_fake_quant(weight, Quantizer())
    output.sum().backward()

    assert torch.equal(weight.grad, torch.ones_like(weight))


def test_iq2_xs_fake_quant_reuses_cached_reconstruction(monkeypatch):
    calls = 0
    original_quantize = iq2_xs_module._quantize_iq2_xs_packed

    def counting_quantize(weight):
        nonlocal calls
        calls += 1
        return original_quantize(weight)

    monkeypatch.setattr(iq2_xs_module, "_quantize_iq2_xs_packed", counting_quantize)
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits="iq2_xs",
            block_sizes={-1: 256},
            backend="ggml",
            backend_extra_args={"search_impl": "auto"},
        )
    ).eval()
    quantizer.freeze_quantizer_cache()
    weight = torch.randn(1, 256)

    quantizer(weight)
    quantizer(weight)
    assert calls == 1

    with torch.no_grad():
        weight.add_(1)
    quantizer(weight)
    assert calls == 2

    weight.data = torch.randn_like(weight)
    quantizer(weight)
    assert calls == 3

    weight.data.add_(1)
    quantizer.reset_amax()
    assert quantizer._reconstruction_cache is None
    quantizer.freeze_quantizer_cache()
    quantizer(weight)
    assert calls == 4

    quantizer(weight.clone())
    assert calls == 5


def test_iq2_xs_fake_quant_handles_inference_tensors_without_a_version(monkeypatch):
    calls = 0
    original_quantize = iq2_xs_module._quantize_iq2_xs_packed

    def counting_quantize(weight):
        nonlocal calls
        calls += 1
        return original_quantize(weight)

    monkeypatch.setattr(iq2_xs_module, "_quantize_iq2_xs_packed", counting_quantize)
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits="iq2_xs",
            block_sizes={-1: 256},
            backend="ggml",
            backend_extra_args={"search_impl": "auto"},
        )
    ).eval()
    quantizer.freeze_quantizer_cache()

    with torch.inference_mode():
        weight = torch.randn(1, 256)
        quantizer(weight)
        quantizer(weight)
        weight.add_(1)
        quantizer.clear_quantizer_cache()
        quantizer(weight)

    assert calls == 2
