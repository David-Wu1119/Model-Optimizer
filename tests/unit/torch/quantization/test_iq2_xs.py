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


def test_iq2_xs_right_pads_each_row_without_crossing_row_boundaries():
    generator = torch.Generator().manual_seed(4321)
    weight = torch.randn((2, 257), generator=generator, dtype=torch.bfloat16)
    explicitly_padded = torch.nn.functional.pad(weight, (0, 255))

    packed, shape = quantize_iq2_xs(weight, block_chunk_size=2)
    expected, _ = quantize_iq2_xs(explicitly_padded, block_chunk_size=2)

    assert packed.shape == (2, 2, IQ2_XS_BLOCK_BYTES)
    assert shape.tolist() == [2, 257]
    assert torch.equal(packed, expected)
    assert dequantize_iq2_xs(packed, shape).shape == weight.shape


def test_iq2_xs_rejects_scalar_weight():
    with pytest.raises(ValueError, match="at least one dimension"):
        quantize_iq2_xs(torch.tensor(1.0))


def test_iq2_xs_fake_quant_has_pass_through_gradient():
    class Quantizer:
        num_bits = "iq2_xs"
        backend_extra_args = {"search_impl": "auto"}

    weight = torch.randn(1, 256, requires_grad=True)
    output = iq2_xs_fake_quant(weight, Quantizer())
    output.sum().backward()

    assert torch.equal(weight.grad, torch.ones_like(weight))


def test_iq2_xs_tensor_quantizer_matches_row_padded_backend(monkeypatch):
    generator = torch.Generator().manual_seed(5918)
    weight = torch.randn((2, 257), generator=generator, dtype=torch.bfloat16)
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits="iq2_xs",
            block_sizes={-1: 256},
            backend="ggml",
            backend_extra_args={"search_impl": "auto"},
        )
    )

    packed_by_fake_quant = []
    original_quantize = quantize_iq2_xs

    def capture_quantize(inputs):
        packed, shape = original_quantize(inputs)
        packed_by_fake_quant.append(packed)
        return packed, shape

    monkeypatch.setattr(iq2_xs_module, "quantize_iq2_xs", capture_quantize)
    reconstructed = quantizer(weight)
    packed, shape = original_quantize(weight)
    expected = dequantize_iq2_xs(packed, shape, dtype=weight.dtype)

    assert len(packed_by_fake_quant) == 1
    assert torch.equal(packed_by_fake_quant[0].reshape_as(packed), packed)
    torch.testing.assert_close(reconstructed, expected, rtol=0, atol=0.0078125)
