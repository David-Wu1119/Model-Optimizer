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

"""Conformance of the GGML IQ decoders against real llama.cpp-produced blocks."""

import numpy as np
import pytest
import torch
from _test_utils.torch.quantization.iq_llama_cpp_vectors import (
    expected_values,
    formats,
    packed_blocks,
)

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.ggml import (
    IQ1_M_BLOCK_BYTES,
    IQ1_S_BLOCK_BYTES,
    IQ2_S_BLOCK_BYTES,
    IQ2_XS_BLOCK_BYTES,
    IQ2_XXS_BLOCK_BYTES,
    dequantize_iq1_m,
    dequantize_iq1_s,
    dequantize_iq2_s,
    dequantize_iq2_xs,
    dequantize_iq2_xxs,
    quantize_iq1_m,
    quantize_iq2_s,
    quantize_iq2_xxs,
)
from modelopt.torch.quantization.nn import TensorQuantizer

DECODERS = {
    "iq1_s": (dequantize_iq1_s, IQ1_S_BLOCK_BYTES),
    "iq1_m": (dequantize_iq1_m, IQ1_M_BLOCK_BYTES),
    "iq2_xxs": (dequantize_iq2_xxs, IQ2_XXS_BLOCK_BYTES),
    "iq2_xs": (dequantize_iq2_xs, IQ2_XS_BLOCK_BYTES),
    "iq2_s": (dequantize_iq2_s, IQ2_S_BLOCK_BYTES),
}
NEW_FORMATS = {
    "iq1_m": (quantize_iq1_m, dequantize_iq1_m, IQ1_M_BLOCK_BYTES),
    "iq2_xxs": (quantize_iq2_xxs, dequantize_iq2_xxs, IQ2_XXS_BLOCK_BYTES),
    "iq2_s": (quantize_iq2_s, dequantize_iq2_s, IQ2_S_BLOCK_BYTES),
}


@pytest.mark.parametrize("name", formats())
def test_decoder_matches_llama_cpp_on_captured_blocks(name):
    """Decode bytes we did not produce and match llama.cpp's own output exactly.

    A round-trip against our own encoder cannot catch a layout error that the encoder
    makes symmetrically; these blocks come from a real checkpoint, so they can.
    """
    decode, block_bytes = DECODERS[name]
    blocks = packed_blocks(name)
    assert blocks.shape[1] == block_bytes

    count = blocks.shape[0]
    decoded = decode(
        torch.from_numpy(blocks).reshape(count, 1, block_bytes),
        torch.tensor([count, 256]),
        dtype=torch.float32,
    ).reshape(count, 256)

    assert np.array_equal(decoded.numpy(), expected_values(name))


@pytest.mark.parametrize("name", sorted(NEW_FORMATS))
def test_round_trip_is_close_and_shaped(name):
    quantize, dequantize, block_bytes = NEW_FORMATS[name]
    generator = torch.Generator().manual_seed(1234)
    weight = torch.randn((2, 512), generator=generator, dtype=torch.bfloat16)

    packed, shape = quantize(weight)
    reconstructed = dequantize(packed, shape)
    chunked = dequantize(packed, shape, block_chunk_size=1)

    assert packed.shape == (2, 2, block_bytes)
    assert reconstructed.shape == weight.shape
    assert reconstructed.dtype == torch.bfloat16
    # Chunking the decode is a memory bound, not a numerical choice.
    assert torch.equal(reconstructed, chunked)
    normalized_mse = (
        reconstructed.float() - weight.float()
    ).square().mean() / weight.float().square().mean()
    assert normalized_mse < 0.25


@pytest.mark.parametrize("name", sorted(NEW_FORMATS))
def test_zero_block_has_canonical_zero_encoding(name):
    quantize, dequantize, block_bytes = NEW_FORMATS[name]
    packed, shape = quantize(torch.zeros(1, 256))
    assert torch.equal(packed, torch.zeros_like(packed))
    assert torch.equal(dequantize(packed, shape), torch.zeros(1, 256, dtype=torch.bfloat16))


@pytest.mark.parametrize("name", sorted(NEW_FORMATS))
def test_requires_complete_last_dimension_blocks(name):
    quantize, _, _ = NEW_FORMATS[name]
    with pytest.raises(ValueError, match="last weight dimension"):
        quantize(torch.ones(2, 257))


@pytest.mark.parametrize("name", sorted(NEW_FORMATS))
def test_nonfinite_values_are_treated_as_zero(name):
    quantize, _, _ = NEW_FORMATS[name]
    weight = torch.zeros(1, 256)
    weight[0, 0] = float("nan")
    weight[0, 1] = float("inf")
    packed, _ = quantize(weight)
    assert torch.equal(packed, torch.zeros_like(packed))


@pytest.mark.parametrize("name", sorted(NEW_FORMATS))
def test_fake_quant_has_pass_through_gradient(name):
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=name, block_sizes={-1: 256}, backend="ggml")
    )
    weight = torch.randn(2, 256, requires_grad=True)
    quantizer(weight).sum().backward()
    assert torch.equal(weight.grad, torch.ones_like(weight))


def test_error_decreases_with_bit_width():
    """More bits must buy less error, or a format's scale handling is wrong."""
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn((4, 1024), generator=generator)
    errors = {}
    for name in ("iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s"):
        quantizer = TensorQuantizer(
            QuantizerAttributeConfig(num_bits=name, block_sizes={-1: 256}, backend="ggml")
        )
        reconstructed = quantizer(weight)
        errors[name] = float((reconstructed - weight).square().mean())

    ordered = [errors[n] for n in ("iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s")]
    assert ordered == sorted(ordered, reverse=True), errors
