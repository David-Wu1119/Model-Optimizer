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

import modelopt.torch.quantization.ggml.iq1_s as iq1_s_module
from modelopt.torch.quantization.extensions import get_cuda_ext_iq1_s
from modelopt.torch.quantization.ggml.iq1_s import dequantize_iq1_s, iq1_s_grid, quantize_iq1_s

pytestmark = pytest.mark.timeout(240)


def _extension():
    extension = get_cuda_ext_iq1_s(raise_if_failed=True)
    assert extension is not None
    return extension


def test_iq1_s_cuda_extension_handles_multiple_scale_grid_blocks():
    generator = torch.Generator(device="cuda").manual_seed(1234)
    weight = torch.randn((257, 256), generator=generator, device="cuda", dtype=torch.bfloat16)

    packed = _extension().pack(weight, iq1_s_grid("cuda")).reshape(257, 1, 50)
    reference, _ = quantize_iq1_s(weight.cpu())
    shape = torch.tensor(weight.shape, dtype=torch.int64, device="cuda")
    reconstructed = dequantize_iq1_s(packed, shape)

    assert packed.shape == (257, 1, 50)
    assert torch.equal(packed.cpu()[..., :2], reference[..., :2])
    normalized_mse = (
        reconstructed.float() - weight.float()
    ).square().mean() / weight.float().square().mean()
    assert normalized_mse < 0.25


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_iq1_s_cuda_quality_matches_cpu_reference_and_public_dispatch(dtype):
    generator = torch.Generator().manual_seed(5678)
    weight_cpu = torch.randn((4, 512), generator=generator, dtype=dtype)
    reference, _ = quantize_iq1_s(weight_cpu)
    weight_cuda = weight_cpu.cuda()

    extension = _extension()
    grid = iq1_s_grid("cuda")
    direct = extension.pack(weight_cuda, grid).reshape(4, 2, 50)
    direct_again = extension.pack(weight_cuda, grid).reshape(4, 2, 50)
    dispatched, _ = quantize_iq1_s(weight_cuda)

    shape = torch.tensor(weight_cpu.shape, dtype=torch.int64, device="cuda")
    reference_reconstructed = dequantize_iq1_s(reference.cuda(), shape).float()
    direct_reconstructed = dequantize_iq1_s(direct, shape).float()
    reference_error = (reference_reconstructed - weight_cuda.float()).square().mean()
    direct_error = (direct_reconstructed - weight_cuda.float()).square().mean()

    assert direct.shape == reference.shape
    assert direct.dtype == torch.uint8
    assert torch.equal(direct.cpu()[..., :2], reference[..., :2])
    assert direct_error <= reference_error * 1.02
    assert torch.equal(direct_again, direct)
    assert torch.equal(dispatched, direct)


def test_iq1_s_cuda_uses_reference_fallback_when_extension_is_unavailable(monkeypatch):
    getter_called = False

    def unavailable_extension():
        nonlocal getter_called
        getter_called = True

    monkeypatch.setattr(iq1_s_module.extensions, "get_cuda_ext_iq1_s", unavailable_extension)
    weight = torch.randn((1, 256), device="cuda", dtype=torch.bfloat16)
    expected = iq1_s_module._encode_blocks(weight.reshape(-1, 256), iq1_s_grid("cuda")).reshape(
        1, 1, 50
    )

    packed, shape = quantize_iq1_s(weight, block_chunk_size=1)

    assert getter_called
    assert packed.shape == (1, 1, 50)
    assert torch.equal(packed, expected)
    assert torch.equal(shape, torch.tensor([1, 256], device="cuda"))
    assert dequantize_iq1_s(packed, shape).shape == weight.shape


def test_iq1_s_cuda_zero_encoding_matches_ggml_block_layout():
    weight = torch.zeros((1, 256), device="cuda", dtype=torch.bfloat16)
    packed = _extension().pack(weight, iq1_s_grid("cuda")).reshape(1, 1, 50)
    shape = torch.tensor(weight.shape, dtype=torch.int64, device="cuda")

    assert not packed.any()
    assert torch.equal(dequantize_iq1_s(packed, shape), weight)
