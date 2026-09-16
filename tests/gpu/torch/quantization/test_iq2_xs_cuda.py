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
from modelopt.torch.quantization.extensions import get_cuda_ext_iq2_xs
from modelopt.torch.quantization.ggml.iq2_xs import dequantize_iq2_xs, iq2_xs_grid, quantize_iq2_xs


def _extension():
    extension = get_cuda_ext_iq2_xs(raise_if_failed=True)
    assert extension is not None
    return extension


def test_iq2_xs_cuda_extension_handles_multiple_scale_grid_blocks():
    generator = torch.Generator(device="cuda").manual_seed(1234)
    weight = torch.randn((257, 256), generator=generator, device="cuda", dtype=torch.bfloat16)

    packed = _extension().pack(weight, iq2_xs_grid("cuda")).reshape(257, 1, 74)
    reference, _ = quantize_iq2_xs(weight.cpu())
    shape = torch.tensor(weight.shape, dtype=torch.int64, device="cuda")
    reconstructed = dequantize_iq2_xs(packed, shape)

    assert packed.shape == (257, 1, 74)
    torch.testing.assert_close(
        packed.cpu()[..., :2].contiguous().view(torch.float16).float(),
        reference[..., :2].contiguous().view(torch.float16).float(),
        rtol=1e-3,
        atol=0,
    )
    normalized_mse = (
        reconstructed.float() - weight.float()
    ).square().mean() / weight.float().square().mean()
    assert normalized_mse < 0.1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_iq2_xs_cuda_quality_matches_cpu_reference_and_public_dispatch(dtype):
    generator = torch.Generator().manual_seed(5678)
    weight_cpu = torch.randn((4, 512), generator=generator, dtype=dtype)
    reference, _ = quantize_iq2_xs(weight_cpu)
    weight_cuda = weight_cpu.cuda()

    extension = _extension()
    grid = iq2_xs_grid("cuda")
    direct = extension.pack(weight_cuda, grid).reshape(4, 2, 74)
    direct_again = extension.pack(weight_cuda, grid).reshape(4, 2, 74)
    dispatched, _ = quantize_iq2_xs(weight_cuda)

    shape = torch.tensor(weight_cpu.shape, dtype=torch.int64, device="cuda")
    reference_reconstructed = dequantize_iq2_xs(reference.cuda(), shape).float()
    direct_reconstructed = dequantize_iq2_xs(direct, shape).float()
    reference_error = (reference_reconstructed - weight_cuda.float()).square().mean()
    direct_error = (direct_reconstructed - weight_cuda.float()).square().mean()

    assert direct.shape == reference.shape
    assert direct.dtype == torch.uint8
    direct_scale = direct.cpu()[..., :2].contiguous().view(torch.float16).float()
    reference_scale = reference[..., :2].contiguous().view(torch.float16).float()
    torch.testing.assert_close(direct_scale, reference_scale, rtol=1e-3, atol=0)
    assert direct_error <= reference_error * 1.02
    assert torch.equal(direct_again, direct)
    assert torch.equal(dispatched, direct)


def test_iq2_xs_cuda_uses_reference_fallback_when_extension_is_unavailable(monkeypatch):
    monkeypatch.setattr(iq2_xs_module.extensions, "get_cuda_ext_iq2_xs", lambda: None)
    weight = torch.randn((1, 256), device="cuda", dtype=torch.bfloat16)

    packed, shape = quantize_iq2_xs(weight, block_chunk_size=1)

    assert packed.shape == (1, 1, 74)
    assert torch.equal(shape, torch.tensor([1, 256], device="cuda"))
    assert dequantize_iq2_xs(packed, shape).shape == weight.shape


def test_iq2_xs_cuda_zero_encoding_matches_ggml_block_layout():
    weight = torch.zeros((1, 256), device="cuda", dtype=torch.bfloat16)
    packed = _extension().pack(weight, iq2_xs_grid("cuda")).reshape(1, 1, 74)
    shape = torch.tensor(weight.shape, dtype=torch.int64, device="cuda")

    assert not packed.any()
    assert torch.equal(dequantize_iq2_xs(packed, shape), weight)
