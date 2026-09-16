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

import torch

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
    shape = torch.tensor(weight.shape, dtype=torch.int64, device="cuda")
    reconstructed = dequantize_iq2_xs(packed, shape)

    assert packed.shape == (257, 1, 74)
    normalized_mse = (
        reconstructed.float() - weight.float()
    ).square().mean() / weight.float().square().mean()
    assert normalized_mse < 0.1


def test_iq2_xs_cuda_payload_matches_cpu_reference_and_public_dispatch():
    generator = torch.Generator().manual_seed(5678)
    weight_cpu = torch.randn((4, 256), generator=generator, dtype=torch.bfloat16)
    reference, _ = quantize_iq2_xs(weight_cpu)
    weight_cuda = weight_cpu.cuda()

    extension = _extension()
    grid = iq2_xs_grid("cuda")
    direct = extension.pack(weight_cuda, grid).reshape(4, 1, 74)
    direct_again = extension.pack(weight_cuda, grid).reshape(4, 1, 74)
    dispatched, _ = quantize_iq2_xs(weight_cuda)

    assert torch.equal(direct.cpu(), reference)
    assert torch.equal(direct_again, direct)
    assert torch.equal(dispatched, direct)


def test_iq2_xs_cuda_zero_encoding_matches_ggml_block_layout():
    weight = torch.zeros((1, 256), device="cuda", dtype=torch.bfloat16)
    packed = _extension().pack(weight, iq2_xs_grid("cuda")).reshape(1, 1, 74)
    shape = torch.tensor(weight.shape, dtype=torch.int64, device="cuda")

    assert not packed.any()
    assert torch.equal(dequantize_iq2_xs(packed, shape), weight)
