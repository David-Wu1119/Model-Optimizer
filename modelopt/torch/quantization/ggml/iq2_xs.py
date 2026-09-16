# This file includes the IQ2_XS codebook adapted from:
# https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-common.h
#
# MIT License
#
# Copyright (c) 2023-2026 The ggml authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
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

"""IQ2_XS fake quantization and GGML-compatible block packing.

The encoder jointly searches grid rows, local scales, and sign patterns using
a heuristic super-block scale. Every 256 logical values become one 74-byte
block_iq2_xs payload:

* bytes 0..1: little-endian FP16 super-block scale d
* bytes 2..65: 32 little-endian uint16 codes (9-bit grid + 7-bit sign)
* bytes 66..73: 16 four-bit local scales, two per byte

The canonical 512 x 8 magnitude grid below comes from llama.cpp
ggml-common.h revision 9b05354ec6fb58b4e665e9a39ebc40285c015638. The payload layout is compatible
with GGML readers, but another encoder may choose different valid entries.
"""

import base64
import hashlib
from functools import cache

import torch

from .common import (
    GGML_BLOCK_SIZE,
    _cached_grid_from_bytes,
    cached_reconstruction,
    detect_fake_mode,
    validate_packed_weights,
    validate_weight,
)

__all__ = [
    "IQ2_XS_BLOCK_BYTES",
    "IQ2_XS_BLOCK_SIZE",
    "dequantize_iq2_xs",
    "iq2_xs_fake_quant",
    "iq2_xs_grid",
    "quantize_iq2_xs",
]

IQ2_XS_BLOCK_SIZE = GGML_BLOCK_SIZE
IQ2_XS_BLOCK_BYTES = 74
_IQ2_XS_DECODE_BLOCK_CHUNK_SIZE = 4096
_IQ2_XS_SUPPORTED_BACKEND_EXTRA_ARGS = frozenset({"search_impl"})
_IQ2_XS_NATIVE_MAX = 43 * 31 / 8
_IQ2_XS_PEAK_TO_RMS_SLOPE = 0.035
_IQ2_XS_MIN_ANCHOR = 0.65
_IQ2_XS_MAX_ANCHOR = 0.92
_IQ2_XS_GRID_SHA256 = "06e47aaca60b4dc1d9b5a3f34540437058a6b142b4d7a59d5ded769b4d1bf1de"

# Compact byte representation of the canonical [512, 8] grid. Values are only
# 8, 25, and 43. Keeping this as checkpoint-independent package data avoids
# adding a pickle-backed torch.save artifact to the wheel.
_IQ2_XS_GRID_B64 = (
    "CAgICAgICAgrCAgICAgICBkZCAgICAgICCsICAgICAgrKwgICAgICBkIGQgICAgICBkZCAgICAgrGRkICAgICBkrGQgICAgICAgr"
    "CAgICAgrCCsICAgICBkZKwgICAgICCsrCAgICAgZCAgZCAgICAgZCBkICAgIKxkIGQgICAgZKwgZCAgICAgIGRkICAgIKwgZGQgI"
    "CAgZGRkZCAgICAgrGRkICAgIGQgrGQgICAgIGSsZCAgICAgICCsICAgIKwgIKwgICAgZGQgrCAgICAgrCCsICAgIGQgZKwgICAgI"
    "GRkrCAgICBkrGSsICAgICAgrKwgICAgZCAgIGQgICAgZCAgZCAgIKxkICBkICAgZKwgIGQgICAgIGQgZCAgIKwgZCBkICAgZGRkI"
    "GQgICAgrGQgZCAgIKysZCBkICAgZCCsIGQgICAgZKwgZCAgICAgIGRkICAgrCAgZGQgICBkZCBkZCAgICCsIGRkICAgZCBkZGQgI"
    "CAgZGRkZCAgICAgrGRkICAgIKysZGQgICBkICCsZCAgICBkIKxkICAgICBkrGQgICAgICAgrCAgIKwgICCsICAgZGQgIKwgICAgr"
    "CAgrCAgIGQgZCCsICAgIGRkIKwgICAgIKwgrCAgIGQgIGSsICAgIGQgZKwgICAgIGRkrCAgIGRkZGSsICAgICAgrKwgICCsrCCsr"
    "CAgIGQgICAgZCAgIGQgICBkICCsZCAgIGQgIGSsICAgZCAgICBkICBkICCsIGQgIGQgIGRkZCAgZCAgIKxkICBkICBkIKwgIGQgI"
    "CBkrCAgZCAgICAgZCBkICCsICBkIGQgIGRkIGQgZCAgIKwgZCBkICBkIGRkIGQgICBkZGQgZCAgrGRkZCBkICAgIKxkIGQgIGQgI"
    "KwgZCAgIGQgrCBkICAgIGSsIGQgICAgICBkZCAgrCAgIGRkICBkZCAgZGQgICCsICBkZCAgZCBkIGRkICAgZGQgZGQgICAgrCBkZ"
    "CAgZCAgZGRkICAgZCBkZGQgICAgZGRkZCAgZCCsZGRkICAgICCsZGQgIGQgICCsZCAgIGQgIKxkICAgIGQgrGQgIKxkrCCsZCAgI"
    "CAgZKxkICCsICBkrGQgICBkIKysZCAgICAgICCsICCsICAgIKwgIGRkICAgrCAgIKwgICCsICCsrCAgIKwgIGQgZCAgrCAgIGRkI"
    "CCsICAgIKwgIKwgIGRkrCAgrCAgZCAgZCCsICAgZCBkIKwgICAgZGQgrCAgIKxkZCCsICAgICCsIKwgICAgrKwgrCAgrKysrCCsI"
    "CBkICAgZKwgICBkICBkrCAgICBkIGSsICAgICBkZKwgIGQgIKxkrCAgZKwgrGSsICAgICAgrKwgICAgrCCsrCAgIKysIKysICCsZ"
    "GSsrKwgICAgrKysrCAgZCAgICAgZCAgZCAgICBkIKxkICAgIGQgZKwgICAgZCAgIGQgICBkIKwgZCAgIGQgZGRkICAgZCAgrGQgI"
    "CBkIGQgrCAgIGQgIGSsICAgZCAgICBkICBkIKwgIGQgIGQgZGQgZCAgZCAgrCBkICBkIGQgZGQgIGQgIGRkZCAgZCAgIKxkICBkI"
    "KysrGQgIGQgZCAgrCAgZCAgZCCsICBkICAgZKwgIGQgICAgIGQgZCCsICAgZCBkIGRkICBkIGQgIKwgIGQgZCBkIGQgZCBkICBkZ"
    "CBkIGQgICCsIGQgZCBkICBkZCBkICBkIGRkIGQgICBkZGQgZCAgICCsZCBkICBkZKxkIGQgrGRkrGQgZCBkICAgrCBkICBkICCsI"
    "GQgrGQgIKwgZCAgIGQgrCBkICAgIGSsIGQgICCsZKwgZCAgICAgIGRkIKwgICAgZGQgZGQgICBkZCAgrCAgIGRkIGQgZCAgZGQgI"
    "GRkICBkZCAgIKwgIGRkIGQgIGQgZGQgIGQgZCBkZCBkrCBkIGRkICAgZGQgZGQgIGSsZCBkZCAgICCsIGRkIGQgICBkZGQgIGQgI"
    "GRkZCAgIGQgZGRkICAgIGRkZGQgICAgIKxkZCAgZGQgrGRkIGSsIGSsZGQgZCAgICCsZCAgZCAgIKxkICAgZCAgrGQgrCBkICCsZ"
    "CAgICBkIKxkICBkZGQgrGQgrGQgrCCsZCAgICAgZKxkIGRkICBkrGQgrGSsZGSsZCBkIGRkrKxkIGSsrKysrGQgICAgICAgrCCsI"
    "CAgICCsIGRkICAgIKwgIKwgICAgrCCsrCAgICCsIGQgZCAgIKwgIGRkICAgrCAgIKwgICCsIGQgIGQgIKwgIGQgZCAgrCAgIGRkI"
    "CCsICAgIKwgIKwgICCsrCAgrCBkICAgZCCsICBkICBkIKwgICBkIGQgrCAgICBkZCCsICCsIGRkIKwgZGSsZGQgrCAgICAgrCCsI"
    "KwgrCCsIKwgICAgrKwgrCAgrKysrCCsIGQgICAgZKwgIGQgICBkrCAgIGQgIGSsIGSsrCAgZKwgICAgZCBkrCAgICAgZGSsIGQgI"
    "GRkZKwgrCBkZGRkrCBkrGSsZGSsIGQgICCsZKwgrKxkIKxkrCCsZKysrGSsICAgICAgrKwgIKwgICCsrCCsrCAgIKysICAgrCAgr"
    "KwgZGRkZCCsrCAgrCCsIKysIKwgrKwgrKwgIKysZGSsrCAgIGSsZKysICCsICCsrKwgICCsIKysrCCsICCsrKysICCsIKysrKwgr"
    "KwgrKysrCBkICAgICAgZCBkICAgICBkrGQgICAgIGRkrCAgICAgZCAgZCAgICBkrCBkICAgIGRkZGQgICAgZCCsZCAgICBkZCCsI"
    "CAgIGQgZKwgICAgZCAgIGQgICBkrCAgZCAgIGRkZCBkICAgZCCsIGQgICBkrKwgZCAgIGRkIGRkICAgZCBkZGQgICBkICCsZCAgI"
    "GRkZKxkICAgZGQgIKwgICBkIGQgrCAgIGQgIGSsICAgZCAgICBkICBkrCAgIGQgIGRkZCAgZCAgZCCsICBkICBkZCBkIGQgIGQgZ"
    "GQgZCAgZCAgrCBkICBkZCAgZGQgIGQgZCBkZCAgZCAgZGRkICBkICAgrGQgIGRkZCCsZCAgZKwgrKxkICBkZCAgIKwgIGQgZCAgr"
    "CAgZCAgZCCsICBkrCBkIKwgIGRkrKwgrCAgZCAgIGSsICBkICAgICBkIGSsICAgIGQgZGRkICAgZCBkIKwgICBkIGRkIGQgIGQgZ"
    "CBkZCAgZCBkZKxkICBkIGQgIKwgIGQgZGQgIGQgZCBkIGQgZCBkIGQgIGRkIGQgZCAgIKwgZCBkIGRkrCBkIGRkICAgZGQgZCBkI"
    "CBkZCBkICBkIGRkIGQgZKwgZGQgZCAgIGRkZCBkrKxkrGRkIGQgICAgrGQgZKysICCsZCBkIGQgZKxkIGQgIGRkrGQgZGQgICAgr"
    "CBkIGQgICCsIGQgIGQgIKwgZCAgIGQgrCBkZGQgZCCsIGQgZGRkIKwgZKwgrGQgrCBkICAgIGSsIGRkIGQgZKwgZCBkIGRkrCBkI"
    "CBkZGSsIGRkrKxkZKwgZCBkICCsrCBkICAgICAgZGSsICAgICBkZGRkICAgIGRkIKwgICAgZGRkIGQgICBkZCBkZCAgIGRkICCsI"
    "CAgZGQgrKwgICBkZGQgIGQgIGRkIGQgZCAgZGQgIGRkICBkZCAgIKwgIGRkZCAgIGQgZGQgZCAgZCBkZCAgZCBkIGRkZGRkIGQgZ"
    "GQgICBkZCBkZKwgIGRkIGRkICAgIKwgZGQgZCBkrCBkZKysrKysIGRkZCAgICBkZGQgZCAgIGRkZCAgZCAgZGRkZCCsICBkZGQgI"
    "CBkIGRkZCAgrGQgZGRkZCAgrCBkZGRkIKysIGRkZCAgICBkZGRkIKwgIGRkZGQgICCsZGRkZCCsIKxkZGRkZCCsIKxkZGQgrKxkr"
    "GRkZGQgrKysZGRkICAgICCsZGQgZGQgIKxkZGQgIGQgrGRkICBkZCCsZGRkrGSsIKxkZKysZCBkrGRkICAgZGSsZGSsICBkZKxkZ"
    "GRkIKysrGRkZCAgICAgrGQgZCAgICCsZCAgZCAgIKxkICAgZCAgrGQgZGRkICCsZKwgrGQgIKxkrGQgrCAgrGRkrKysICCsZCAgI"
    "CBkIKxkIGSsIKwgrGSsrCBkrCCsZKwgZKysIKxkICAgICBkrGSsZGQgIGSsZCAgZCBkZKxkICAgZGRkrGRkZCBkZGSsZCBkrKxkZ"
    "KxkZCAgICCsrGSsrKxkIKysZGRkrCBkrKxkrGQgIKysrGQgZGRkrKysZKwgrGSsrKxkICAgICAgIKysICAgICAgrGRkICAgICCsI"
    "KwgICAgIKxkIGQgICAgrCBkZCAgICCsICCsICAgIKysrKwgICAgrGQgIGQgICCsIGQgZCAgIKwgIGRkICAgrCAgIKwgICCsrCAgr"
    "CAgIKwgrKysICAgrKysrKwgICCsZCAgIGQgIKwgZCAgZCAgrKxkICBkICCsICBkIGQgIKwgICBkZCAgrGQgZGRkICCsZKxkZGQgI"
    "KwgICAgrCAgrCAgrCCsICCsICAgrKwgIKysICCsrCAgrCAgrKysICCsIKysrKwgIKxkICAgIGQgrCBkICAgZCCsICBkICBkIKysI"
    "GQgIGQgrGRkZCAgZCCsICAgZCBkIKwgIKxkIGQgrGSsIKwgZCCsICAgIGRkIKwgZCBkZGQgrGRkrKxkZCCsIKxkIKxkIKysrKxkr"
    "GQgrCAgICAgrCCsIKwgICCsIKxkZKwgIKwgrKysZGQgrCCsICAgrCCsIKysICCsIKwgrCCsrKwgrCCsrGQgIGSsIKysIKwgrKwgr"
    "CAgIKysrCCsIKwgrKysIKysZGSsrKwgrCCsrKysrCCsZCAgICAgZKwgZCAgICBkrCAgZCAgIGSsICAgZCAgZKysZGRkICBkrCBkI"
    "KwgIGSsICAgIGQgZKysIKwgZCBkrCBkrGRkIGSsrGRkZKwgZKxkrCCsrCBkrCAgICAgZGSsZGQgICBkZKwgZCBkIGRkrCAgZGQgZ"
    "GSsIKxkZCBkZKxkrKwgZGRkrCAgZKxkZGSsrCBkrGRkZKxkICBkrGRkrGQgZGQgrGSsrGSsrCCsZKxkrCBkZKxkrGRkZCCsrGSsI"
    "CCsZKysZKwgICAgICCsrKwgICAgIKysIKwgICAgrKysrCAgICCsrCAgrCAgIKysrKysICAgrKwgIKysICCsrGQgZGRkIKysZKxkZ"
    "GQgrKysZKysZCCsrCAgICCsIKysrCAgIKwgrKwgrCAgrCCsrKysrCCsIKysICAgrKwgrKwgIKysrCCsrCAgIGQgZKysZGRkrCBkr"
    "KxkZKxkrGSsrCCsZKysZKysrKwgICCsrKwgIKwgIKysrKwgrCAgrKysIKysICCsrKwgIKysIKysrCCsrKwgrKysIGQgIGSsrKwgZ"
    "CCsZKysrKxkIKxkrKysIKysIKysrKysrKwgrKysrGQgZKysrKysrKysrKysrKw=="
)

_GRID_CACHE: dict[torch.device, torch.Tensor] = {}
_SIGN_TABLE_CACHE: dict[torch.device, torch.Tensor] = {}


@cache
def _grid_bytes() -> bytes:
    raw = base64.b64decode(_IQ2_XS_GRID_B64)
    if hashlib.sha256(raw).hexdigest() != _IQ2_XS_GRID_SHA256:
        raise RuntimeError("IQ2_XS grid checksum mismatch")
    return raw


def _cached_iq2_xs_grid(device: torch.device | str | None = None) -> torch.Tensor:
    """Return the private canonical grid used by pack and unpack paths."""
    return _cached_grid_from_bytes(_grid_bytes, 512, _GRID_CACHE, signed=False, device=device)


@cache
def _sign_table_bytes() -> bytes:
    """Return 128 rows of eight signs with bit 7 derived for even parity."""
    values = bytearray()
    for sign_index in range(128):
        sign_mask = sign_index | ((sign_index.bit_count() & 1) << 7)
        values.extend((-1 if sign_mask & (1 << bit) else 1) & 0xFF for bit in range(8))
    return bytes(values)


def _cached_iq2_xs_sign_table(device: torch.device | str | None = None) -> torch.Tensor:
    """Return the private [128, 8] sign table used by the unpack path."""
    return _cached_grid_from_bytes(
        _sign_table_bytes, 128, _SIGN_TABLE_CACHE, signed=True, device=device
    )


def iq2_xs_grid(device: torch.device | str | None = None) -> torch.Tensor:
    """Return a caller-owned copy of the canonical IQ2_XS magnitude grid as float32."""
    return _cached_iq2_xs_grid(device).clone()


def _encode_blocks(blocks: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Encode a moderate-size batch of flattened 256-value blocks."""
    x = blocks.float()
    block_count = x.shape[0]
    vectors = x.reshape(block_count, 32, 8)
    magnitudes = vectors.abs()
    negative = vectors < 0
    odd_parity = negative.sum(dim=-1).remainder(2).bool()

    amax = x.abs().amax(dim=1)
    rms = x.square().mean(dim=1).sqrt()
    peak_to_rms = torch.where(rms > 0, amax / rms, torch.zeros_like(rms))
    anchor_ratio = (1.0 - _IQ2_XS_PEAK_TO_RMS_SLOPE * peak_to_rms).clamp(
        _IQ2_XS_MIN_ANCHOR, _IQ2_XS_MAX_ANCHOR
    )
    d = ((amax / _IQ2_XS_NATIVE_MAX) * anchor_ratio).clamp(max=65504.0).to(torch.float16)
    d_float = d.float()

    xnorm = vectors.square().sum(dim=-1)
    qnorm = grid.square().sum(dim=-1)
    best_error = torch.full((block_count, 32, 16), torch.inf, dtype=torch.float32, device=x.device)
    best_entry = torch.zeros((block_count, 32, 16), dtype=torch.int64, device=x.device)
    # Search the codebook in tiles to cap temporary memory. Strict comparison
    # preserves the lowest grid index on equal error, matching the CUDA key.
    for entry_start in range(0, 512, 64):
        grid_tile = grid[entry_start : entry_start + 64]
        products = magnitudes.unsqueeze(2) * grid_tile.reshape(1, 1, -1, 8)
        dot = products.sum(dim=-1)
        dot = torch.where(odd_parity.unsqueeze(-1), dot - 2.0 * products.amin(dim=-1), dot)
        tile_qnorm = qnorm[entry_start : entry_start + 64].reshape(1, 1, -1)

        for local in range(16):
            scale = d_float.reshape(-1, 1, 1) * ((2 * local + 1) / 8.0)
            error = (
                xnorm.unsqueeze(-1) - 2.0 * scale * dot + scale.square() * tile_qnorm
            ).clamp_min_(0)
            tile_error, tile_index = error.min(dim=-1)
            replace = tile_error < best_error[:, :, local]
            best_error[:, :, local] = torch.where(replace, tile_error, best_error[:, :, local])
            best_entry[:, :, local] = torch.where(
                replace, tile_index + entry_start, best_entry[:, :, local]
            )

    group_error = best_error.reshape(block_count, 16, 2, 16).sum(dim=2)
    selected_local = group_error.argmin(dim=-1)
    vector_local = selected_local.repeat_interleave(2, dim=1)
    selected_entry = best_entry.gather(2, vector_local.unsqueeze(-1)).squeeze(-1)

    selected_grid = grid[selected_entry]
    weakest_index = (magnitudes * selected_grid).argmin(dim=-1)
    flip = torch.nn.functional.one_hot(weakest_index, num_classes=8).bool()
    encoded_negative = negative ^ (flip & odd_parity.unsqueeze(-1))
    sign_bits = torch.arange(8, dtype=torch.int64, device=x.device)
    sign_mask = (encoded_negative.to(torch.int64) << sign_bits).sum(dim=-1)

    codes = selected_entry | ((sign_mask & 0x7F) << 9)
    packed = torch.empty((block_count, IQ2_XS_BLOCK_BYTES), dtype=torch.uint8, device=x.device)
    packed[:, :2] = d.contiguous().view(torch.uint8).reshape(block_count, 2)
    packed[:, 2:66:2] = (codes & 0xFF).to(torch.uint8)
    packed[:, 3:66:2] = (codes >> 8).to(torch.uint8)
    packed[:, 66:] = (selected_local[:, 0::2] | (selected_local[:, 1::2] << 4)).to(torch.uint8)
    return packed


@torch.no_grad()
def _quantize_iq2_xs_packed(
    weight: torch.Tensor, *, block_chunk_size: int | None = None
) -> torch.Tensor:
    """Pack a weight without allocating the public logical-shape tensor."""
    validate_weight(weight, "IQ2_XS")
    if block_chunk_size is None:
        if detect_fake_mode(weight) is not None:
            # Tracing allocates no tensor storage, so chunking only unrolls the graph.
            block_chunk_size = max(1, weight.numel() // IQ2_XS_BLOCK_SIZE)
        else:
            block_chunk_size = 1024 if weight.is_cuda else 64
    if block_chunk_size <= 0:
        raise ValueError(f"block_chunk_size must be positive, got {block_chunk_size}")

    blocks = weight.contiguous().reshape(-1, IQ2_XS_BLOCK_SIZE)
    grid = _cached_iq2_xs_grid(weight.device)
    chunks = [
        _encode_blocks(blocks[start : start + block_chunk_size], grid)
        for start in range(0, blocks.shape[0], block_chunk_size)
    ]
    packed_shape = (
        *weight.shape[:-1],
        weight.shape[-1] // IQ2_XS_BLOCK_SIZE,
        IQ2_XS_BLOCK_BYTES,
    )
    return torch.cat(chunks).reshape(packed_shape)


@torch.no_grad()
def quantize_iq2_xs(
    weight: torch.Tensor, *, block_chunk_size: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a floating-point weight into GGML-compatible IQ2_XS blocks.

    Returned shapes are ``[*weight.shape[:-1], weight.shape[-1] // 256, 74]``
    and ``[weight.ndim]``. Both tensors remain on the weight's device.
    """
    packed = _quantize_iq2_xs_packed(weight, block_chunk_size=block_chunk_size)
    logical_shape = torch.tensor(weight.shape, dtype=torch.int64, device=weight.device)
    return packed, logical_shape


@torch.no_grad()
def dequantize_iq2_xs(
    packed_weights: torch.Tensor,
    weight_shape: torch.Tensor | tuple[int, ...],
    *,
    dtype: torch.dtype = torch.bfloat16,
    block_chunk_size: int | None = None,
) -> torch.Tensor:
    """Decode GGML-compatible IQ2_XS payload bytes."""
    shape = validate_packed_weights(
        packed_weights, weight_shape, block_bytes=IQ2_XS_BLOCK_BYTES, format_name="IQ2_XS"
    )

    blocks = packed_weights.contiguous().reshape(-1, IQ2_XS_BLOCK_BYTES)
    if block_chunk_size is None:
        block_chunk_size = (
            max(1, blocks.shape[0])
            if detect_fake_mode(packed_weights) is not None
            else _IQ2_XS_DECODE_BLOCK_CHUNK_SIZE
        )
    if block_chunk_size <= 0:
        raise ValueError(f"block_chunk_size must be positive, got {block_chunk_size}")

    decoded = torch.empty((blocks.shape[0], IQ2_XS_BLOCK_SIZE), dtype=dtype, device=blocks.device)
    grid = _cached_iq2_xs_grid(blocks.device)
    sign_table = _cached_iq2_xs_sign_table(blocks.device)
    for start in range(0, blocks.shape[0], block_chunk_size):
        chunk = blocks[start : start + block_chunk_size]
        d = chunk[:, :2].contiguous().view(torch.float16).reshape(-1).float()
        codes = chunk[:, 2:66:2].to(torch.int32) | (chunk[:, 3:66:2].to(torch.int32) << 8)
        entries = codes & 0x1FF
        signs = sign_table[codes >> 9]

        scale_bytes = chunk[:, 66:].to(torch.int32)
        local = torch.empty((chunk.shape[0], 16), dtype=torch.int32, device=blocks.device)
        local[:, 0::2] = scale_bytes & 0x0F
        local[:, 1::2] = scale_bytes >> 4
        scales = d.unsqueeze(-1) * (2 * local + 1).float() / 8.0
        values = grid[entries]
        values.mul_(signs)
        values.mul_(scales.repeat_interleave(2, dim=1).unsqueeze(-1))
        decoded[start : start + chunk.shape[0]].copy_(values.reshape(-1, IQ2_XS_BLOCK_SIZE))
    return decoded.reshape(shape)


def iq2_xs_fake_quant(inputs: torch.Tensor, quantizer) -> torch.Tensor:
    """IQ2_XS backend for TensorQuantizer, with pass-through backward."""
    if getattr(quantizer, "num_bits", None) != "iq2_xs":
        raise ValueError("The ggml IQ2_XS backend requires num_bits='iq2_xs'")
    extra_args = getattr(quantizer, "backend_extra_args", None) or {}
    unknown_keys = set(extra_args) - _IQ2_XS_SUPPORTED_BACKEND_EXTRA_ARGS
    if unknown_keys:
        raise ValueError(
            f"Unsupported IQ2_XS backend_extra_args keys: {sorted(unknown_keys)}; "
            f"supported: {sorted(_IQ2_XS_SUPPORTED_BACKEND_EXTRA_ARGS)}"
        )
    search_impl = extra_args.get("search_impl", "auto")
    if search_impl != "auto":
        raise NotImplementedError("Only IQ2_XS search_impl='auto' is currently supported")
    reconstructed = cached_reconstruction(
        inputs,
        quantizer,
        cache_namespace="iq2_xs",
        quantize=_quantize_iq2_xs_packed,
        dequantize=dequantize_iq2_xs,
    )
    return inputs + (reconstructed - inputs).detach()
