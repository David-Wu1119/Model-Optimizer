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

"""Shared validation for GGML-compatible block quantizers."""

import torch

GGML_BLOCK_SIZE = 256


def padded_weight_shape(shape: tuple[int, ...] | torch.Size) -> tuple[int, ...]:
    """Return ``shape`` with its last dimension rounded up to one GGML block."""
    if not shape:
        raise ValueError("GGML block quantization requires a tensor with at least one dimension")
    return (*shape[:-1], ((shape[-1] + GGML_BLOCK_SIZE - 1) // GGML_BLOCK_SIZE) * GGML_BLOCK_SIZE)


def pad_weight_rows(weight: torch.Tensor) -> torch.Tensor:
    """Right-pad every logical row to a complete GGML block."""
    padded_shape = padded_weight_shape(weight.shape)
    padding = padded_shape[-1] - weight.shape[-1]
    return torch.nn.functional.pad(weight, (0, padding)) if padding else weight.contiguous()


def validate_weight(weight: torch.Tensor, format_name: str) -> None:
    """Validate a weight accepted by the current GGML block encoders."""
    if weight.numel() == 0:
        raise ValueError(f"{format_name} requires a non-empty weight")
    if weight.dim() == 0:
        raise ValueError(f"{format_name} requires a tensor with at least one dimension")
    if not weight.is_floating_point():
        raise TypeError(f"{format_name} requires a floating-point weight, got {weight.dtype}")
    if not torch.isfinite(weight).all():
        raise ValueError(f"{format_name} requires finite weight values")


def validate_packed_weights(
    packed_weights: torch.Tensor,
    weight_shape: torch.Tensor,
    *,
    block_bytes: int,
    format_name: str,
) -> tuple[int, ...]:
    """Validate a packed payload and return its logical shape."""
    if packed_weights.dtype != torch.uint8 or packed_weights.shape[-1] != block_bytes:
        raise ValueError(
            f"packed_weights must be uint8 with last dimension {block_bytes}, "
            f"got {packed_weights.dtype} {tuple(packed_weights.shape)}"
        )
    shape = tuple(int(v) for v in weight_shape.detach().cpu().tolist())
    if not shape or any(v <= 0 for v in shape):
        raise ValueError(f"invalid {format_name} logical weight shape: {shape}")
    padded_shape = padded_weight_shape(shape)
    expected_shape = (*shape[:-1], padded_shape[-1] // GGML_BLOCK_SIZE, block_bytes)
    if tuple(packed_weights.shape) != expected_shape:
        raise ValueError(
            f"packed_weights shape does not match {format_name} logical weight shape: "
            f"expected {expected_shape}, got {tuple(packed_weights.shape)}"
        )
    return shape
