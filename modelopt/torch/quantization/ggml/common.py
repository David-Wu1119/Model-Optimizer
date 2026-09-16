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

import math
import weakref
from collections.abc import Callable
from typing import Any

import torch

GGML_BLOCK_SIZE = 256


def cached_reconstruction(
    inputs: torch.Tensor,
    quantizer: Any,
    *,
    cache_namespace: str,
    quantize: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    dequantize: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Return a cached reconstruction while an input tensor remains unchanged."""
    cache = getattr(quantizer, "_quantizer_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        quantizer._quantizer_cache = cache

    input_key = f"{cache_namespace}_input"
    signature_key = f"{cache_namespace}_signature"
    value_key = f"{cache_namespace}_value"
    signature = (
        inputs._version,
        tuple(inputs.shape),
        tuple(inputs.stride()),
        inputs.dtype,
        inputs.device,
    )
    input_ref = cache.get(input_key)
    if (
        isinstance(input_ref, weakref.ReferenceType)
        and input_ref() is inputs
        and cache.get(signature_key) == signature
    ):
        return cache[value_key]

    packed, shape = quantize(inputs)
    reconstructed = dequantize(packed, shape, dtype=inputs.dtype)
    cache[input_key] = weakref.ref(inputs)
    cache[signature_key] = signature
    cache[value_key] = reconstructed
    return reconstructed


def validate_weight(weight: torch.Tensor, format_name: str) -> None:
    """Validate a weight accepted by the current GGML block encoders."""
    if weight.numel() == 0:
        raise ValueError(f"{format_name} requires a non-empty weight")
    if weight.dim() == 0 or weight.shape[-1] % GGML_BLOCK_SIZE:
        raise ValueError(
            f"{format_name} requires the last weight dimension to be divisible by "
            f"{GGML_BLOCK_SIZE}, got shape {tuple(weight.shape)}"
        )
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
    if not shape or shape[-1] % GGML_BLOCK_SIZE:
        raise ValueError(f"invalid {format_name} logical weight shape: {shape}")
    expected_payload_values = math.prod(shape) // GGML_BLOCK_SIZE * block_bytes
    if packed_weights.numel() != expected_payload_values:
        raise ValueError("packed_weights size does not match weight_shape")
    return shape
