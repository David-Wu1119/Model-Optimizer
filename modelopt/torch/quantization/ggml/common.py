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
from collections.abc import Callable
from typing import Any

import torch

GGML_BLOCK_SIZE = 256


def _cache_identity(inputs: torch.Tensor) -> tuple[Any, tuple[Any, ...]]:
    """Return storage ownership plus a signature that works in every grad mode."""
    storage = inputs.untyped_storage()
    anchor = inputs
    while (base := getattr(anchor, "_base", None)) is not None:
        anchor = base
    try:
        version = anchor._version
    except RuntimeError:
        # Tensors created in inference mode do not track a version counter. Storage identity and
        # explicit invalidation still provide a stable cache key for frozen inference weights.
        version = None
    signature = (
        version,
        storage.data_ptr(),
        inputs.storage_offset(),
        tuple(inputs.shape),
        tuple(inputs.stride()),
        inputs.dtype,
        inputs.device,
    )
    return storage, signature


def cached_reconstruction(
    inputs: torch.Tensor,
    quantizer: Any,
    *,
    cache_namespace: str,
    quantize: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    dequantize: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Return a reconstruction, caching only the compact payload for frozen weights.

    Static block quantization reshapes the weight before backend dispatch, so the transient
    ``inputs`` view is not a stable cache key.  Follow its view chain to the owning tensor and
    use that tensor's version counter.  Cache reuse is restricted to eval mode; training always
    recomputes.  Callers that rewrite a weight without advancing its version counter must clear
    the quantizer cache explicitly.
    """
    cache = getattr(quantizer, "_quantizer_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        quantizer._quantizer_cache = cache

    storage_key = f"{cache_namespace}_storage"
    signature_key = f"{cache_namespace}_signature"
    packed_key = f"{cache_namespace}_packed"
    shape_key = f"{cache_namespace}_shape"

    storage, signature = _cache_identity(inputs)
    cache_hit = (
        not getattr(quantizer, "training", True)
        and storage_key in cache
        and cache.get(signature_key) == signature
    )

    if cache_hit:
        packed, shape = cache[packed_key], cache[shape_key]
    else:
        packed, _ = quantize(inputs)
        # The backend receives the final logical block view, so its Python shape is the exact
        # metadata needed by the decoder and avoids a device-to-host copy on every cache hit.
        shape = tuple(inputs.shape)
        if not getattr(quantizer, "training", True):
            # Keep the storage alive while the signature is cached. That prevents allocator reuse
            # from making a different tensor appear to own the same data pointer.
            cache[storage_key] = storage
            cache[signature_key] = signature
            cache[packed_key] = packed
            cache[shape_key] = shape
        else:
            quantizer._quantizer_cache = None
    reconstructed = dequantize(packed, shape, dtype=inputs.dtype)
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
    weight_shape: torch.Tensor | tuple[int, ...],
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
    if isinstance(weight_shape, torch.Tensor):
        shape = tuple(int(v) for v in weight_shape.detach().cpu().tolist())
    else:
        shape = tuple(int(v) for v in weight_shape)
    if not shape or shape[-1] % GGML_BLOCK_SIZE:
        raise ValueError(f"invalid {format_name} logical weight shape: {shape}")
    expected_payload_values = math.prod(shape) // GGML_BLOCK_SIZE * block_bytes
    if packed_weights.numel() != expected_payload_values:
        raise ValueError("packed_weights size does not match weight_shape")
    return shape
