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

"""Fake quantization of the GatedDeltaNet (GDN) recurrent state.

The chunked gated-delta-rule kernel keeps each head's ``[K, V]`` recurrent state in fp32 inside
one Triton launch and carries it from chunk to chunk. To emulate a deployment that stores that
state in FP8, ModelOpt runs an adapted copy of the kernel
(:mod:`modelopt.torch.kernels.quantization.linear_attention`) that fake-quantizes the state to
E4M3 at the end of every chunk, with a scale computed inside the kernel from the state itself.
The backward pass recomputes the same quantized states and passes the state gradient straight
through the quantization, so QAT and QAD train against the quantized recurrence. A second
quantizer covers ``w``, the WY-transformed keys that multiply the state; ``w`` is a regular tensor,
so it is fake-quantized by the ``TensorQuantizer`` itself before the kernel reads it.
"""

from collections.abc import Callable
from typing import Any

import torch

from ..config import QuantizerAttributeConfig
from ..linear_attention.config import LinearAttentionConfig
from ..linear_attention.matmul import LinearAttentionMatmulSites, _validate_operand_quantizer
from ..linear_attention.prefill import matmul_gdn
from ..linear_attention.validation import validate_gdn_quantizer
from ..nn import QuantModule, TensorQuantizer

__all__ = ["GatedDeltaNetStateQuantMixin"]

GatedDeltaRuleFn = Callable[..., tuple[torch.Tensor, torch.Tensor | None]]


def _state_qdq_chunk_gated_delta_rule() -> GatedDeltaRuleFn:
    # Imported on first use: flash-linear-attention is a heavy optional dependency that only the
    # enabled quantizer needs, and importing it warns on machines without a GPU.
    try:
        from modelopt.torch.kernels.quantization.linear_attention.fla_chunk_gated_delta_rule import (
            chunk_gated_delta_rule,
        )
    except ImportError as e:
        raise RuntimeError(
            "GDN fake quantization needs Triton and fla-core==0.5.1 on a CUDA "
            f"device; importing the state-quantizing kernel failed with {e!r}."
        ) from e
    return chunk_gated_delta_rule


class GatedDeltaNetStateQuantMixin(QuantModule):
    """Adds ``gdn_state_quantizer`` and ``gdn_w_quantizer`` to a GatedDeltaNet module.

    Subclasses route the module's chunked gated-delta-rule call through
    :meth:`_state_quantized_chunk_gated_delta_rule`. Both quantizers start disabled; enable them
    with ``quant_cfg`` entries on ``*gdn_state_quantizer`` / ``*gdn_w_quantizer`` such as the
    ``configs/ptq/units/gdn_state_fp8_dynamic`` and ``gdn_w_fp8_dynamic`` recipe units. The state
    quantizer carries the fused QDQ configuration. Both sites currently require dynamic
    E4M3 and identity STE. The execution policy is saved in ModelOpt metadata.
    """

    def _setup(self):
        self.gdn_state_quantizer = TensorQuantizer(QuantizerAttributeConfig(enable=False))
        self.gdn_w_quantizer = TensorQuantizer(QuantizerAttributeConfig(enable=False))
        self.linear_attn_sites = LinearAttentionMatmulSites()
        self.linear_attention_config = LinearAttentionConfig()

    @property
    def linear_attention_is_enabled(self) -> bool:
        """Whether any operand, state, or arithmetic policy changes the computation."""
        return (
            self.gdn_state_quantizer.is_enabled
            or self.gdn_w_quantizer.is_enabled
            or self.linear_attn_sites.is_enabled
            or bool(self.linear_attention_config.matmul or self.linear_attention_config.elementwise)
        )

    @property
    def gdn_state_qdq_block_v(self) -> int:
        """Value-column scale grouping from the saved execution policy."""
        return self.linear_attention_config.state.block_v

    def validate_linear_attention(self) -> None:
        """Reject numerical settings that the fused training path cannot implement."""
        for state, quantizer in ((True, self.gdn_state_quantizer), (False, self.gdn_w_quantizer)):
            if quantizer.is_enabled:
                if not state and self.linear_attention_config.backend == "matmul":
                    _validate_operand_quantizer(quantizer, "gdn_w_quantizer")
                else:
                    validate_gdn_quantizer(quantizer, state=state)
        self.linear_attn_sites.validate()
        if self.linear_attn_sites.is_enabled and self.linear_attention_config.backend != "matmul":
            raise ValueError("Additional GDN operand sites require backend='matmul'")

    def modelopt_post_restore(self, prefix: str = ""):
        """Validate restored quantizers before using the saved execution policy."""
        super().modelopt_post_restore(prefix)
        self.validate_linear_attention()

    def _state_quantized_chunk_gated_delta_rule(
        self, gated_delta_rule: GatedDeltaRuleFn, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Call ``gated_delta_rule`` or, if a quantizer is on, the vendored quantizing copy."""
        self.validate_linear_attention()
        quantize_state = self.gdn_state_quantizer.is_enabled and self.gdn_state_quantizer._if_quant
        quantize_w = self.gdn_w_quantizer.is_enabled
        if not self.linear_attention_is_enabled:
            return gated_delta_rule(*args, **kwargs)
        if getattr(gated_delta_rule, "__name__", None) != "chunk_gated_delta_rule":
            raise NotImplementedError(
                "GatedDeltaNet quantizers require the fla chunked kernel; the deterministic torch "
                f"kernel ({gated_delta_rule!r}) is not supported."
            )
        chunk_size = kwargs.pop("chunk_size", self.linear_attention_config.chunk_size)
        if chunk_size != self.linear_attention_config.chunk_size:
            raise ValueError("GDN fake quantization supports only chunk_size=64")
        if self.linear_attention_config.backend == "matmul":
            return matmul_gdn(
                *args,
                sites=self.linear_attn_sites,
                policy=self.linear_attention_config,
                w_quantizer=self.gdn_w_quantizer,
                state_qdq=quantize_state,
                chunk_size=chunk_size,
                **kwargs,
            )
        return _state_qdq_chunk_gated_delta_rule()(
            *args,
            chunk_size=chunk_size,
            state_qdq=int(quantize_state),
            state_qdq_block_v=self.gdn_state_qdq_block_v,
            w_quantizer=self.gdn_w_quantizer if quantize_w else None,
            **kwargs,
        )
