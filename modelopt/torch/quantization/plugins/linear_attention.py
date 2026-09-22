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

"""Shared module policy and checkpoint support for linear-attention QAT."""

from ..config import QuantizerAttributeConfig
from ..linear_attention.config import LinearAttentionConfig
from ..linear_attention.matmul import LinearAttentionMatmulSites, _validate_operand_quantizer
from ..linear_attention.validation import validate_gdn_quantizer
from ..nn import QuantModule, TensorQuantizer

__all__ = []


class _LinearAttentionQuantMixin(QuantModule):
    linear_attention_quantizer_names = ("gdn_state_quantizer", "gdn_w_quantizer")

    def _setup(self):
        for name in self.linear_attention_quantizer_names:
            setattr(self, name, TensorQuantizer(QuantizerAttributeConfig(enable=False)))
        self.linear_attn_sites = LinearAttentionMatmulSites()
        self.linear_attention_config = LinearAttentionConfig()

    @property
    def _linear_attn_state(self):
        return getattr(self, self.linear_attention_quantizer_names[0])

    @property
    def _linear_attn_w(self):
        return getattr(self, self.linear_attention_quantizer_names[1])

    @property
    def linear_attention_is_enabled(self):
        """Whether an operand, state, or arithmetic policy changes the computation."""
        return (
            self._linear_attn_state.is_enabled
            or self._linear_attn_w.is_enabled
            or self.linear_attn_sites.is_enabled
            or bool(self.linear_attention_config.matmul or self.linear_attention_config.elementwise)
        )

    def validate_linear_attention(self):
        """Validate quantizer contracts shared by GDN and KDA."""
        if self._linear_attn_state.is_enabled:
            validate_gdn_quantizer(
                self._linear_attn_state, state=True, name=self.linear_attention_quantizer_names[0]
            )
        if self.linear_attention_config.backend == "matmul":
            _validate_operand_quantizer(
                self._linear_attn_w, self.linear_attention_quantizer_names[1]
            )
        elif self._linear_attn_w.is_enabled:
            validate_gdn_quantizer(
                self._linear_attn_w, state=False, name=self.linear_attention_quantizer_names[1]
            )
        self.linear_attn_sites.validate()
        if self.linear_attn_sites.is_enabled and self.linear_attention_config.backend != "matmul":
            raise ValueError("Additional linear-attention operand sites require backend='matmul'")

    def modelopt_post_restore(self, prefix=""):
        """Validate the restored numerical policy and quantizers."""
        super().modelopt_post_restore(prefix)
        self.validate_linear_attention()
