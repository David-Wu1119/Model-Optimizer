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

"""Saved execution policy for GDN training-time numerical emulation."""

from typing import Literal

from pydantic import Field, model_validator

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField

__all__ = ["LinearAttentionConfig", "LinearAttentionMatmulConfig", "LinearAttentionPolicyEntry"]

_PrefillSite = Literal[
    "key_interaction",
    "wy_value",
    "wy_key",
    "state_read",
    "state_update",
    "output_state",
    "output_score",
    "output_value",
]
_ElementwiseSite = Literal[
    "gate_prefix",
    "gate_exp",
    "value_residual",
    "state_decay",
    "state_add",
    "output_add",
]
_ArithmeticDtype = Literal["float32", "float16", "bfloat16"]


class LinearAttentionMatmulConfig(ModeloptBaseConfig):
    """Round an accumulator after each left-to-right reduction block.

    Partial products use the baseline working dtype. This specifies an emulation
    schedule, not the internal accumulation order of a hardware MMA instruction.
    """

    accumulator_dtype: _ArithmeticDtype | None = ModeloptField(default=None)
    reduction_block: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _require_complete_schedule(self):
        if (self.accumulator_dtype is None) != (self.reduction_block is None):
            raise ValueError("accumulator_dtype and reduction_block must be specified together")
        return self


class _StateConfig(ModeloptBaseConfig):
    mode: Literal["chunk"] = ModeloptField(default="chunk")
    block_v: Literal[16, 32, 64, 128] = ModeloptField(default=64)
    quantize_initial: Literal[True] = ModeloptField(default=True)


class _SolveConfig(ModeloptBaseConfig):
    method: Literal["exact"] = ModeloptField(default="exact")


class LinearAttentionConfig(ModeloptBaseConfig):
    """GDN/KDA chunk-64 policy; unsupported numerical modes fail config validation.

    ``state.block_v`` defines one dynamic scale per ``[Dk, block_v]`` tile of each
    sequence/head. The initial state and every chunk's final state are rounded when
    the module's state quantizer is enabled. Outputs use the incoming rounded state.
    """

    schema_version: Literal[1] = ModeloptField(default=1)
    backend: Literal["fla", "matmul"] = ModeloptField(default="fla")
    chunk_size: Literal[64] = ModeloptField(default=64)
    state: _StateConfig = ModeloptField(default=_StateConfig())
    solve: _SolveConfig = ModeloptField(default=_SolveConfig())
    matmul: dict[_PrefillSite, LinearAttentionMatmulConfig] = ModeloptField(default={})
    elementwise: dict[_ElementwiseSite, _ArithmeticDtype] = ModeloptField(default={})

    @model_validator(mode="after")
    def _validate_arithmetic_backend(self):
        if self.backend == "fla" and (self.matmul or self.elementwise):
            raise ValueError("Prefill arithmetic policies require backend='matmul'")
        return self


class LinearAttentionPolicyEntry(ModeloptBaseConfig):
    """Assign a complete policy to supported modules matching ``module_name``.

    Rules apply in order: the last match wins, without merging nested fields.
    A rule must match at least one supported linear-attention module.
    """

    module_name: str = Field(...)
    cfg: LinearAttentionConfig = ModeloptField(default=LinearAttentionConfig())
