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

from pydantic import Field

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField

__all__ = ["LinearAttentionConfig", "LinearAttentionPolicyEntry"]


class _StateConfig(ModeloptBaseConfig):
    mode: Literal["chunk"] = ModeloptField(default="chunk")
    block_v: Literal[16, 32, 64, 128] = ModeloptField(default=64)
    quantize_initial: Literal[True] = ModeloptField(default=True)


class _SolveConfig(ModeloptBaseConfig):
    method: Literal["exact"] = ModeloptField(default="exact")


class LinearAttentionConfig(ModeloptBaseConfig):
    """GDN chunk-64 policy; unsupported numerical modes fail config validation.

    ``state.block_v`` defines one dynamic scale per ``[Dk, block_v]`` tile of each
    sequence/head. The initial state and every chunk's final state are rounded when
    ``gdn_state_quantizer`` is enabled. Outputs use the incoming rounded state.
    """

    schema_version: Literal[1] = ModeloptField(default=1)
    backend: Literal["fla"] = ModeloptField(default="fla")
    chunk_size: Literal[64] = ModeloptField(default=64)
    state: _StateConfig = ModeloptField(default=_StateConfig())
    solve: _SolveConfig = ModeloptField(default=_SolveConfig())


class LinearAttentionPolicyEntry(ModeloptBaseConfig):
    """Assign a complete policy to supported modules matching ``module_name``.

    Rules apply in order: the last match wins, without merging nested fields.
    A rule must match at least one supported linear-attention module.
    """

    module_name: str = Field(...)
    cfg: LinearAttentionConfig = ModeloptField(default=LinearAttentionConfig())
