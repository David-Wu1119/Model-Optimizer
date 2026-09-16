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

"""GGML-compatible block quantization formats."""

# Importing the backend installs its TensorQuantizer dispatch entry.
from . import backend as _backend
from .iq1_s import (
    IQ1_S_BLOCK_BYTES,
    IQ1_S_BLOCK_SIZE,
    IQ1_S_EFFECTIVE_BITS,
    dequantize_iq1_s,
    iq1_s_fake_quant,
    iq1_s_grid,
    quantize_iq1_s,
)
from .iq2_xs import (
    IQ2_XS_BLOCK_BYTES,
    IQ2_XS_BLOCK_SIZE,
    IQ2_XS_EFFECTIVE_BITS,
    dequantize_iq2_xs,
    iq2_xs_fake_quant,
    iq2_xs_grid,
    quantize_iq2_xs,
)

__all__ = [
    "IQ1_S_BLOCK_BYTES",
    "IQ1_S_BLOCK_SIZE",
    "IQ1_S_EFFECTIVE_BITS",
    "IQ2_XS_BLOCK_BYTES",
    "IQ2_XS_BLOCK_SIZE",
    "IQ2_XS_EFFECTIVE_BITS",
    "dequantize_iq1_s",
    "dequantize_iq2_xs",
    "iq1_s_fake_quant",
    "iq1_s_grid",
    "iq2_xs_fake_quant",
    "iq2_xs_grid",
    "quantize_iq1_s",
    "quantize_iq2_xs",
]
