/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>

at::Tensor iq2_xs_pack_cuda(at::Tensor input, at::Tensor grid, bool validate_grid);

at::Tensor iq2_xs_pack(at::Tensor input, at::Tensor grid) {
  TORCH_CHECK(input.is_cuda(), "IQ2_XS packing requires a CUDA input");
  TORCH_CHECK(grid.is_cuda(), "IQ2_XS packing requires a CUDA grid");
  return iq2_xs_pack_cuda(input.contiguous(), grid.contiguous(), true);
}

at::Tensor iq2_xs_pack_canonical(at::Tensor input, at::Tensor grid) {
  TORCH_CHECK(input.is_cuda(), "IQ2_XS packing requires a CUDA input");
  TORCH_CHECK(grid.is_cuda(), "IQ2_XS packing requires a CUDA grid");
  return iq2_xs_pack_cuda(input.contiguous(), grid.contiguous(), false);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("pack", &iq2_xs_pack,
             "Pack a CUDA floating-point tensor whose numel is a positive multiple of 256. "
             "Validates a float32 [512, 8] non-negative magnitude grid. Returns uint8 [numel / "
             "256, 74] on the input device; leading dimensions are flattened.");
  module.def("_pack_canonical", &iq2_xs_pack_canonical,
             "Same input and output contract as pack, but skips grid-value validation. Only for "
             "the checksum-verified canonical IQ2_XS grid.");
}
