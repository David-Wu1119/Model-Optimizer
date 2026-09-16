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

#pragma once

#include <cstdint>

namespace modelopt {
namespace ggml_pack {

template <typename scalar_t> __device__ __forceinline__ float load_float(const scalar_t *input) {
  return static_cast<float>(*input);
}

__device__ __forceinline__ float warp_min(float value) {
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1)
    value = fminf(value, __shfl_down_sync(0xffffffff, value, delta));
  return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1)
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, delta));
  return value;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1)
    value += __shfl_down_sync(0xffffffff, value, delta);
  return value;
}

__device__ __forceinline__ unsigned long long warp_min_key(unsigned long long key) {
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    const auto other = __shfl_down_sync(0xffffffff, key, delta);
    key = other < key ? other : key;
  }
  return key;
}

template <int Items, int Threads>
__device__ __forceinline__ void stage_grid_int8(const float *grid, int8_t *shared_grid, int tid) {
  for (int i = tid; i < Items; i += Threads)
    shared_grid[i] = static_cast<int8_t>(grid[i]);
}

// Accumulate per-choice block minima into group_error.
//
// Preconditions:
//   * Every thread in the block must reach this call -- warp_min uses full-mask shuffles,
//     so a divergent caller gets undefined results.
//   * blockDim.x = Warps * warpSize and blockDim.x >= Choices; warp_best has at least
//     Warps * Choices elements and group_error at least Choices.
//   * group_error is accumulated into, so the caller must zero it before the first call
//     of each group.
//
// Postconditions:
//   * group_error[choice] grows by the block-wide minimum of local_best[choice].
//   * Ends with a barrier, so warp_best and group_error may be reused immediately.
template <int Choices, int Warps>
__device__ __forceinline__ void accumulate_choice_min(const float (&local_best)[Choices],
                                                      float *warp_best, float *group_error, int tid,
                                                      int lane, int warp) {
#pragma unroll
  for (int choice = 0; choice < Choices; ++choice) {
    const float value = warp_min(local_best[choice]);
    if (lane == 0)
      warp_best[warp * Choices + choice] = value;
  }
  __syncthreads();
  if (tid < Choices) {
    float value = warp_best[tid];
#pragma unroll
    for (int w = 1; w < Warps; ++w)
      value = fminf(value, warp_best[w * Choices + tid]);
    group_error[tid] += value;
  }
  __syncthreads();
}

// Block-wide minimum over a packed comparison key.
//
// Preconditions:
//   * Every thread in the block must reach this call -- the warp reduction uses full-mask
//     shuffles, so a divergent caller gets undefined results.
//   * Threads = Warps * warpSize, and warp_keys has at least Warps elements.
//
// Postconditions:
//   * Only thread 0 receives the block minimum; other threads retain their warp-local value.
//   * No trailing barrier: the caller must consume the result in thread 0 and reach a
//     __syncthreads() before any thread overwrites warp_keys.
template <int Warps>
__device__ __forceinline__ unsigned long long
block_min_key(unsigned long long key, unsigned long long *warp_keys, int tid, int lane, int warp) {
  key = warp_min_key(key);
  if (lane == 0)
    warp_keys[warp] = key;
  __syncthreads();
  if (tid == 0) {
    key = warp_keys[0];
#pragma unroll
    for (int w = 1; w < Warps; ++w)
      key = warp_keys[w] < key ? warp_keys[w] : key;
  }
  return key;
}

} // namespace ggml_pack
} // namespace modelopt
