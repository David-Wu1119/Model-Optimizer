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

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cuda_fp16.h>

#include <cfloat>
#include <cstdint>
#include <limits>

#include "ggml_pack_common.cuh"

namespace {

constexpr int kBlockSize = 256;
constexpr int kVectorSize = 8;
constexpr int kEntries = 512;
constexpr int kEntryMask = kEntries - 1;
constexpr int kGroups = 16;
constexpr int kVectorsPerGroup = kBlockSize / (kGroups * kVectorSize);
constexpr int kGroupValues = kVectorsPerGroup * kVectorSize;
constexpr int kLocalScales = 16;
constexpr int kPayloadBytes = 74;
constexpr int kCodeOffset = 2;
constexpr int kCodeBytesPerGroup = 2 * kVectorsPerGroup;
constexpr int kLocalScaleOffset = kCodeOffset + kCodeBytesPerGroup * kGroups;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarps = kThreads / kWarpSize;
// Largest grid value times the largest local multiplier, divided by 8.
constexpr float kNativeMax = 43.0f * 31.0f / 8.0f;
// Peak-clipping heuristic mirrored from the reference encoder: the anchor backs off from 1.0 as
// the block's peak-to-RMS ratio grows, bounded to this interval.
constexpr float kPeakToRmsSlope = 0.035f;
constexpr float kMinAnchor = 0.65f;
constexpr float kMaxAnchor = 0.92f;
static_assert(kThreads % kWarpSize == 0);
static_assert(kWarpSize == 32, "the shuffle reductions below start at delta = 16");
static_assert(kThreads >= kBlockSize, "shared_input is filled one value per thread");
static_assert(kThreads >= kPayloadBytes, "the zero-block path writes one byte per thread");
static_assert(kThreads >= kLocalScales, "local-scale reductions assign one thread per scale");
static_assert(kThreads >= kGroups / 2, "local-scale bytes are written one per thread");
static_assert(kEntries == 512, "the packed code stores a 9-bit grid index");
static_assert(kLocalScales <= 16, "each local scale must fit in one nibble");
static_assert(kGroups * kGroupValues == kBlockSize, "group tiling must cover the block");
static_assert(kPayloadBytes == kLocalScaleOffset + kGroups / 2,
              "payload layout must match scale, code, and local-scale fields");

__device__ __forceinline__ float quant_error(float xnorm, float dot, float qnorm, float scale) {
  return fmaxf(fmaf(scale * scale, qnorm, fmaf(-2.0f * scale, dot, xnorm)), 0.0f);
}

// The packed word stores seven of eight sign bits; the decoder rebuilds the eighth from parity,
// so signs must have even popcount. For odd input parity, flip the smallest-magnitude |x|*q term,
// which subtracts it twice from the dot. The payload-writing search below uses the same tie-break.
struct ParityDot {
  float dot;
  int flip_index;
};

__device__ __forceinline__ ParityDot even_parity_dot(const float *x, const int8_t *q,
                                                     bool odd_parity) {
  float dot = 0.0f;
  float weakest = FLT_MAX;
  int flip_index = 0;
#pragma unroll
  for (int j = 0; j < kVectorSize; ++j) {
    const float term = fabsf(x[j]) * static_cast<float>(q[j]);
    dot += term;
    if (term < weakest) {
      weakest = term;
      flip_index = j;
    }
  }
  return {odd_parity ? dot - 2.0f * weakest : dot, flip_index};
}

template <typename scalar_t>
__global__ void encode(const scalar_t *input, int64_t num_blocks, const float *grid,
                       uint8_t *output) {
  // Canonical IQ2_XS magnitudes are at most 43 and therefore fit in int8_t.
  __shared__ int8_t shared_grid[kEntries * kVectorSize];
  __shared__ float grid_norm[kEntries];
  __shared__ float shared_input[kBlockSize];
  __shared__ float vector_norm[kBlockSize / kVectorSize];
  __shared__ uint8_t vector_odd_parity[kBlockSize / kVectorSize];
  __shared__ float warp_best[kWarps * kLocalScales];
  __shared__ float group_error[kLocalScales];
  __shared__ unsigned long long warp_keys[kWarps];
  __shared__ float warp_amax[kWarps];
  __shared__ float warp_sumsq[kWarps];
  __shared__ uint16_t shared_d_bits;
  __shared__ int selected_local;
  __shared__ uint8_t locals[kGroups];

  const int tid = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  const int64_t block = blockIdx.x;
  if (block >= num_blocks)
    return;
  const scalar_t *source = input + block * kBlockSize;
  uint8_t *payload = output + block * kPayloadBytes;
  const float value = tid < kBlockSize ? modelopt::ggml_pack::load_float(source + tid) : 0.0f;
  if (tid < kBlockSize)
    shared_input[tid] = value;
  float amax = modelopt::ggml_pack::warp_max(fabsf(value));
  float sumsq = modelopt::ggml_pack::warp_sum(value * value);
  if (lane == 0) {
    warp_amax[warp] = amax;
    warp_sumsq[warp] = sumsq;
  }
  __syncthreads();
  if (tid == 0) {
    amax = warp_amax[0];
    sumsq = warp_sumsq[0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) {
      amax = fmaxf(amax, warp_amax[w]);
      sumsq += warp_sumsq[w];
    }
    const float rms = sqrtf(sumsq / kBlockSize);
    const float peak_to_rms = rms > 0.0f ? amax / rms : 0.0f;
    const float anchor = fminf(kMaxAnchor, fmaxf(kMinAnchor, 1.0f - kPeakToRmsSlope * peak_to_rms));
    const __half scale = __float2half_rn(fminf((amax / kNativeMax) * anchor, 65504.0f));
    shared_d_bits = __half_as_ushort(scale);
    payload[0] = static_cast<uint8_t>(shared_d_bits);
    payload[1] = static_cast<uint8_t>(shared_d_bits >> 8);
  }
  __syncthreads();
  const uint16_t d_bits = shared_d_bits;
  if (d_bits == 0) {
    if (tid < kPayloadBytes)
      payload[tid] = 0;
    return;
  }
  const float d = __half2float(__ushort_as_half(d_bits));

  modelopt::ggml_pack::stage_grid_int8<kEntries * kVectorSize, kThreads>(grid, shared_grid, tid);
  __syncthreads();
  for (int entry = tid; entry < kEntries; entry += kThreads) {
    float norm = 0.0f;
#pragma unroll
    for (int j = 0; j < kVectorSize; ++j) {
      const float q = static_cast<float>(shared_grid[entry * kVectorSize + j]);
      norm = fmaf(q, q, norm);
    }
    grid_norm[entry] = norm;
  }
  __syncthreads();
  if (tid < kBlockSize / kVectorSize) {
    float norm = 0.0f;
    int negative_count = 0;
#pragma unroll
    for (int j = 0; j < kVectorSize; ++j) {
      const float x = shared_input[tid * kVectorSize + j];
      norm = fmaf(x, x, norm);
      negative_count += x < 0.0f;
    }
    vector_norm[tid] = norm;
    vector_odd_parity[tid] = static_cast<uint8_t>(negative_count & 1);
  }
  __syncthreads();

#pragma unroll 1
  for (int group = 0; group < kGroups; ++group) {
    if (tid < kLocalScales)
      group_error[tid] = 0.0f;
    __syncthreads();

#pragma unroll
    for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
      const int offset = group * kGroupValues + vector * kVectorSize;
      const int vector_index = offset / kVectorSize;
      const float *x = shared_input + offset;
      const float xnorm = vector_norm[vector_index];
      const bool odd_parity = vector_odd_parity[vector_index] != 0;
      float local_best[kLocalScales];
#pragma unroll
      for (int local = 0; local < kLocalScales; ++local)
        local_best[local] = FLT_MAX;
      for (int entry = tid; entry < kEntries; entry += kThreads) {
        const int8_t *q = shared_grid + entry * kVectorSize;
        const float dot = even_parity_dot(x, q, odd_parity).dot;
#pragma unroll
        for (int local = 0; local < kLocalScales; ++local) {
          const float scale = d * (2 * local + 1) * 0.125f;
          local_best[local] =
              fminf(local_best[local], quant_error(xnorm, dot, grid_norm[entry], scale));
        }
      }
      modelopt::ggml_pack::accumulate_choice_min<kLocalScales, kWarps>(
          local_best, warp_best, group_error, tid, lane, warp);
    }

    if (tid == 0) {
      selected_local = 0;
      float best = group_error[0];
#pragma unroll
      for (int local = 1; local < kLocalScales; ++local) {
        if (group_error[local] < best) {
          best = group_error[local];
          selected_local = local;
        }
      }
      locals[group] = static_cast<uint8_t>(selected_local);
    }
    __syncthreads();
    const float selected_scale = d * (2 * selected_local + 1) * 0.125f;

#pragma unroll
    for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
      const int offset = group * kGroupValues + vector * kVectorSize;
      const int vector_index = offset / kVectorSize;
      const float *x = shared_input + offset;
      const float xnorm = vector_norm[vector_index];
      const bool odd_parity = vector_odd_parity[vector_index] != 0;
      unsigned long long key = ~0ULL;
      for (int entry = tid; entry < kEntries; entry += kThreads) {
        const auto parity = even_parity_dot(x, shared_grid + entry * kVectorSize, odd_parity);
        const float error = quant_error(xnorm, parity.dot, grid_norm[entry], selected_scale);
        const unsigned long long candidate =
            (static_cast<unsigned long long>(__float_as_uint(error)) << 32) |
            static_cast<unsigned long long>(entry);
        key = candidate < key ? candidate : key;
      }
      key = modelopt::ggml_pack::block_min_key<kWarps>(key, warp_keys, tid, lane, warp);
      if (tid == 0) {
        const int entry = static_cast<int>(key & kEntryMask);
        const int8_t *q = shared_grid + entry * kVectorSize;
        const int flip_index = even_parity_dot(x, q, odd_parity).flip_index;
        int sign_mask = 0;
#pragma unroll
        for (int j = 0; j < kVectorSize; ++j) {
          bool is_negative = x[j] < 0.0f;
          if (odd_parity && j == flip_index)
            is_negative = !is_negative;
          sign_mask |= static_cast<int>(is_negative) << j;
        }
        const uint16_t code = static_cast<uint16_t>(entry | ((sign_mask & 0x7f) << 9));
        const int code_offset = kCodeOffset + 2 * (group * kVectorsPerGroup + vector);
        payload[code_offset] = static_cast<uint8_t>(code);
        payload[code_offset + 1] = static_cast<uint8_t>(code >> 8);
      }
      __syncthreads();
    }
  }

  if (tid < kGroups / 2)
    payload[kLocalScaleOffset + tid] = locals[2 * tid] | (locals[2 * tid + 1] << 4);
}

} // namespace

at::Tensor iq2_xs_pack_cuda(at::Tensor input, at::Tensor grid, bool validate_grid) {
  TORCH_CHECK(input.is_contiguous() && grid.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(input.numel() > 0 && input.numel() % kBlockSize == 0,
              "input size must be a positive multiple of 256");
  TORCH_CHECK(grid.scalar_type() == at::kFloat && grid.dim() == 2 && grid.size(0) == kEntries &&
                  grid.size(1) == kVectorSize,
              "grid must be float32 [512, 8]");
  TORCH_CHECK(input.get_device() == grid.get_device(), "input and grid must share a device");
  c10::cuda::CUDAGuard guard(input.device());
  if (validate_grid) {
    // The upper bound is int8 staging; the lower bound is semantic. IQ2_XS grid entries are
    // magnitudes and signs live in the code word, so even_parity_dot assumes nonnegative q.
    const auto valid =
        grid.eq(grid.trunc()).logical_and(grid.ge(0.0f)).logical_and(grid.le(127.0f));
    TORCH_CHECK(valid.all().item<bool>(),
                "grid values must be integral and within [0, 127]: IQ2_XS grid entries are "
                "non-negative magnitudes staged as int8");
  }
  const int64_t num_blocks = input.numel() / kBlockSize;
  TORCH_CHECK(num_blocks <= std::numeric_limits<int>::max(), "IQ2_XS CUDA grid is too large");
  auto output = at::empty({num_blocks, kPayloadBytes}, input.options().dtype(at::kByte));
  const auto stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "iq2_xs_pack", [&] {
        encode<scalar_t><<<static_cast<int>(num_blocks), kThreads, 0, stream>>>(
            input.data_ptr<scalar_t>(), num_blocks, grid.data_ptr<float>(),
            output.data_ptr<uint8_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}
