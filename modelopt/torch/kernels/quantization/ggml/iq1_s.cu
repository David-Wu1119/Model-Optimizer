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

namespace {

constexpr int kBlockSize = 256;
constexpr int kVectorSize = 8;
constexpr int kEntries = 2048;
constexpr int kGroups = 8;
constexpr int kVectorsPerGroup = kBlockSize / (kGroups * kVectorSize);
constexpr int kGroupValues = kVectorsPerGroup * kVectorSize;
constexpr int kLocalScales = 8;
constexpr int kChoices = 2 * kLocalScales;
constexpr int kPayloadBytes = 50;
constexpr int kIndexOffset = 2;
constexpr int kMetadataOffset = kIndexOffset + kVectorsPerGroup * kGroups;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarps = kThreads / kWarpSize;
constexpr float kDelta = 0.125f;
// Largest shifted grid magnitude times the largest local multiplier: 1.125 * 15.
constexpr float kNativeMax = 16.875f;
// Mirrors the reference encoder's peak-clipping anchor.
constexpr float kScaleAnchor = 0.61f;
static_assert(kThreads % kWarpSize == 0);
static_assert(kWarpSize == 32, "the shuffle reductions below start at delta = 16");
static_assert(kThreads >= kBlockSize, "shared_input is filled one value per thread");
static_assert(kThreads >= kPayloadBytes, "the zero-block path writes one byte per thread");
static_assert(kThreads >= kChoices, "choice reductions assign one thread per choice");
static_assert(kGroups * kGroupValues == kBlockSize, "group tiling must cover the block");
static_assert(kPayloadBytes == kMetadataOffset + 2 * kGroups,
              "payload layout must match scale, index, and metadata fields");

template <typename scalar_t> __device__ __forceinline__ float load_float(const scalar_t *input) {
  return static_cast<float>(*input);
}

__device__ __forceinline__ float quant_dot(const float *x, const int8_t *q) {
  float dot = 0.0f;
#pragma unroll
  for (int j = 0; j < kVectorSize; ++j)
    dot = fmaf(x[j], static_cast<float>(q[j]), dot);
  return dot;
}

__device__ __forceinline__ float shifted_error(float xnorm, float dot, float xsum, float qnorm,
                                               float qsum, float scale, float delta) {
  const float shifted_dot = dot + delta * xsum;
  const float shifted_norm =
      qnorm + 2.0f * delta * qsum + static_cast<float>(kVectorSize) * delta * delta;
  return fmaxf(fmaf(scale * scale, shifted_norm, fmaf(-2.0f * scale, shifted_dot, xnorm)), 0.0f);
}

__device__ __forceinline__ float quant_error(float xnorm, float xsum, const float *x,
                                             const int8_t *q, float qnorm, float qsum, float scale,
                                             float delta) {
  return shifted_error(xnorm, quant_dot(x, q), xsum, qnorm, qsum, scale, delta);
}

template <typename scalar_t>
__global__ void find_scale(const scalar_t *input, int64_t num_blocks, int16_t *scale_bits) {
  const int64_t block = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (block >= num_blocks)
    return;

  float amax = 0.0f;
  const scalar_t *values = input + block * kBlockSize;
#pragma unroll 1
  for (int i = 0; i < kBlockSize; ++i)
    amax = fmaxf(amax, fabsf(load_float(values + i)));
  const __half scale = __float2half_rn(fminf((amax / kNativeMax) * kScaleAnchor, 65504.0f));
  scale_bits[block] = static_cast<int16_t>(__half_as_ushort(scale));
}

template <typename scalar_t>
__global__ void encode(const scalar_t *input, int64_t num_blocks, const float *grid,
                       const int16_t *scale_bits, uint8_t *output) {
  __shared__ int8_t shared_grid[kEntries * kVectorSize];
  __shared__ uint8_t grid_norm[kEntries];
  __shared__ int8_t grid_sum[kEntries];
  __shared__ float shared_input[kBlockSize];
  __shared__ float vector_norm[kBlockSize / kVectorSize];
  __shared__ float vector_sum[kBlockSize / kVectorSize];
  __shared__ float warp_best[kWarps * kChoices];
  __shared__ float group_error[kChoices];
  __shared__ unsigned long long warp_keys[kWarps];
  __shared__ int selected_choice;
  __shared__ uint16_t selected_entries[kVectorsPerGroup];

  const int tid = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  const int64_t block = blockIdx.x;
  if (block >= num_blocks)
    return;

  const scalar_t *source = input + block * kBlockSize;
  uint8_t *payload = output + block * kPayloadBytes;
  const uint16_t d_bits = static_cast<uint16_t>(scale_bits[block]);
  const float d = __half2float(__ushort_as_half(d_bits));
  if (d_bits == 0) {
    if (tid < kPayloadBytes)
      payload[tid] = 0;
    return;
  }
  if (tid == 0) {
    payload[0] = static_cast<uint8_t>(d_bits);
    payload[1] = static_cast<uint8_t>(d_bits >> 8);
  }

  for (int i = tid; i < kEntries * kVectorSize; i += kThreads)
    shared_grid[i] = static_cast<int8_t>(grid[i]);
  __syncthreads();
  for (int entry = tid; entry < kEntries; entry += kThreads) {
    int norm = 0;
    int sum = 0;
#pragma unroll
    for (int j = 0; j < kVectorSize; ++j) {
      const int q = static_cast<int>(shared_grid[entry * kVectorSize + j]);
      norm += q * q;
      sum += q;
    }
    grid_norm[entry] = static_cast<uint8_t>(norm);
    grid_sum[entry] = static_cast<int8_t>(sum);
  }
  if (tid < kBlockSize)
    shared_input[tid] = load_float(source + tid);
  __syncthreads();
  if (tid < kBlockSize / kVectorSize) {
    float norm = 0.0f;
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < kVectorSize; ++j) {
      const float x = shared_input[tid * kVectorSize + j];
      norm = fmaf(x, x, norm);
      sum += x;
    }
    vector_norm[tid] = norm;
    vector_sum[tid] = sum;
  }
  __syncthreads();

#pragma unroll 1
  for (int group = 0; group < kGroups; ++group) {
    if (tid < kChoices)
      group_error[tid] = 0.0f;
    __syncthreads();

#pragma unroll
    for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
      const int offset = group * kGroupValues + vector * kVectorSize;
      const int vector_index = offset / kVectorSize;
      const float *x = shared_input + offset;
      const float xnorm = vector_norm[vector_index];
      const float xsum = vector_sum[vector_index];
      float local_best[kChoices];
#pragma unroll
      for (int choice = 0; choice < kChoices; ++choice)
        local_best[choice] = FLT_MAX;
      for (int entry = tid; entry < kEntries; entry += kThreads) {
        const int8_t *q = shared_grid + entry * kVectorSize;
        const float dot = quant_dot(x, q);
        const float qnorm = static_cast<float>(grid_norm[entry]);
        const float qsum = static_cast<float>(grid_sum[entry]);
#pragma unroll
        for (int choice = 0; choice < kChoices; ++choice) {
          const int local = choice & (kLocalScales - 1);
          const float delta = choice < kLocalScales ? kDelta : -kDelta;
          const float scale = d * (2 * local + 1);
          const float error = shifted_error(xnorm, dot, xsum, qnorm, qsum, scale, delta);
          local_best[choice] = fminf(local_best[choice], error);
        }
      }
#pragma unroll
      for (int choice = 0; choice < kChoices; ++choice) {
        float value = local_best[choice];
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1)
          value = fminf(value, __shfl_down_sync(0xffffffff, value, delta));
        if (lane == 0)
          warp_best[warp * kChoices + choice] = value;
      }
      __syncthreads();
      if (tid < kChoices) {
        float value = warp_best[tid];
#pragma unroll
        for (int w = 1; w < kWarps; ++w)
          value = fminf(value, warp_best[w * kChoices + tid]);
        group_error[tid] += value;
      }
      __syncthreads();
    }

    if (tid == 0) {
      selected_choice = 0;
      float best = group_error[0];
#pragma unroll
      for (int choice = 1; choice < kChoices; ++choice) {
        if (group_error[choice] < best) {
          best = group_error[choice];
          selected_choice = choice;
        }
      }
    }
    __syncthreads();
    const int selected_local = selected_choice & (kLocalScales - 1);
    const float selected_delta = selected_choice < kLocalScales ? kDelta : -kDelta;
    const float selected_scale = d * (2 * selected_local + 1);

#pragma unroll
    for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
      const int offset = group * kGroupValues + vector * kVectorSize;
      const int vector_index = offset / kVectorSize;
      const float *x = shared_input + offset;
      const float xnorm = vector_norm[vector_index];
      const float xsum = vector_sum[vector_index];
      unsigned long long key = ~0ULL;
      for (int entry = tid; entry < kEntries; entry += kThreads) {
        const float error = quant_error(
            xnorm, xsum, x, shared_grid + entry * kVectorSize, static_cast<float>(grid_norm[entry]),
            static_cast<float>(grid_sum[entry]), selected_scale, selected_delta);
        const unsigned long long candidate =
            (static_cast<unsigned long long>(__float_as_uint(error)) << 32) |
            static_cast<unsigned long long>(entry);
        key = candidate < key ? candidate : key;
      }
#pragma unroll
      for (int delta = 16; delta > 0; delta >>= 1) {
        const auto other = __shfl_down_sync(0xffffffff, key, delta);
        key = other < key ? other : key;
      }
      if (lane == 0)
        warp_keys[warp] = key;
      __syncthreads();
      if (tid == 0) {
        key = warp_keys[0];
#pragma unroll
        for (int w = 1; w < kWarps; ++w)
          key = warp_keys[w] < key ? warp_keys[w] : key;
        const uint16_t entry = static_cast<uint16_t>(key & 0x7ff);
        selected_entries[vector] = entry;
        payload[kIndexOffset + group * kVectorsPerGroup + vector] = static_cast<uint8_t>(entry);
      }
      __syncthreads();
    }

    if (tid == 0) {
      uint16_t qh = 0;
#pragma unroll
      for (int vector = 0; vector < kVectorsPerGroup; ++vector)
        qh |= static_cast<uint16_t>(((selected_entries[vector] >> 8) & 7) << (3 * vector));
      qh |= static_cast<uint16_t>((selected_local << 12) | ((selected_choice >> 3) << 15));
      payload[kMetadataOffset + 2 * group] = static_cast<uint8_t>(qh);
      payload[kMetadataOffset + 2 * group + 1] = static_cast<uint8_t>(qh >> 8);
    }
    __syncthreads();
  }
}

} // namespace

at::Tensor iq1_s_pack_cuda(at::Tensor input, at::Tensor grid) {
  TORCH_CHECK(input.is_contiguous() && grid.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(input.numel() > 0 && input.numel() % kBlockSize == 0,
              "input size must be a positive multiple of 256");
  TORCH_CHECK(grid.scalar_type() == at::kFloat && grid.numel() == kEntries * kVectorSize,
              "grid must be float32 [2048, 8]");
  TORCH_CHECK(input.get_device() == grid.get_device(), "input and grid must share a device");
  c10::cuda::CUDAGuard guard(input.device());
  const int64_t num_blocks = input.numel() / kBlockSize;
  TORCH_CHECK(num_blocks <= std::numeric_limits<int>::max(), "IQ1_S CUDA grid is too large");
  auto scales = at::empty({num_blocks}, input.options().dtype(at::kShort));
  auto output = at::empty({num_blocks, kPayloadBytes}, input.options().dtype(at::kByte));
  const auto stream = c10::cuda::getCurrentCUDAStream();
  const int scale_grid = static_cast<int>((num_blocks + kThreads - 1) / kThreads);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "iq1_s_pack", [&] {
        find_scale<scalar_t><<<scale_grid, kThreads, 0, stream>>>(
            input.data_ptr<scalar_t>(), num_blocks, scales.data_ptr<int16_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        encode<scalar_t><<<static_cast<int>(num_blocks), kThreads, 0, stream>>>(
            input.data_ptr<scalar_t>(), num_blocks, grid.data_ptr<float>(),
            scales.data_ptr<int16_t>(), output.data_ptr<uint8_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}
