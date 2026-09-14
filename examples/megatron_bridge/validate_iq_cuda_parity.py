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

"""Require byte-identical CPU and CUDA GGML IQ packing on deterministic inputs."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from modelopt.torch.quantization.extensions import get_cuda_ext_iq1_s, get_cuda_ext_iq2_xs
from modelopt.torch.quantization.ggml.iq1_s import quantize_iq1_s
from modelopt.torch.quantization.ggml.iq2_xs import quantize_iq2_xs


def _compare_format(name: str, weight: torch.Tensor, quantize, get_extension) -> dict:
    extension = get_extension()
    if extension is None:
        raise RuntimeError(f"The CUDA extension for {name} is unavailable")

    packed_cpu, shape_cpu = quantize(weight)
    packed_cuda, shape_cuda = quantize(weight.cuda())
    packed_cuda_cpu = packed_cuda.cpu()
    differing_bytes = int(torch.count_nonzero(packed_cpu != packed_cuda_cpu).item())
    result = {
        "format": name,
        "logical_shape": list(weight.shape),
        "padded_shape": [*weight.shape[:-1], packed_cpu.shape[-2] * 256],
        "packed_shape": list(packed_cpu.shape),
        "payload_bytes": packed_cpu.numel(),
        "differing_payload_bytes": differing_bytes,
        "logical_shape_matches": torch.equal(shape_cpu, shape_cuda.cpu()),
        "cuda_extension_loaded": True,
    }
    if differing_bytes or not result["logical_shape_matches"]:
        raise AssertionError(f"CPU/CUDA parity failed for {name}: {result}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--row-width", type=int, default=257)
    parser.add_argument("--seed", type=int, default=5918)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.rows <= 0:
        raise ValueError("--rows must be positive")
    if args.row_width <= 0:
        raise ValueError("--row-width must be positive")

    generator = torch.Generator().manual_seed(args.seed)
    weight = torch.randn((args.rows, args.row_width), generator=generator, dtype=torch.bfloat16)
    results = [
        _compare_format("IQ1_S", weight, quantize_iq1_s, get_cuda_ext_iq1_s),
        _compare_format("IQ2_XS", weight, quantize_iq2_xs, get_cuda_ext_iq2_xs),
    ]
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed",
        "seed": args.seed,
        "rows": args.rows,
        "row_width": args.row_width,
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
