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

"""Validate GGML block alignment from safetensors metadata.

GGML block-quantized tensors require every logical row to contain an integral
number of format blocks. This preflight reads only safetensors headers and can
therefore reject an incompatible tensor policy before loading a model or
allocating a GPU.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any

_INDEX_NAME = "model.safetensors.index.json"


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        header_length_raw = file.read(8)
        if len(header_length_raw) != 8:
            raise ValueError(f"Invalid safetensors header in {path}")
        header_length = struct.unpack("<Q", header_length_raw)[0]
        header_raw = file.read(header_length)
        if len(header_raw) != header_length:
            raise ValueError(f"Truncated safetensors header in {path}")
    return json.loads(header_raw)


def _tensor_metadata(checkpoint: Path) -> dict[str, dict[str, Any]]:
    index_path = checkpoint / _INDEX_NAME
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        headers: dict[str, dict[str, Any]] = {}
        tensors = {}
        for name, shard_name in weight_map.items():
            if shard_name not in headers:
                headers[shard_name] = _read_safetensors_header(checkpoint / shard_name)
            tensors[name] = headers[shard_name][name]
        return tensors

    tensors = {}
    for shard_path in sorted(checkpoint.glob("*.safetensors")):
        header = _read_safetensors_header(shard_path)
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"Duplicate tensor {name!r} without {_INDEX_NAME}")
            tensors[name] = metadata
    if not tensors:
        raise FileNotFoundError(f"No safetensors files found in {checkpoint}")
    return tensors


def _matches(name: str, includes: list[str], excludes: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in includes) and not any(
        fnmatch.fnmatchcase(name, pattern) for pattern in excludes
    )


def validate_row_alignment(
    checkpoint: Path,
    *,
    block_size: int,
    includes: list[str],
    excludes: list[str],
) -> dict[str, Any]:
    """Return a machine-readable GGML row-alignment report."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if not includes:
        raise ValueError("At least one include pattern is required")

    selected = []
    shape_groups: Counter[tuple[tuple[int, ...], int, bool]] = Counter()
    for name, metadata in _tensor_metadata(checkpoint).items():
        if not _matches(name, includes, excludes):
            continue
        shape = tuple(int(value) for value in metadata["shape"])
        if not shape:
            raise ValueError(f"Selected tensor {name!r} has scalar shape")
        remainder = shape[-1] % block_size
        compatible = remainder == 0
        selected.append(
            {
                "name": name,
                "shape": list(shape),
                "row_width": shape[-1],
                "remainder": remainder,
                "compatible": compatible,
            }
        )
        shape_groups[(shape, remainder, compatible)] += 1

    if not selected:
        raise ValueError(f"No tensors matched include patterns: {includes}")

    incompatible = [tensor for tensor in selected if not tensor["compatible"]]
    groups = [
        {
            "shape": list(shape),
            "row_width": shape[-1],
            "remainder": remainder,
            "compatible": compatible,
            "tensor_count": count,
        }
        for (shape, remainder, compatible), count in sorted(shape_groups.items())
    ]
    return {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "block_size": block_size,
        "include_patterns": includes,
        "exclude_patterns": excludes,
        "selected_tensors": len(selected),
        "compatible_tensors": len(selected) - len(incompatible),
        "incompatible_tensors": len(incompatible),
        "shape_groups": groups,
        "incompatible": incompatible,
        "status": "passed" if not incompatible else "failed",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument(
        "--include",
        action="append",
        required=True,
        help="fnmatch pattern selecting tensor names; may be repeated",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="fnmatch pattern excluding tensor names; may be repeated",
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = validate_row_alignment(
        args.checkpoint,
        block_size=args.block_size,
        includes=args.include,
        excludes=args.exclude,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered)
    print(rendered, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
