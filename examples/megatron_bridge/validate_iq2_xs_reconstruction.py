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

"""Compare packed IQ2_XS rows with their source checkpoint tensors.

This validator closes a gap left by structural and decoder-oracle checks: a byte-valid block can
still belong to the wrong source row. It samples logical rows from a packed unified Hugging Face
checkpoint, reconstructs them, and compares them with the same rows in the floating-point source
checkpoint. Optionally, it also requantizes every sampled source row and requires byte-identical
payloads. The report separates complete 256-value blocks from the final partial block so row-padding
regressions are visible.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from validate_mixed_quantized_hf import TensorRecord, _read_int64_vector, read_checkpoint_index

from modelopt.torch.quantization.ggml import dequantize_iq2_xs, quantize_iq2_xs

_BLOCK_SIZE = 256
_PAYLOAD_BYTES = 74


@dataclass(frozen=True)
class ErrorMetrics:
    """Numerical reconstruction metrics for one vector region."""

    values: int
    sum_squared_error: float
    normalized_squared_error: float | None
    cosine_similarity: float | None
    max_absolute_error: float


def _metrics(reference: torch.Tensor, reconstructed: torch.Tensor) -> ErrorMetrics | None:
    if reference.numel() == 0:
        return None
    reference = reference.float()
    reconstructed = reconstructed.float()
    error = reconstructed - reference
    squared_error = error.square().sum().item()
    signal_energy = reference.square().sum().item()
    reference_norm = reference.norm().item()
    reconstructed_norm = reconstructed.norm().item()
    return ErrorMetrics(
        values=reference.numel(),
        sum_squared_error=squared_error,
        normalized_squared_error=squared_error / signal_energy if signal_energy else None,
        cosine_similarity=(
            torch.dot(reference, reconstructed).item() / (reference_norm * reconstructed_norm)
            if reference_norm and reconstructed_norm
            else None
        ),
        max_absolute_error=error.abs().max().item(),
    )


def _row_coordinates(shape: tuple[int, ...], flat_row: int) -> tuple[int, ...]:
    coordinates = []
    for dimension in reversed(shape[:-1]):
        coordinates.append(flat_row % dimension)
        flat_row //= dimension
    return tuple(reversed(coordinates))


def _sample_indices(count: int, samples: int) -> list[int]:
    if count <= 0 or samples <= 0:
        return []
    if samples >= count:
        return list(range(count))
    if samples == 1:
        return [0]
    return sorted({round(index * (count - 1) / (samples - 1)) for index in range(samples)})


def _select_tensors(names: list[str], patterns: tuple[str, ...], maximum: int | None) -> list[str]:
    selected = [
        name
        for name in sorted(names)
        if not patterns or any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
    ]
    if maximum is None or maximum >= len(selected):
        return selected
    return [selected[index] for index in _sample_indices(len(selected), maximum)]


class _CheckpointReader:
    """Keep only the required safetensors shards open while reading row slices."""

    def __init__(self, root: Path, records: dict[str, TensorRecord], stack: ExitStack):
        self.root = root
        self.records = records
        self.stack = stack
        self.handles: dict[str, Any] = {}

    def row(self, name: str, flat_row: int, *, trailing_dimensions: int = 1) -> torch.Tensor:
        record = self.records[name]
        if record.shard not in self.handles:
            self.handles[record.shard] = self.stack.enter_context(
                safe_open(str(self.root / record.shard), framework="pt", device="cpu")
            )
        logical_row_shape = (*record.shape[:-trailing_dimensions], 1)
        coordinates = _row_coordinates(logical_row_shape, flat_row)
        trailing_slices = (slice(None),) * trailing_dimensions
        return self.handles[record.shard].get_slice(name)[(*coordinates, *trailing_slices)]


def _iq2_tensor_names(
    checkpoint: Path, records: dict[str, TensorRecord]
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    shapes = {}
    for name, record in records.items():
        if record.dtype != "U8" or not name.endswith(".weight"):
            continue
        if len(record.shape) < 2 or record.shape[-1] != _PAYLOAD_BYTES:
            continue
        base = name.removesuffix(".weight")
        logical_record = records.get(base + ".weight_logical_shape")
        padded_record = records.get(base + ".weight_padded_shape")
        if logical_record is None or padded_record is None:
            continue
        logical_shape = _read_int64_vector(checkpoint, logical_record)
        padded_shape = _read_int64_vector(checkpoint, padded_record)
        expected_padded_width = math.ceil(logical_shape[-1] / _BLOCK_SIZE) * _BLOCK_SIZE
        expected_packed_shape = (*logical_shape[:-1], expected_padded_width // _BLOCK_SIZE, 74)
        if padded_shape != (*logical_shape[:-1], expected_padded_width):
            raise ValueError(f"{name} has invalid padded shape {padded_shape}")
        if record.shape != expected_packed_shape:
            raise ValueError(f"{name} has invalid packed shape {record.shape}")
        shapes[name] = (logical_shape, padded_shape)
    if not shapes:
        raise ValueError(f"No packed IQ2_XS tensors found in {checkpoint}")
    return shapes


def _aggregate_metrics(samples: list[dict], region: str) -> dict:
    metrics = [sample[region] for sample in samples if sample[region] is not None]
    values = sum(metric["values"] for metric in metrics)
    squared_error = sum(metric["sum_squared_error"] for metric in metrics)
    signal_energy = sum(metric["signal_energy"] for metric in metrics)
    reconstructed_energy = sum(metric["reconstructed_energy"] for metric in metrics)
    dot_product = sum(metric["dot_product"] for metric in metrics)
    return {
        "values": values,
        "sum_squared_error": squared_error,
        "normalized_squared_error": squared_error / signal_energy if signal_energy else None,
        "cosine_similarity": (
            dot_product / math.sqrt(signal_energy * reconstructed_energy)
            if signal_energy and reconstructed_energy
            else None
        ),
        "max_absolute_error": max(
            (metric["max_absolute_error"] for metric in metrics), default=0.0
        ),
    }


def _metric_report(reference: torch.Tensor, reconstructed: torch.Tensor) -> dict | None:
    metrics = _metrics(reference, reconstructed)
    if metrics is None:
        return None
    return {
        **asdict(metrics),
        "signal_energy": reference.float().square().sum().item(),
        "reconstructed_energy": reconstructed.float().square().sum().item(),
        "dot_product": torch.dot(reference.float(), reconstructed.float()).item(),
    }


@torch.no_grad()
def validate_reconstruction(
    checkpoint: Path,
    reference_checkpoint: Path,
    *,
    rows_per_tensor: int = 1,
    maximum_tensors: int | None = 32,
    tensor_patterns: tuple[str, ...] = (),
    require_repack_match: bool = False,
    device: str = "cpu",
    maximum_normalized_error: float | None = None,
) -> dict:
    """Sample packed rows, compare reconstruction, and return a JSON-serializable report."""

    checkpoint = checkpoint.resolve()
    reference_checkpoint = reference_checkpoint.resolve()
    records = read_checkpoint_index(checkpoint)
    reference_records = read_checkpoint_index(reference_checkpoint)
    shapes = _iq2_tensor_names(checkpoint, records)
    tensor_names = _select_tensors(list(shapes), tensor_patterns, maximum_tensors)
    if not tensor_names:
        raise ValueError(f"No IQ2_XS tensor names match {list(tensor_patterns)}")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation was requested, but CUDA is not available")

    errors: list[str] = []
    samples: list[dict] = []
    with ExitStack() as stack:
        packed_reader = _CheckpointReader(checkpoint, records, stack)
        reference_reader = _CheckpointReader(reference_checkpoint, reference_records, stack)
        for name in tensor_names:
            logical_shape, padded_shape = shapes[name]
            reference_record = reference_records.get(name)
            if reference_record is None:
                errors.append(f"{name} is absent from the reference checkpoint")
                continue
            if reference_record.shape != logical_shape:
                errors.append(
                    f"{name} source shape {reference_record.shape} does not match {logical_shape}"
                )
                continue

            row_count = math.prod(logical_shape[:-1])
            for flat_row in _sample_indices(row_count, rows_per_tensor):
                coordinates = _row_coordinates(logical_shape, flat_row)
                packed = packed_reader.row(name, flat_row, trailing_dimensions=2).to(
                    device=resolved_device
                )
                reference = reference_reader.row(name, flat_row).to(
                    device=resolved_device, dtype=torch.float32
                )
                row_shape = torch.tensor([logical_shape[-1]], device=resolved_device)
                reconstructed = dequantize_iq2_xs(packed, row_shape, dtype=torch.float32)
                complete_values = logical_shape[-1] // _BLOCK_SIZE * _BLOCK_SIZE

                repack_match = None
                mismatched_bytes = None
                if require_repack_match:
                    repacked, _ = quantize_iq2_xs(reference)
                    difference = repacked != packed
                    mismatched_bytes = int(difference.sum().item())
                    repack_match = not bool(mismatched_bytes)
                    if not repack_match:
                        errors.append(
                            f"{name} row {coordinates} differs from direct packing in "
                            f"{mismatched_bytes} bytes"
                        )

                sample: dict[str, Any] = {
                    "tensor": name,
                    "row": list(coordinates),
                    "logical_width": logical_shape[-1],
                    "padded_width": padded_shape[-1],
                    "repack_match": repack_match,
                    "mismatched_bytes": mismatched_bytes,
                    "all_values": _metric_report(reference, reconstructed),
                    "complete_blocks": _metric_report(
                        reference[:complete_values], reconstructed[:complete_values]
                    ),
                    "partial_tail": _metric_report(
                        reference[complete_values:], reconstructed[complete_values:]
                    ),
                }
                samples.append(sample)
                normalized_error = sample["all_values"]["normalized_squared_error"]
                if (
                    maximum_normalized_error is not None
                    and normalized_error is not None
                    and normalized_error > maximum_normalized_error
                ):
                    errors.append(
                        f"{name} row {coordinates} normalized error {normalized_error:.6g} "
                        f"exceeds {maximum_normalized_error:.6g}"
                    )

    summary = {
        "available_iq2_xs_tensors": len(shapes),
        "sampled_tensors": len({sample["tensor"] for sample in samples}),
        "sampled_rows": len(samples),
        "repack_matches": sum(sample["repack_match"] is True for sample in samples),
        "repack_mismatches": sum(sample["repack_match"] is False for sample in samples),
        "all_values": _aggregate_metrics(samples, "all_values"),
        "complete_blocks": _aggregate_metrics(samples, "complete_blocks"),
        "partial_tails": _aggregate_metrics(samples, "partial_tail"),
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed" if not errors else "failed",
        "checkpoint": str(checkpoint),
        "reference_checkpoint": str(reference_checkpoint),
        "device": str(resolved_device),
        "settings": {
            "rows_per_tensor": rows_per_tensor,
            "maximum_tensors": maximum_tensors,
            "tensor_patterns": list(tensor_patterns),
            "require_repack_match": require_repack_match,
            "maximum_normalized_error": maximum_normalized_error,
        },
        "summary": summary,
        "samples": samples,
        "errors": errors,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rows-per-tensor", type=int, default=1)
    parser.add_argument(
        "--max-tensors",
        type=int,
        default=32,
        help="Sample tensors evenly from the sorted matching names; use 0 for all tensors.",
    )
    parser.add_argument(
        "--tensor-pattern",
        action="append",
        default=[],
        help="Optional shell-style tensor-name pattern. May be specified more than once.",
    )
    parser.add_argument("--require-repack-match", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-normalized-error", type=float)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.rows_per_tensor <= 0:
        raise ValueError("--rows-per-tensor must be positive")
    if args.max_tensors < 0:
        raise ValueError("--max-tensors cannot be negative")
    report = validate_reconstruction(
        args.checkpoint,
        args.reference_checkpoint,
        rows_per_tensor=args.rows_per_tensor,
        maximum_tensors=args.max_tensors or None,
        tensor_patterns=tuple(args.tensor_pattern),
        require_repack_match=args.require_repack_match,
        device=args.device,
        maximum_normalized_error=args.max_normalized_error,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], **report["summary"]}, indent=2))
    print(f"Validation report: {args.report}")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
