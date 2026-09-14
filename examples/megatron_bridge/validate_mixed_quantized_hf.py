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

"""Validate a mixed-precision unified Hugging Face checkpoint without loading its tensors.

The validator reads safetensors headers and streams packed payload bytes directly from each
shard.  For the Nemotron 3.5 Lightning policy it checks that expert weights use canonical
IQ2_XS blocks, Mamba projections use NVFP4, and all remaining weights stay unquantized.
It also writes stable per-tensor and aggregate digests that can be compared with payloads
extracted from another GGML-compatible container.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

_IQ2_XS_BLOCK_SIZE = 256
_IQ2_XS_PAYLOAD_BYTES = 74
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class TensorRecord:
    """Location and metadata for one tensor payload in a safetensors shard."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: str
    payload_offset: int
    payload_bytes: int


def _read_shard_header(checkpoint: Path, shard_name: str) -> dict[str, TensorRecord]:
    shard_path = checkpoint / shard_name
    with shard_path.open("rb") as file:
        header_size_bytes = file.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"{shard_path} has a truncated safetensors header")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header = json.loads(file.read(header_size))

    payload_base = 8 + header_size
    records = {}
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        start, end = metadata["data_offsets"]
        records[name] = TensorRecord(
            name=name,
            dtype=metadata["dtype"],
            shape=tuple(metadata["shape"]),
            shard=shard_name,
            payload_offset=payload_base + start,
            payload_bytes=end - start,
        )
    return records


def read_checkpoint_index(checkpoint: Path) -> dict[str, TensorRecord]:
    """Return all tensor records in a sharded or single-file checkpoint."""

    index_path = checkpoint / "model.safetensors.index.json"
    expected_shards: dict[str, str] | None = None
    if index_path.exists():
        expected_shards = json.loads(index_path.read_text())["weight_map"]
        shard_names = sorted(set(expected_shards.values()))
    else:
        shard_names = sorted(path.name for path in checkpoint.glob("*.safetensors"))
    if not shard_names:
        raise FileNotFoundError(f"No safetensors shards found in {checkpoint}")

    records: dict[str, TensorRecord] = {}
    for shard_name in shard_names:
        for name, record in _read_shard_header(checkpoint, shard_name).items():
            # Distributed exporters may repeat shared or tied tensors in more than one shard.
            # The index is authoritative: ignore a duplicate copy unless this is the shard named
            # by weight_map. Unique unindexed tensors remain in ``records`` and are rejected by
            # the index-consistency check below.
            if (
                expected_shards is not None
                and name in expected_shards
                and expected_shards[name] != shard_name
            ):
                continue
            if name in records:
                raise ValueError(f"Tensor {name!r} appears in more than one shard")
            records[name] = record

    if expected_shards is not None:
        if set(records) != set(expected_shards):
            missing = sorted(set(expected_shards) - set(records))
            extra = sorted(set(records) - set(expected_shards))
            raise ValueError(f"Safetensors index mismatch: missing={missing}, extra={extra}")
        for name, shard_name in expected_shards.items():
            if records[name].shard != shard_name:
                raise ValueError(
                    f"Safetensors index maps {name!r} to {shard_name}, "
                    f"but its header is in {records[name].shard}"
                )
    return records


def _load_quantization_config(checkpoint: Path) -> tuple[dict, str]:
    hf_quant_config = checkpoint / "hf_quant_config.json"
    if hf_quant_config.exists():
        config = json.loads(hf_quant_config.read_text())
        return config["quantization"], hf_quant_config.name

    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    if "quantization_config" not in config:
        raise ValueError("Neither hf_quant_config.json nor config.json contains quantization data")
    return config["quantization_config"], config_path.name


def _matching_quantized_layer(name: str, quantized_layers: dict[str, dict]) -> str | None:
    matches = [
        layer
        for layer in quantized_layers
        if name == f"{layer}.weight" or (name.startswith(f"{layer}.") and name.endswith(".weight"))
    ]
    return max(matches, key=len) if matches else None


def _is_expert_weight(name: str) -> bool:
    return (
        not _is_mtp_tensor(name)
        and name.endswith(".weight")
        and (".mixer.experts." in name or ".mixer.shared_experts." in name)
    )


def _is_mamba_projection_weight(name: str) -> bool:
    return not _is_mtp_tensor(name) and name.endswith(
        (".mixer.in_proj.weight", ".mixer.out_proj.weight")
    )


def _is_mtp_tensor(name: str) -> bool:
    return name.startswith("mtp.") or ".mtp." in name


def _read_int64_vector(checkpoint: Path, record: TensorRecord) -> tuple[int, ...]:
    """Read one small int64 shape sidecar directly from its safetensors payload."""
    if record.dtype != "I64" or len(record.shape) != 1:
        raise ValueError(
            f"Shape sidecar {record.name} must be a one-dimensional I64 tensor, got "
            f"{record.dtype} {record.shape}"
        )
    with (checkpoint / record.shard).open("rb") as file:
        file.seek(record.payload_offset)
        payload = file.read(record.payload_bytes)
    if len(payload) != record.shape[0] * 8:
        raise ValueError(f"Shape sidecar {record.name} has an invalid byte count")
    return struct.unpack(f"<{record.shape[0]}q", payload)


def _digest_iq2_tensors(
    checkpoint: Path, records: dict[str, TensorRecord], names: list[str]
) -> tuple[dict[str, str], str]:
    """Hash each packed tensor and the ordered concatenation of all packed payloads."""

    per_tensor = {}
    aggregate = hashlib.sha256()
    open_shard: str | None = None
    file: BinaryIO | None = None
    try:
        for name in sorted(names):
            record = records[name]
            if record.shard != open_shard:
                if file is not None:
                    file.close()
                file = (checkpoint / record.shard).open("rb")
                open_shard = record.shard
            assert file is not None
            digest = hashlib.sha256()
            file.seek(record.payload_offset)
            remaining = record.payload_bytes
            while remaining:
                chunk = file.read(min(remaining, _HASH_CHUNK_BYTES))
                if not chunk:
                    raise EOFError(
                        f"Unexpected end of {record.shard} while hashing tensor {record.name}"
                    )
                digest.update(chunk)
                aggregate.update(chunk)
                remaining -= len(chunk)
            per_tensor[name] = digest.hexdigest()
    finally:
        if file is not None:
            file.close()
    return per_tensor, aggregate.hexdigest()


def validate_checkpoint(
    checkpoint: Path,
    reference_checkpoint: Path,
    *,
    compute_digests: bool = True,
) -> dict:
    """Validate the Nemotron mixed-format policy and return a JSON-serializable report."""

    checkpoint = checkpoint.resolve()
    reference_checkpoint = reference_checkpoint.resolve()
    records = read_checkpoint_index(checkpoint)
    reference_records = read_checkpoint_index(reference_checkpoint)
    quantization, quant_config_file = _load_quantization_config(checkpoint)
    quantized_layers = quantization.get("quantized_layers", {})
    errors: list[str] = []

    if quantization.get("quant_algo") != "MIXED_PRECISION":
        errors.append(
            f"Expected quant_algo=MIXED_PRECISION, got {quantization.get('quant_algo')!r}"
        )
    if not quantized_layers:
        errors.append("The quantization config does not declare quantized_layers")

    iq2_layers = {
        name: cfg for name, cfg in quantized_layers.items() if cfg.get("quant_algo") == "IQ2_XS"
    }
    nvfp4_layers = {
        name
        for name, cfg in quantized_layers.items()
        if cfg.get("quant_algo") in {"NVFP4", "W4A16_NVFP4"}
    }
    other_layers = {
        name: cfg.get("quant_algo")
        for name, cfg in quantized_layers.items()
        if name not in iq2_layers and name not in nvfp4_layers
    }
    if not iq2_layers:
        errors.append("No IQ2_XS layers were declared")
    if not nvfp4_layers:
        errors.append("No NVFP4 layers were declared")
    if other_layers:
        errors.append(f"Unexpected quantized formats: {other_layers}")

    for layer, cfg in iq2_layers.items():
        if not (".mixer.experts" in layer or ".mixer.shared_experts" in layer):
            errors.append(f"IQ2_XS is applied outside expert weights: {layer}")
        expected = {
            "quant_algo": "IQ2_XS",
            "group_size": _IQ2_XS_BLOCK_SIZE,
            "block_payload_bytes": _IQ2_XS_PAYLOAD_BYTES,
            "packing": "ggml",
            "row_padding": "right",
            "logical_shape_key": "weight_logical_shape",
            "padded_shape_key": "weight_padded_shape",
        }
        if cfg != expected:
            errors.append(f"Unexpected IQ2_XS metadata for {layer}: {cfg}")
    errors.extend(
        f"NVFP4 is applied outside Mamba in_proj/out_proj: {layer}"
        for layer in nvfp4_layers
        if not layer.endswith((".mixer.in_proj", ".mixer.out_proj"))
    )
    if any("mtp" in name.lower() for name in quantized_layers):
        errors.append("MTP must remain unquantized")

    iq2_tensor_names: list[str] = []
    iq2_shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    nvfp4_tensor_names: list[str] = []
    unquantized_weight_names: list[str] = []
    for name, record in records.items():
        if not name.endswith(".weight"):
            continue
        layer = _matching_quantized_layer(name, quantized_layers)
        algo = quantized_layers[layer]["quant_algo"] if layer else None
        reference = reference_records.get(name)

        if algo == "IQ2_XS":
            iq2_tensor_names.append(name)
            base = name.removesuffix(".weight")
            logical_record = records.get(base + ".weight_logical_shape")
            padded_record = records.get(base + ".weight_padded_shape")
            if record.dtype != "U8":
                errors.append(f"IQ2_XS tensor {name} has dtype {record.dtype}, expected U8")
            if len(record.shape) < 2 or record.shape[-1] != _IQ2_XS_PAYLOAD_BYTES:
                errors.append(
                    f"IQ2_XS tensor {name} has shape {record.shape}, expected [..., blocks, 74]"
                )
            if record.payload_bytes != math.prod(record.shape):
                errors.append(
                    f"IQ2_XS tensor {name} stores {record.payload_bytes} bytes for shape "
                    f"{record.shape}"
                )
            if reference is None:
                errors.append(f"IQ2_XS tensor {name} is absent from the BF16 reference")
            if logical_record is None or padded_record is None:
                errors.append(f"IQ2_XS tensor {name} is missing logical or padded shape metadata")
            else:
                try:
                    logical_shape = _read_int64_vector(checkpoint, logical_record)
                    padded_shape = _read_int64_vector(checkpoint, padded_record)
                except ValueError as error:
                    errors.append(str(error))
                else:
                    iq2_shapes[name] = (logical_shape, padded_shape)
                    expected_padded_shape = (
                        *logical_shape[:-1],
                        math.ceil(logical_shape[-1] / _IQ2_XS_BLOCK_SIZE) * _IQ2_XS_BLOCK_SIZE,
                    )
                    expected_packed_shape = (
                        *padded_shape[:-1],
                        padded_shape[-1] // _IQ2_XS_BLOCK_SIZE,
                        _IQ2_XS_PAYLOAD_BYTES,
                    )
                    if reference is not None and logical_shape != reference.shape:
                        errors.append(
                            f"IQ2_XS tensor {name} has logical shape {logical_shape}, "
                            f"but the BF16 reference shape is {reference.shape}"
                        )
                    if padded_shape != expected_padded_shape:
                        errors.append(
                            f"IQ2_XS tensor {name} has padded shape {padded_shape}, expected "
                            f"{expected_padded_shape}"
                        )
                    if record.shape != expected_packed_shape:
                        errors.append(
                            f"IQ2_XS tensor {name} has packed shape {record.shape}, expected "
                            f"{expected_packed_shape}"
                        )
        elif algo in {"NVFP4", "W4A16_NVFP4"}:
            nvfp4_tensor_names.append(name)
            if record.dtype != "U8":
                errors.append(f"NVFP4 tensor {name} has dtype {record.dtype}, expected U8")
            base = name.removesuffix(".weight")
            errors.extend(
                f"NVFP4 tensor {name} is missing {base}{suffix}"
                for suffix in (".weight_scale", ".weight_scale_2")
                if f"{base}{suffix}" not in records
            )
            if reference is None:
                errors.append(f"NVFP4 tensor {name} is absent from the BF16 reference")
            elif record.shape != (*reference.shape[:-1], reference.shape[-1] // 2):
                errors.append(
                    f"NVFP4 tensor {name} has packed shape {record.shape}, expected "
                    f"{(*reference.shape[:-1], reference.shape[-1] // 2)}"
                )
        else:
            unquantized_weight_names.append(name)
            if record.dtype == "U8":
                errors.append(f"Unaccounted packed U8 weight tensor: {name}")

    reference_experts = sorted(name for name in reference_records if _is_expert_weight(name))
    reference_mamba = sorted(
        name for name in reference_records if _is_mamba_projection_weight(name)
    )
    missing_experts = sorted(set(reference_experts) - set(iq2_tensor_names))
    unexpected_iq2 = sorted(set(iq2_tensor_names) - set(reference_experts))
    missing_mamba = sorted(set(reference_mamba) - set(nvfp4_tensor_names))
    unexpected_nvfp4 = sorted(set(nvfp4_tensor_names) - set(reference_mamba))
    if missing_experts:
        errors.append(f"Expert weights not exported as IQ2_XS: {missing_experts}")
    if unexpected_iq2:
        errors.append(f"IQ2_XS tensors are not expert weights: {unexpected_iq2}")
    if missing_mamba:
        errors.append(f"Mamba projections not exported as NVFP4: {missing_mamba}")
    if unexpected_nvfp4:
        errors.append(f"NVFP4 tensors are not Mamba projections: {unexpected_nvfp4}")

    tensor_digests: dict[str, str] = {}
    aggregate_digest: str | None = None
    if compute_digests and iq2_tensor_names:
        tensor_digests, aggregate_digest = _digest_iq2_tensors(
            checkpoint, records, iq2_tensor_names
        )

    iq2_blocks = sum(math.prod(records[name].shape[:-1]) for name in iq2_tensor_names)
    iq2_logical_weights = sum(math.prod(shapes[0]) for shapes in iq2_shapes.values())
    iq2_padded_weights = sum(math.prod(shapes[1]) for shapes in iq2_shapes.values())
    report = {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed" if not errors else "failed",
        "checkpoint": str(checkpoint),
        "reference_checkpoint": str(reference_checkpoint),
        "quantization_config_file": quant_config_file,
        "quant_algo": quantization.get("quant_algo"),
        "summary": {
            "total_tensors": len(records),
            "total_weight_tensors": sum(name.endswith(".weight") for name in records),
            "iq2_xs_layers": len(iq2_layers),
            "iq2_xs_tensors": len(iq2_tensor_names),
            "iq2_xs_blocks": iq2_blocks,
            "iq2_xs_logical_weights": iq2_logical_weights,
            "iq2_xs_padded_weights": iq2_padded_weights,
            "iq2_xs_payload_bytes": iq2_blocks * _IQ2_XS_PAYLOAD_BYTES,
            "nvfp4_layers": len(nvfp4_layers),
            "nvfp4_tensors": len(nvfp4_tensor_names),
            "unquantized_weight_tensors": len(unquantized_weight_names),
        },
        "iq2_xs": {
            "block_size": _IQ2_XS_BLOCK_SIZE,
            "block_payload_bytes": _IQ2_XS_PAYLOAD_BYTES,
            "aggregate_sha256": aggregate_digest,
            "tensors": [
                {
                    **asdict(records[name]),
                    "shape": list(records[name].shape),
                    "logical_shape": list(iq2_shapes[name][0]) if name in iq2_shapes else None,
                    "padded_shape": list(iq2_shapes[name][1]) if name in iq2_shapes else None,
                    "sha256": tensor_digests.get(name),
                }
                for name in sorted(iq2_tensor_names)
            ],
        },
        "nvfp4_tensors": sorted(nvfp4_tensor_names),
        "errors": errors,
    }
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--skip-digests",
        action="store_true",
        help="Validate layouts and policy without hashing packed IQ2_XS payloads.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = validate_checkpoint(
        args.checkpoint,
        args.reference_checkpoint,
        compute_digests=not args.skip_digests,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], **report["summary"]}, indent=2))
    print(f"Validation report: {args.report}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
