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

"""Compare exported IQ2_XS blocks with an unmodified stock GGML decoder.

The script samples complete 74-byte blocks from every IQ2_XS tensor in a unified
Hugging Face checkpoint. It sends those exact bytes to GGML's exported
``dequantize_row_iq2_xs`` function and to ModelOpt's decoder, then requires
bit-identical FP32 reconstruction. The report records the pinned llama.cpp commit,
sample locations, encoded-field digests, and mismatch counts.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from validate_mixed_quantized_hf import (
    _load_quantization_config,
    _matching_quantized_layer,
    read_checkpoint_index,
)

from modelopt.torch.quantization.ggml import dequantize_iq2_xs

_BLOCK_SIZE = 256
_BLOCK_BYTES = 74


def _llama_commit(source: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _iq2_tensor_names(checkpoint: Path) -> list[str]:
    records = read_checkpoint_index(checkpoint)
    quantization, _ = _load_quantization_config(checkpoint)
    quantized_layers = quantization.get("quantized_layers", {})
    names = []
    for name, record in records.items():
        layer = _matching_quantized_layer(name, quantized_layers)
        if layer is None or quantized_layers[layer].get("quant_algo") != "IQ2_XS":
            continue
        if record.dtype != "U8" or record.payload_bytes % _BLOCK_BYTES:
            raise ValueError(f"Invalid IQ2_XS payload for {name}: {record}")
        names.append(name)
    if not names:
        raise ValueError(f"No IQ2_XS tensors found in {checkpoint}")
    return sorted(names)


def _sample_indices(num_blocks: int, blocks_per_tensor: int) -> list[int]:
    if num_blocks <= 0:
        raise ValueError("An IQ2_XS tensor must contain at least one block")
    if blocks_per_tensor <= 0:
        raise ValueError("blocks_per_tensor must be positive")
    if blocks_per_tensor == 1:
        return [0]
    if num_blocks <= blocks_per_tensor:
        return list(range(num_blocks))
    return sorted(
        {round(i * (num_blocks - 1) / (blocks_per_tensor - 1)) for i in range(blocks_per_tensor)}
    )


def _read_samples(checkpoint: Path, blocks_per_tensor: int) -> tuple[bytes, list[dict]]:
    records = read_checkpoint_index(checkpoint)
    chunks: list[bytes] = []
    samples: list[dict] = []
    open_shard: str | None = None
    file = None
    try:
        for name in _iq2_tensor_names(checkpoint):
            record = records[name]
            num_blocks = record.payload_bytes // _BLOCK_BYTES
            indices = _sample_indices(num_blocks, blocks_per_tensor)
            if record.shard != open_shard:
                if file is not None:
                    file.close()
                file = (checkpoint / record.shard).open("rb")
                open_shard = record.shard
            assert file is not None
            for block_index in indices:
                file.seek(record.payload_offset + block_index * _BLOCK_BYTES)
                block = file.read(_BLOCK_BYTES)
                if len(block) != _BLOCK_BYTES:
                    raise EOFError(f"Truncated IQ2_XS block {block_index} in {name}")
                chunks.append(block)
                samples.append(
                    {
                        "tensor": name,
                        "shard": record.shard,
                        "block_index": block_index,
                        "sha256": hashlib.sha256(block).hexdigest(),
                    }
                )
    finally:
        if file is not None:
            file.close()
    return b"".join(chunks), samples


def _stock_decode(library: Path, packed: bytes) -> np.ndarray:
    if len(packed) % _BLOCK_BYTES:
        raise ValueError(f"Packed byte count {len(packed)} is not divisible by {_BLOCK_BYTES}")
    num_values = len(packed) // _BLOCK_BYTES * _BLOCK_SIZE
    ggml = ctypes.CDLL(str(library.resolve()))
    decode = ggml.dequantize_row_iq2_xs
    decode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    decode.restype = None
    source = (ctypes.c_uint8 * len(packed)).from_buffer_copy(packed)
    output = np.empty(num_values, dtype=np.float32)
    decode(source, output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), num_values)
    return output


def _write_test_gguf(
    output_path: Path, llama_source: Path, packed_bytes: bytes, samples: list[dict]
) -> dict:
    """Round-trip sampled payloads through the stock GGUF writer and reader."""

    gguf_package = str(llama_source.resolve() / "gguf-py")
    if gguf_package not in sys.path:
        sys.path.insert(0, gguf_package)
    import gguf

    source_payloads: dict[str, bytearray] = defaultdict(bytearray)
    for offset, sample in enumerate(samples):
        start = offset * _BLOCK_BYTES
        source_payloads[sample["tensor"]].extend(packed_bytes[start : start + _BLOCK_BYTES])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(output_path, "llama")
    writer.add_name("ModelOpt IQ2_XS compatibility sample")
    tensor_sources = {}
    expected_by_name = {}
    for index, (source_name, payload) in enumerate(sorted(source_payloads.items())):
        gguf_name = f"iq2_xs_sample_{index:05d}"
        raw = np.frombuffer(payload, dtype=np.uint8).reshape(1, -1)
        writer.add_tensor(gguf_name, raw, raw_dtype=gguf.GGMLQuantizationType.IQ2_XS)
        tensor_sources[gguf_name] = source_name
        expected_by_name[gguf_name] = bytes(payload)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    payloads = []
    for tensor in gguf.GGUFReader(output_path).tensors:
        expected = expected_by_name[tensor.name]
        actual = tensor.data.tobytes()
        payloads.append(
            {
                "gguf_tensor": tensor.name,
                "source_tensor": tensor_sources[tensor.name],
                "tensor_type": tensor.tensor_type.name,
                "payload_bytes": len(actual),
                "source_sha256": hashlib.sha256(expected).hexdigest(),
                "gguf_sha256": hashlib.sha256(actual).hexdigest(),
                "byte_identical": actual == expected,
            }
        )
    mismatches = sum(not payload["byte_identical"] for payload in payloads)
    return {
        "path": str(output_path.resolve()),
        "payloads": payloads,
        "payload_mismatches": mismatches,
    }


def compare_checkpoint(
    checkpoint: Path,
    ggml_library: Path,
    llama_source: Path,
    *,
    blocks_per_tensor: int = 3,
    test_gguf: Path | None = None,
) -> dict:
    """Return the stock-GGML compatibility report for one checkpoint."""

    checkpoint = checkpoint.resolve()
    packed_bytes, samples = _read_samples(checkpoint, blocks_per_tensor)
    num_blocks = len(packed_bytes) // _BLOCK_BYTES

    packed = torch.frombuffer(bytearray(packed_bytes), dtype=torch.uint8).reshape(
        num_blocks, 1, _BLOCK_BYTES
    )
    logical_shape = torch.tensor([num_blocks, _BLOCK_SIZE], dtype=torch.int64)
    modelopt = dequantize_iq2_xs(packed, logical_shape, dtype=torch.float32).numpy().reshape(-1)
    stock = _stock_decode(ggml_library, packed_bytes)
    modelopt_bits = modelopt.view(np.uint32)
    stock_bits = stock.view(np.uint32)
    different = modelopt_bits != stock_bits

    fields = np.frombuffer(packed_bytes, dtype=np.uint8).reshape(num_blocks, _BLOCK_BYTES)
    scale_bytes = fields[:, :2]
    vector_words = fields[:, 2:66]
    local_scale_bytes = fields[:, 66:74]
    max_abs_difference = float(np.max(np.abs(modelopt - stock))) if modelopt.size else 0.0
    gguf_report = (
        _write_test_gguf(test_gguf, llama_source, packed_bytes, samples)
        if test_gguf is not None
        else None
    )
    passed = not np.any(different) and (
        gguf_report is None or gguf_report["payload_mismatches"] == 0
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed" if passed else "failed",
        "checkpoint": str(checkpoint),
        "llama_cpp": {
            "source": str(llama_source.resolve()),
            "commit": _llama_commit(llama_source),
            "library": str(ggml_library.resolve()),
            "symbol": "dequantize_row_iq2_xs",
        },
        "summary": {
            "iq2_xs_tensors": len({sample["tensor"] for sample in samples}),
            "sampled_blocks": num_blocks,
            "decoded_values": int(modelopt.size),
            "bitwise_differences": int(np.count_nonzero(different)),
            "max_abs_difference": max_abs_difference,
        },
        "encoded_fields": {
            "global_fp16_scale_bytes_sha256": hashlib.sha256(scale_bytes.tobytes()).hexdigest(),
            "vector_words_sha256": hashlib.sha256(vector_words.tobytes()).hexdigest(),
            "local_scale_bytes_sha256": hashlib.sha256(local_scale_bytes.tobytes()).hexdigest(),
            "sample_payload_sha256": hashlib.sha256(packed_bytes).hexdigest(),
        },
        "test_gguf": gguf_report,
        "samples": samples,
    }
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--llama-source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--blocks-per-tensor", type=int, default=3)
    parser.add_argument(
        "--test-gguf",
        type=Path,
        help="Write sampled blocks with stock gguf-py and verify every payload digest.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = compare_checkpoint(
        args.checkpoint,
        args.ggml_library,
        args.llama_source,
        blocks_per_tensor=args.blocks_per_tensor,
        test_gguf=args.test_gguf,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], **report["summary"]}, indent=2))
    print(f"Validation report: {args.report}")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
