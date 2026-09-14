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

"""Materialize a mixed ModelOpt checkpoint as GGUF without changing llama.cpp.

The pinned stock converter owns model metadata, tensor naming, BF16 conversion,
and NVFP4 repacking. This validation bridge consumes only the already-packed
IQ2_XS tensors, writes their canonical bytes through stock ``gguf-py``, and then
verifies the resulting GGUF payloads byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

_IQ2_XS_BLOCK_BYTES = 74
_IQ2_XS_BLOCK_SIZE = 256
_ROUTED_EXPERT = re.compile(
    r"^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up_proj|down_proj)\.weight$"
)


def packed_rows(tensor: torch.Tensor) -> tuple[np.ndarray, list[int]]:
    """Collapse the explicit block/payload axes into GGUF's byte-row layout."""

    if tensor.dtype != torch.uint8:
        raise ValueError(f"IQ2_XS payload must use uint8, got {tensor.dtype}")
    if tensor.ndim < 3 or tensor.shape[-1] != _IQ2_XS_BLOCK_BYTES:
        raise ValueError(
            f"IQ2_XS payload must have shape [..., blocks, {_IQ2_XS_BLOCK_BYTES}], "
            f"got {tuple(tensor.shape)}"
        )
    raw_shape = (*tensor.shape[:-2], tensor.shape[-2] * _IQ2_XS_BLOCK_BYTES)
    raw = tensor.contiguous().reshape(raw_shape).cpu().numpy()
    logical_shape = [*raw_shape[:-1], tensor.shape[-2] * _IQ2_XS_BLOCK_SIZE]
    return raw, logical_shape


def _shape_sidecar_names(weight_name: str) -> tuple[str, str]:
    base = weight_name.removesuffix(".weight")
    return base + ".weight_logical_shape", base + ".weight_padded_shape"


def _git_commit(source: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _normalize_layer_name(name: str) -> str:
    if name.startswith("model.layers.") and ".mixer." in name:
        return name.replace("model.layers.", "backbone.layers.", 1)
    return name


def _load_iq2_layers(checkpoint: Path) -> set[str]:
    config_path = checkpoint / "hf_quant_config.json"
    config = json.loads(config_path.read_text())
    layers = config["quantization"]["quantized_layers"]
    return {
        _normalize_layer_name(name)
        for name, layer_config in layers.items()
        if layer_config.get("quant_algo") == "IQ2_XS"
    }


def _matches_iq2_layer(name: str, layers: set[str]) -> bool:
    return any(
        name == f"{layer}.weight" or (name.startswith(f"{layer}.") and name.endswith(".weight"))
        for layer in layers
    )


def _target_record(
    raw: np.ndarray,
    source_tensors: list[str],
    logical_shape: list[int],
    padded_shape: list[int],
) -> dict:
    payload = raw.tobytes()
    return {
        "source_tensors": source_tensors,
        "logical_shape": logical_shape,
        "padded_shape": padded_shape,
        "payload_bytes": len(payload),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _verify_gguf_payloads(
    gguf: Any, path: Path, expected: dict[str, dict]
) -> tuple[list[dict], list[str]]:
    tensors = {tensor.name: tensor for tensor in gguf.GGUFReader(path).tensors}
    results = []
    errors = []
    for name, record in sorted(expected.items()):
        tensor = tensors.get(name)
        if tensor is None:
            errors.append(f"Missing IQ2_XS tensor in GGUF: {name}")
            continue
        payload = tensor.data.tobytes()
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        result = {
            "gguf_tensor": name,
            **record,
            "tensor_type": tensor.tensor_type.name,
            "gguf_sha256": actual_sha256,
            "byte_identical": actual_sha256 == record["source_sha256"],
        }
        results.append(result)
        if tensor.tensor_type.name != "IQ2_XS":
            errors.append(f"Tensor {name} has type {tensor.tensor_type.name}, expected IQ2_XS")
        if not result["byte_identical"]:
            errors.append(f"IQ2_XS payload differs for {name}")
    return results, errors


def materialize(checkpoint: Path, llama_source: Path, outfile: Path) -> tuple[dict[str, dict], str]:
    """Run the stock converter with an in-process IQ2_XS payload adapter."""

    checkpoint = checkpoint.resolve()
    llama_source = llama_source.resolve()
    sys.path.insert(0, str(llama_source))
    sys.path.insert(1, str(llama_source / "gguf-py"))

    import gguf
    from conversion.base import LazyTorchTensor, ModelBase
    from conversion.nemotron import NemotronHModel

    expected: dict[str, dict] = {}

    @ModelBase.register("NemotronHForCausalLM")
    class ModelOptIQNemotronHModel(NemotronHModel):
        model_arch = NemotronHModel.model_arch

        def _pop_iq2_shapes(self, source_name: str) -> tuple[list[int], list[int]]:
            logical_name, padded_name = _shape_sidecar_names(source_name)
            try:
                logical = LazyTorchTensor.to_eager(self.model_tensors.pop(logical_name)())
                padded = LazyTorchTensor.to_eager(self.model_tensors.pop(padded_name)())
            except KeyError as error:
                raise ValueError(f"Missing IQ2_XS shape sidecar for {source_name}") from error
            return logical.tolist(), padded.tolist()

        def _write_iq2_tensor(self, source_name: str, target_name: str) -> None:
            generator = self.model_tensors.pop(source_name)
            tensor = LazyTorchTensor.to_eager(generator())
            raw, inferred_padded_shape = packed_rows(tensor)
            logical_shape, padded_shape = self._pop_iq2_shapes(source_name)
            if padded_shape != inferred_padded_shape:
                raise ValueError(
                    f"IQ2_XS padded shape mismatch for {source_name}: metadata "
                    f"{padded_shape}, payload {inferred_padded_shape}"
                )
            self.gguf_writer.add_tensor(
                target_name, raw, raw_dtype=gguf.GGMLQuantizationType.IQ2_XS
            )
            expected[target_name] = _target_record(raw, [source_name], logical_shape, padded_shape)

        def _write_routed_iq2_tensors(self, names: list[str]) -> set[str]:
            grouped: dict[tuple[int, str], list[tuple[int, str]]] = defaultdict(list)
            for name in names:
                match = _ROUTED_EXPERT.match(name)
                if match:
                    layer, expert, projection = match.groups()
                    grouped[(int(layer), projection)].append((int(expert), name))

            consumed = set()
            expected_experts = int(self.hparams["n_routed_experts"])
            for (layer, projection), experts in sorted(grouped.items()):
                experts.sort()
                expert_ids = [expert for expert, _ in experts]
                if expert_ids != list(range(expected_experts)):
                    raise ValueError(
                        f"Layer {layer} {projection} has expert IDs {expert_ids}, "
                        f"expected 0..{expected_experts - 1}"
                    )
                source_names = [name for _, name in experts]
                tensors = [
                    LazyTorchTensor.to_eager(self.model_tensors.pop(name)())
                    for name in source_names
                ]
                raw_parts = [packed_rows(tensor)[0] for tensor in tensors]
                source_shapes = [self._pop_iq2_shapes(name) for name in source_names]
                if any(shapes != source_shapes[0] for shapes in source_shapes[1:]):
                    raise ValueError(
                        f"Layer {layer} {projection} has inconsistent expert shape metadata"
                    )
                raw = np.stack(raw_parts, axis=0)
                logical_shape = [
                    len(tensors),
                    *source_shapes[0][0],
                ]
                padded_shape = [len(tensors), *source_shapes[0][1]]
                inferred_padded_shape = [len(tensors), *packed_rows(tensors[0])[1]]
                if padded_shape != inferred_padded_shape:
                    raise ValueError(
                        f"Layer {layer} {projection} padded shape metadata {padded_shape} "
                        f"does not match payload {inferred_padded_shape}"
                    )
                merged_name = f"model.layers.{layer}.mlp.experts.{projection}.weight"
                target_name = self.map_tensor_name(merged_name)
                self.gguf_writer.add_tensor(
                    target_name, raw, raw_dtype=gguf.GGMLQuantizationType.IQ2_XS
                )
                expected[target_name] = _target_record(
                    raw, source_names, logical_shape, padded_shape
                )
                consumed.update(source_names)
            return consumed

        def _write_iq2_tensors(self) -> None:
            layers = _load_iq2_layers(checkpoint)
            names = sorted(name for name in self.model_tensors if _matches_iq2_layer(name, layers))
            if not names:
                raise ValueError("No IQ2_XS tensors were found in the checkpoint")

            consumed = self._write_routed_iq2_tensors(names)
            for name in names:
                if name in consumed:
                    continue
                target_name = self.map_tensor_name(name)
                self._write_iq2_tensor(name, target_name)

        def prepare_tensors(self) -> None:
            self._write_iq2_tensors()
            super().prepare_tensors()

    original_argv = sys.argv
    try:
        sys.argv = [
            str(llama_source / "convert_hf_to_gguf.py"),
            str(checkpoint),
            "--outfile",
            str(outfile.resolve()),
            "--outtype",
            "bf16",
            "--use-temp-file",
            "--no-mtp",
        ]
        runpy.run_path(str(llama_source / "convert_hf_to_gguf.py"), run_name="__main__")
    finally:
        sys.argv = original_argv
    return expected, _git_commit(llama_source)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--llama-source", type=Path, required=True)
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    expected, commit = materialize(args.checkpoint, args.llama_source, args.outfile)

    sys.path.insert(0, str(args.llama_source.resolve() / "gguf-py"))
    import gguf

    payloads, errors = _verify_gguf_payloads(gguf, args.outfile, expected)
    source_tensors = sum(len(payload["source_tensors"]) for payload in payloads)
    summary = {
        "iq2_xs_source_tensors": source_tensors,
        "iq2_xs_gguf_tensors": len(payloads),
        "iq2_xs_payload_bytes": sum(payload["payload_bytes"] for payload in payloads),
        "payload_mismatches": sum(not payload["byte_identical"] for payload in payloads),
    }
    report: dict[str, Any] = {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed" if not errors else "failed",
        "checkpoint": str(args.checkpoint.resolve()),
        "gguf": str(args.outfile.resolve()),
        "llama_cpp": {
            "source": str(args.llama_source.resolve()),
            "commit": commit,
            "source_modified": subprocess.run(
                ["git", "-C", str(args.llama_source), "diff", "--quiet"], check=False
            ).returncode
            != 0,
        },
        "summary": summary,
        "iq2_xs_payloads": payloads,
        "errors": errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], **summary}, indent=2))
    print(f"Validation report: {args.report}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
