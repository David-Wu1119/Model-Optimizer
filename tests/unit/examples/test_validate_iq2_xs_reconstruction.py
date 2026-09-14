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

"""Tests for the IQ2_XS source-reconstruction validator."""

import importlib.util
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from modelopt.torch.quantization.ggml import quantize_iq2_xs

_EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "megatron_bridge"
_STRUCTURAL_SCRIPT = _EXAMPLES / "validate_mixed_quantized_hf.py"
_STRUCTURAL_SPEC = importlib.util.spec_from_file_location(
    "validate_mixed_quantized_hf", _STRUCTURAL_SCRIPT
)
assert _STRUCTURAL_SPEC is not None and _STRUCTURAL_SPEC.loader is not None
structural_validator = importlib.util.module_from_spec(_STRUCTURAL_SPEC)
sys.modules[_STRUCTURAL_SPEC.name] = structural_validator
_STRUCTURAL_SPEC.loader.exec_module(structural_validator)

_SCRIPT = _EXAMPLES / "validate_iq2_xs_reconstruction.py"
_SPEC = importlib.util.spec_from_file_location("validate_iq2_xs_reconstruction", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
validator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = validator
_SPEC.loader.exec_module(validator)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    checkpoint = tmp_path / "checkpoint"
    reference = tmp_path / "reference"
    checkpoint.mkdir()
    reference.mkdir()
    name = "model.layers.0.mixer.experts.0.up_proj.weight"
    generator = torch.Generator().manual_seed(1234)
    source_weight = torch.randn((2, 257), generator=generator, dtype=torch.float32).bfloat16()
    packed, logical_shape = quantize_iq2_xs(source_weight)
    save_file({name: source_weight}, reference / "model.safetensors")
    save_file(
        {
            name: packed,
            name.removesuffix(".weight") + ".weight_logical_shape": logical_shape.cpu(),
            name.removesuffix(".weight") + ".weight_padded_shape": torch.tensor([2, 512]),
        },
        checkpoint / "model.safetensors",
    )
    return checkpoint, reference, name


def test_reconstruction_matches_direct_packing_and_reports_partial_tail(tmp_path):
    checkpoint, reference, name = _write_fixture(tmp_path)

    report = validator.validate_reconstruction(
        checkpoint,
        reference,
        rows_per_tensor=2,
        maximum_tensors=None,
        require_repack_match=True,
    )

    assert report["status"] == "passed"
    assert report["errors"] == []
    assert report["summary"]["available_iq2_xs_tensors"] == 1
    assert report["summary"]["sampled_tensors"] == 1
    assert report["summary"]["sampled_rows"] == 2
    assert report["summary"]["repack_matches"] == 2
    assert report["summary"]["repack_mismatches"] == 0
    assert report["summary"]["complete_blocks"]["values"] == 512
    assert report["summary"]["partial_tails"]["values"] == 2
    assert all(sample["tensor"] == name for sample in report["samples"])
    assert all(sample["partial_tail"]["values"] == 1 for sample in report["samples"])


def test_reconstruction_detects_source_row_mismatch(tmp_path):
    checkpoint, reference, name = _write_fixture(tmp_path)
    tensors = load_file(reference / "model.safetensors")
    tensors[name] = tensors[name].flip(0)
    save_file(tensors, reference / "model.safetensors")

    report = validator.validate_reconstruction(
        checkpoint,
        reference,
        rows_per_tensor=2,
        maximum_tensors=None,
        require_repack_match=True,
    )

    assert report["status"] == "failed"
    assert report["summary"]["repack_matches"] == 0
    assert report["summary"]["repack_mismatches"] == 2
    assert all(sample["mismatched_bytes"] > 0 for sample in report["samples"])
    assert all("differs from direct packing" in error for error in report["errors"])


def test_tensor_selection_is_even_and_honors_patterns():
    names = [f"layer.{index}.experts.weight" for index in range(10)]

    assert validator._select_tensors(names, (), 3) == [names[0], names[4], names[9]]
    assert validator._select_tensors(names, ("layer.[135].*",), None) == [
        names[1],
        names[3],
        names[5],
    ]
