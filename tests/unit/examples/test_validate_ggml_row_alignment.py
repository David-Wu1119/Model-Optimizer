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

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

_SCRIPT = (
    Path(__file__).parents[3] / "examples" / "megatron_bridge" / "validate_ggml_row_alignment.py"
)
_SPEC = importlib.util.spec_from_file_location("validate_ggml_row_alignment", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _write_indexed_checkpoint(checkpoint: Path) -> None:
    first = {
        "model.layers.0.mixer.experts.0.up_proj.weight": torch.zeros((8, 256)),
        "model.layers.0.mixer.experts.0.down_proj.weight": torch.zeros((8, 320)),
    }
    second = {
        "model.layers.0.mixer.shared_experts.up_proj.weight": torch.zeros((16, 512)),
        "mtp.layers.0.mixer.experts.0.up_proj.weight": torch.zeros((8, 384)),
    }
    save_file(first, checkpoint / "model-00001-of-00002.safetensors")
    save_file(second, checkpoint / "model-00002-of-00002.safetensors")
    weight_map = {
        name: shard
        for shard, tensors in (
            ("model-00001-of-00002.safetensors", first),
            ("model-00002-of-00002.safetensors", second),
        )
        for name in tensors
    }
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def test_validate_row_alignment_reports_incompatible_shapes(tmp_path: Path) -> None:
    _write_indexed_checkpoint(tmp_path)

    report = _MODULE.validate_row_alignment(
        tmp_path,
        block_size=256,
        includes=["*mixer.experts*.weight", "*mixer.shared_experts*.weight"],
        excludes=["mtp.*"],
    )

    assert report["status"] == "failed"
    assert report["selected_tensors"] == 3
    assert report["compatible_tensors"] == 2
    assert report["incompatible_tensors"] == 1
    assert report["incompatible"] == [
        {
            "name": "model.layers.0.mixer.experts.0.down_proj.weight",
            "shape": [8, 320],
            "row_width": 320,
            "remainder": 64,
            "compatible": False,
        }
    ]


def test_validate_row_alignment_passes_aligned_selection(tmp_path: Path) -> None:
    save_file(
        {"model.layers.0.mixer.experts.0.up_proj.weight": torch.zeros((8, 512))},
        tmp_path / "model.safetensors",
    )

    report = _MODULE.validate_row_alignment(
        tmp_path,
        block_size=256,
        includes=["*experts*.weight"],
        excludes=[],
    )

    assert report["status"] == "passed"
    assert report["compatible_tensors"] == 1
    assert report["incompatible"] == []


def test_validate_row_alignment_requires_matches(tmp_path: Path) -> None:
    save_file({"model.embed.weight": torch.zeros((8, 256))}, tmp_path / "model.safetensors")

    with pytest.raises(ValueError, match="No tensors matched"):
        _MODULE.validate_row_alignment(
            tmp_path,
            block_size=256,
            includes=["*experts*.weight"],
            excludes=[],
        )
