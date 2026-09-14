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

"""Tests for the mixed unified-HF checkpoint validator."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "megatron_bridge"
    / "validate_mixed_quantized_hf.py"
)
_SPEC = importlib.util.spec_from_file_location("validate_mixed_quantized_hf", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
validator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = validator
_SPEC.loader.exec_module(validator)

_ORACLE_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "megatron_bridge"
    / "validate_iq2_xs_stock_ggml.py"
)
_ORACLE_SPEC = importlib.util.spec_from_file_location("validate_iq2_xs_stock_ggml", _ORACLE_SCRIPT)
assert _ORACLE_SPEC is not None and _ORACLE_SPEC.loader is not None
oracle = importlib.util.module_from_spec(_ORACLE_SPEC)
sys.modules[_ORACLE_SPEC.name] = oracle
_ORACLE_SPEC.loader.exec_module(oracle)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    source.mkdir()
    checkpoint.mkdir()

    source_tensors = {
        "model.embed_tokens.weight": torch.ones((8, 256), dtype=torch.bfloat16),
        "model.layers.0.mixer.in_proj.weight": torch.ones((8, 256), dtype=torch.bfloat16),
        "model.layers.0.mixer.out_proj.weight": torch.ones((8, 256), dtype=torch.bfloat16),
        "model.layers.1.mixer.experts.0.up_proj.weight": torch.ones((8, 257), dtype=torch.bfloat16),
        "model.layers.1.mixer.shared_experts.down_proj.weight": torch.ones(
            (8, 257), dtype=torch.bfloat16
        ),
        "model.layers.1.mixer.gate.weight": torch.ones((8, 256), dtype=torch.bfloat16),
        "mtp.layers.0.mixer.experts.0.up_proj.weight": torch.ones((8, 256), dtype=torch.bfloat16),
    }
    save_file(source_tensors, source / "model.safetensors")

    checkpoint_tensors = {
        "model.embed_tokens.weight": source_tensors["model.embed_tokens.weight"],
        "model.layers.0.mixer.in_proj.weight": torch.arange(8 * 128, dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(8, 128),
        "model.layers.0.mixer.in_proj.weight_scale": torch.ones((8, 16), dtype=torch.float8_e4m3fn),
        "model.layers.0.mixer.in_proj.weight_scale_2": torch.tensor(1.0),
        "model.layers.0.mixer.out_proj.weight": torch.zeros((8, 128), dtype=torch.uint8),
        "model.layers.0.mixer.out_proj.weight_scale": torch.ones(
            (8, 16), dtype=torch.float8_e4m3fn
        ),
        "model.layers.0.mixer.out_proj.weight_scale_2": torch.tensor(1.0),
        "model.layers.1.mixer.experts.0.up_proj.weight": torch.arange(8 * 2 * 74, dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(8, 2, 74),
        "model.layers.1.mixer.experts.0.up_proj.weight_logical_shape": torch.tensor([8, 257]),
        "model.layers.1.mixer.experts.0.up_proj.weight_padded_shape": torch.tensor([8, 512]),
        "model.layers.1.mixer.shared_experts.down_proj.weight": torch.zeros(
            (8, 2, 74), dtype=torch.uint8
        ),
        "model.layers.1.mixer.shared_experts.down_proj.weight_logical_shape": torch.tensor(
            [8, 257]
        ),
        "model.layers.1.mixer.shared_experts.down_proj.weight_padded_shape": torch.tensor([8, 512]),
        "model.layers.1.mixer.gate.weight": source_tensors["model.layers.1.mixer.gate.weight"],
    }
    save_file(checkpoint_tensors, checkpoint / "model.safetensors")

    quantized_layers = {
        "model.layers.0.mixer.in_proj": {
            "quant_algo": "W4A16_NVFP4",
            "group_size": 16,
        },
        "model.layers.0.mixer.out_proj": {
            "quant_algo": "W4A16_NVFP4",
            "group_size": 16,
        },
        "model.layers.1.mixer.experts": {
            "quant_algo": "IQ2_XS",
            "group_size": 256,
            "block_payload_bytes": 74,
            "packing": "ggml",
            "row_padding": "right",
            "logical_shape_key": "weight_logical_shape",
            "padded_shape_key": "weight_padded_shape",
        },
        "model.layers.1.mixer.shared_experts.down_proj": {
            "quant_algo": "IQ2_XS",
            "group_size": 256,
            "block_payload_bytes": 74,
            "packing": "ggml",
            "row_padding": "right",
            "logical_shape_key": "weight_logical_shape",
            "padded_shape_key": "weight_padded_shape",
        },
    }
    (checkpoint / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "producer": {"name": "modelopt", "version": "test"},
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                    "kv_cache_quant_algo": None,
                    "quantized_layers": quantized_layers,
                },
            }
        )
    )
    return checkpoint, source


def test_validate_checkpoint_passes_and_hashes_iq2_payloads(tmp_path):
    checkpoint, source = _write_fixture(tmp_path)

    report = validator.validate_checkpoint(checkpoint, source)

    assert report["status"] == "passed"
    assert report["errors"] == []
    assert report["summary"] == {
        "total_tensors": 14,
        "total_weight_tensors": 6,
        "iq2_xs_layers": 2,
        "iq2_xs_tensors": 2,
        "iq2_xs_blocks": 32,
        "iq2_xs_logical_weights": 4112,
        "iq2_xs_padded_weights": 8192,
        "iq2_xs_payload_bytes": 2368,
        "nvfp4_layers": 2,
        "nvfp4_tensors": 2,
        "unquantized_weight_tensors": 2,
    }
    assert len(report["iq2_xs"]["aggregate_sha256"]) == 64
    assert all(len(tensor["sha256"]) == 64 for tensor in report["iq2_xs"]["tensors"])


def test_checkpoint_index_selects_authoritative_duplicate(tmp_path):
    save_file(
        {
            "shared.weight": torch.ones((2, 2)),
            "first.weight": torch.ones((1, 2)),
        },
        tmp_path / "model-00001-of-00002.safetensors",
    )
    save_file(
        {
            "shared.weight": torch.zeros((3, 2)),
            "second.weight": torch.ones((1, 3)),
        },
        tmp_path / "model-00002-of-00002.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "first.weight": "model-00001-of-00002.safetensors",
                    "second.weight": "model-00002-of-00002.safetensors",
                    "shared.weight": "model-00002-of-00002.safetensors",
                }
            }
        )
    )

    records = validator.read_checkpoint_index(tmp_path)

    assert set(records) == {"first.weight", "second.weight", "shared.weight"}
    assert records["shared.weight"].shard == "model-00002-of-00002.safetensors"
    assert records["shared.weight"].shape == (3, 2)


def test_validate_checkpoint_rejects_wrong_iq2_payload_shape(tmp_path):
    checkpoint, source = _write_fixture(tmp_path)
    state = load_file(checkpoint / "model.safetensors")
    state["model.layers.1.mixer.experts.0.up_proj.weight"] = torch.zeros(
        (8, 1, 73), dtype=torch.uint8
    )
    save_file(state, checkpoint / "model.safetensors")

    report = validator.validate_checkpoint(checkpoint, source, compute_digests=False)

    assert report["status"] == "failed"
    assert any("expected [..., blocks, 74]" in error for error in report["errors"])


def test_stock_ggml_oracle_samples_every_iq2_tensor(tmp_path, monkeypatch):
    checkpoint, _ = _write_fixture(tmp_path)

    def modelopt_decode(_library, packed_bytes):
        num_blocks = len(packed_bytes) // 74
        packed = torch.frombuffer(bytearray(packed_bytes), dtype=torch.uint8).reshape(
            num_blocks, 1, 74
        )
        shape = torch.tensor([num_blocks, 256], dtype=torch.int64)
        return oracle.dequantize_iq2_xs(packed, shape, dtype=torch.float32).numpy().reshape(-1)

    monkeypatch.setattr(oracle, "_stock_decode", modelopt_decode)
    monkeypatch.setattr(oracle, "_llama_commit", lambda _source: "abc123")

    report = oracle.compare_checkpoint(
        checkpoint,
        tmp_path / "libggml-base.so",
        tmp_path / "llama.cpp",
        blocks_per_tensor=3,
    )

    assert report["status"] == "passed"
    assert report["llama_cpp"]["commit"] == "abc123"
    assert report["summary"] == {
        "iq2_xs_tensors": 2,
        "sampled_blocks": 6,
        "decoded_values": 1536,
        "bitwise_differences": 0,
        "max_abs_difference": 0.0,
    }
    assert len(report["samples"]) == 6
    assert len(report["encoded_fields"]["sample_payload_sha256"]) == 64


def test_stock_ggml_oracle_reports_bit_difference(tmp_path, monkeypatch):
    checkpoint, _ = _write_fixture(tmp_path)

    def mismatching_decode(_library, packed_bytes):
        output = np.zeros(len(packed_bytes) // 74 * 256, dtype=np.float32)
        output[0] = 1.0
        return output

    monkeypatch.setattr(oracle, "_stock_decode", mismatching_decode)
    monkeypatch.setattr(oracle, "_llama_commit", lambda _source: "abc123")

    report = oracle.compare_checkpoint(
        checkpoint,
        tmp_path / "libggml-base.so",
        tmp_path / "llama.cpp",
        blocks_per_tensor=1,
    )

    assert report["status"] == "failed"
    assert report["summary"]["bitwise_differences"] > 0
