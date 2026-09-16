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

import gc
import weakref

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.ggml as ggml
import modelopt.torch.quantization.ggml.common as ggml_common
import modelopt.torch.quantization.ggml.iq1_s as iq1_s_module
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.ggml.iq1_s import (
    IQ1_S_BLOCK_BYTES,
    dequantize_iq1_s,
    iq1_s_fake_quant,
    iq1_s_grid,
    quantize_iq1_s,
)
from modelopt.torch.quantization.nn import TensorQuantizer


def test_ggml_public_api_excludes_internal_modules():
    assert "backend" not in ggml.__all__
    assert "common" not in ggml.__all__
    assert "iq1_s" not in ggml.__all__
    assert "iq2_xs" not in ggml.__all__
    assert mtq.ggml is ggml
    assert not hasattr(mtq, "quantize_iq1_s")
    assert not hasattr(mtq, "quantize_iq2_xs")
    assert not hasattr(mtq, "_import_module")


def test_iq1_s_canonical_grid():
    grid = iq1_s_grid()

    assert grid.shape == (2048, 8)
    assert grid.dtype == torch.float32
    assert set(grid.unique().tolist()) == {-1.0, 0.0, 1.0}
    assert grid[0].tolist() == [-1.0] * 8
    assert grid.unique(dim=0).shape[0] == grid.shape[0]


def test_iq1_s_public_grid_mutation_does_not_change_cached_grid():
    grid = iq1_s_grid()
    grid.zero_()

    assert set(iq1_s_grid().unique().tolist()) == {-1.0, 0.0, 1.0}


def test_iq1_s_fake_grid_does_not_poison_real_grid_cache():
    iq1_s_module._GRID_CACHE.clear()
    with FakeTensorMode():
        fake_grid = iq1_s_grid()

    assert isinstance(fake_grid, FakeTensor)
    assert not iq1_s_module._GRID_CACHE
    assert not isinstance(iq1_s_grid(), FakeTensor)


def test_iq1_s_fake_mode_uses_one_default_chunk(monkeypatch):
    chunk_sizes = []

    def fake_encode(blocks, _grid):
        chunk_sizes.append(blocks.shape[0])
        return torch.empty(
            (blocks.shape[0], IQ1_S_BLOCK_BYTES), dtype=torch.uint8, device=blocks.device
        )

    monkeypatch.setattr(iq1_s_module, "_encode_blocks", fake_encode)
    with FakeTensorMode():
        packed, _ = quantize_iq1_s(torch.empty(65, 256))

    assert isinstance(packed, FakeTensor)
    assert chunk_sizes == [65]


def test_iq1_s_fake_mode_dequantizes_public_tensor_shape():
    with FakeTensorMode():
        weight = torch.empty(2, 256)
        packed, shape = quantize_iq1_s(weight)
        reconstructed = dequantize_iq1_s(packed, shape)

    assert isinstance(reconstructed, FakeTensor)
    assert reconstructed.shape == weight.shape


def test_iq1_s_zero_block_has_canonical_zero_encoding():
    weight = torch.zeros((2, 256), dtype=torch.bfloat16)

    packed, shape = quantize_iq1_s(weight)

    assert packed.shape == (2, 1, IQ1_S_BLOCK_BYTES)
    assert packed.dtype == torch.uint8
    assert not packed.any()
    assert shape.tolist() == [2, 256]
    assert torch.equal(dequantize_iq1_s(packed, shape), weight)


def test_iq1_s_dequantizes_ggml_metadata_bit_fields():
    packed = torch.zeros((1, 1, 50), dtype=torch.uint8)
    d = torch.tensor([2.0], dtype=torch.float16).view(torch.uint8)
    packed[0, 0, :2] = d
    entries = torch.tensor([0, 256, 511, 2047], dtype=torch.int64)
    packed[0, 0, 2:6] = (entries & 0xFF).to(torch.uint8)
    qh = (
        ((entries[0] >> 8) & 7)
        | (((entries[1] >> 8) & 7) << 3)
        | (((entries[2] >> 8) & 7) << 6)
        | (((entries[3] >> 8) & 7) << 9)
        | (3 << 12)
        | (1 << 15)
    )
    packed[0, 0, 34] = (qh & 0xFF).to(torch.uint8)
    packed[0, 0, 35] = (qh >> 8).to(torch.uint8)

    decoded = dequantize_iq1_s(packed, torch.tensor([1, 256]), dtype=torch.float32)
    expected = (iq1_s_grid()[entries] - 0.125) * 14.0

    assert torch.equal(decoded[0, :32].reshape(4, 8), expected)


def test_iq1_s_round_trip_and_payload_fields():
    generator = torch.Generator().manual_seed(1234)
    weight = torch.randn((2, 256), generator=generator, dtype=torch.bfloat16)

    packed, shape = quantize_iq1_s(weight, block_chunk_size=1)
    reconstructed = dequantize_iq1_s(packed, shape)

    assert packed.shape == (2, 1, 50)
    assert reconstructed.shape == weight.shape
    assert reconstructed.dtype == torch.bfloat16
    normalized_mse = (
        reconstructed.float() - weight.float()
    ).square().mean() / weight.float().square().mean()
    assert normalized_mse < 0.25

    blocks = packed.reshape(-1, 50)
    qh = blocks[:, 34:50:2].to(torch.int64) | (blocks[:, 35:50:2].to(torch.int64) << 8)
    assert torch.all(((qh >> 12) & 0x7) < 8)
    assert torch.all((qh & 0xFFF) < 0x1000)


def test_iq1_s_encoding_is_independent_of_default_dtype():
    weight = torch.randn((1, 256), generator=torch.Generator().manual_seed(4321))
    expected, _ = quantize_iq1_s(weight)
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        actual, _ = quantize_iq1_s(weight)
    finally:
        torch.set_default_dtype(original_dtype)

    assert torch.equal(actual, expected)


def test_iq1_s_requires_complete_last_dimension_blocks():
    with pytest.raises(ValueError, match="last weight dimension"):
        quantize_iq1_s(torch.ones(2, 257))


def test_iq1_s_fake_quant_has_pass_through_gradient():
    class Quantizer:
        num_bits = "iq1_s"
        backend_extra_args = {"search_impl": "auto"}
        _quantizer_cache = None

    weight = torch.randn(1, 256, requires_grad=True)
    output = iq1_s_fake_quant(weight, Quantizer())
    output.sum().backward()

    assert torch.equal(weight.grad, torch.ones_like(weight))


def test_iq1_s_cache_is_frozen_after_quantization():
    model = torch.nn.Linear(256, 1, bias=False)
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {
                    "num_bits": "iq1_s",
                    "block_sizes": {-1: 256},
                    "backend": "ggml",
                    "backend_extra_args": {"search_impl": "auto"},
                },
                "enable": True,
            },
        ],
        "algorithm": "max",
    }

    mtq.quantize(model, config)

    assert model.weight_quantizer._reconstruction_cache_frozen
    assert model.weight_quantizer._reconstruction_cache is None


def test_iq1_s_fake_quant_reuses_cached_reconstruction(monkeypatch):
    calls = 0
    original_quantize = iq1_s_module._quantize_iq1_s_packed

    def counting_quantize(weight):
        nonlocal calls
        calls += 1
        return original_quantize(weight)

    monkeypatch.setattr(iq1_s_module, "_quantize_iq1_s_packed", counting_quantize)
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits="iq1_s",
            block_sizes={-1: 256},
            backend="ggml",
            backend_extra_args={"search_impl": "auto"},
        )
    ).eval()
    weight = torch.randn(1, 256)

    quantizer(weight)
    quantizer(weight)
    assert calls == 2

    quantizer.freeze_reconstruction_cache()
    quantizer(weight)
    quantizer(weight)
    assert calls == 3

    with torch.no_grad():
        weight.add_(1)
    quantizer(weight)
    assert calls == 4

    weight.data = torch.randn_like(weight)
    quantizer(weight)
    assert calls == 5

    weight.data.add_(1)
    quantizer.reset_amax()
    assert quantizer._reconstruction_cache is None
    quantizer.freeze_reconstruction_cache()
    quantizer(weight)
    assert calls == 6

    quantizer(weight.clone())
    assert calls == 7


def test_iq1_s_fake_quant_preserves_a_foreign_cache_object():
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits="iq1_s",
            block_sizes={-1: 256},
            backend="ggml",
            backend_extra_args={"search_impl": "auto"},
        )
    ).eval()
    foreign_cache = object()
    quantizer._quantizer_cache = foreign_cache
    quantizer.freeze_reconstruction_cache()

    quantizer(torch.randn(1, 256))

    assert quantizer._quantizer_cache is foreign_cache


def test_iq1_s_fake_quant_preserves_a_foreign_dict_cache():
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits="iq1_s",
            block_sizes={-1: 256},
            backend="ggml",
            backend_extra_args={"search_impl": "auto"},
        )
    ).eval()
    foreign_value = object()
    foreign_cache = {"foreign": foreign_value, "iq1_s_packed": torch.empty(0)}
    quantizer._quantizer_cache = foreign_cache

    quantizer(torch.randn(1, 256))

    assert quantizer._quantizer_cache is foreign_cache


def test_iq1_s_fake_quant_abandons_cache_for_transient_inputs(monkeypatch):
    calls = 0
    warning_messages = []
    original_quantize = iq1_s_module._quantize_iq1_s_packed

    def counting_quantize(weight):
        nonlocal calls
        calls += 1
        return original_quantize(weight)

    monkeypatch.setattr(iq1_s_module, "_quantize_iq1_s_packed", counting_quantize)
    monkeypatch.setattr(ggml_common, "warn_rank_0", warning_messages.append)
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits="iq1_s",
            block_sizes={-1: 256},
            backend="ggml",
            backend_extra_args={"search_impl": "auto"},
        )
    ).eval()
    quantizer.freeze_reconstruction_cache()

    temporary = torch.randn(1, 256, dtype=torch.bfloat16).float()
    quantizer(temporary)
    del temporary
    gc.collect()

    quantizer(torch.randn(1, 256, dtype=torch.bfloat16).float())
    assert quantizer._reconstruction_cache == {"abandoned": True}
    assert len(warning_messages) == 1
    assert "reconstruction cache abandoned" in warning_messages[0]

    quantizer(torch.randn(1, 256, dtype=torch.bfloat16).float())
    assert calls == 3
    assert quantizer._reconstruction_cache == {"abandoned": True}
    assert len(warning_messages) == 1


def test_cached_reconstruction_bypasses_storage_identity_in_fake_mode(monkeypatch):
    quantizer = TensorQuantizer().eval()
    quantizer.freeze_reconstruction_cache()
    existing_cache = {"existing": object()}
    quantizer._reconstruction_cache = existing_cache
    monkeypatch.setattr(
        ggml_common,
        "_cache_identity",
        lambda _inputs: pytest.fail("fake tensors must not use storage identity"),
    )

    mode = FakeTensorMode()
    inputs = mode.from_tensor(torch.randn(1, 256))
    monkeypatch.setattr(ggml_common, "_torch_detect_fake_mode", None)
    output = ggml_common.cached_reconstruction(
        inputs,
        quantizer,
        cache_namespace="test",
        quantize=lambda value: (value, torch.empty(0)),
        dequantize=lambda packed, _shape, dtype: packed.to(dtype),
    )

    assert isinstance(output, FakeTensor)
    assert quantizer._reconstruction_cache is existing_cache


def test_iq1_s_fake_quant_skips_storage_identity_when_cache_is_disabled(monkeypatch):
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits="iq1_s",
            block_sizes={-1: 256},
            backend="ggml",
            backend_extra_args={"search_impl": "auto"},
        )
    ).eval()

    monkeypatch.setattr(
        ggml_common,
        "_cache_identity",
        lambda _inputs: pytest.fail("storage identity must not run while caching is disabled"),
    )

    quantizer(torch.randn(1, 256))


def test_iq1_s_frozen_cache_requires_explicit_clear_after_data_write():
    """Document that data-mediated writes bypass the signature and require invalidation."""
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits="iq1_s",
            block_sizes={-1: 256},
            backend="ggml",
            backend_extra_args={"search_impl": "auto"},
        )
    ).eval()
    weight = torch.randn(1, 256)
    quantizer.freeze_reconstruction_cache()

    before = quantizer(weight)
    weight.data.add_(1)
    assert torch.allclose(quantizer(weight), before, rtol=1e-5, atol=1e-6)

    quantizer.clear_reconstruction_cache()
    assert not torch.allclose(quantizer(weight), before, rtol=1e-5, atol=1e-6)


def test_iq1_s_fake_quant_handles_inference_tensors_without_a_version(monkeypatch):
    calls = 0
    original_quantize = iq1_s_module._quantize_iq1_s_packed

    def counting_quantize(weight):
        nonlocal calls
        calls += 1
        return original_quantize(weight)

    monkeypatch.setattr(iq1_s_module, "_quantize_iq1_s_packed", counting_quantize)
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits="iq1_s",
            block_sizes={-1: 256},
            backend="ggml",
            backend_extra_args={"search_impl": "auto"},
        )
    ).eval()
    quantizer.freeze_reconstruction_cache()

    with torch.inference_mode():
        weight = torch.randn(1, 256)
        quantizer(weight)
        quantizer(weight)
        weight.add_(1)
        quantizer.clear_reconstruction_cache()
        quantizer(weight)

    assert calls == 2


def test_iq1_s_fake_quant_cache_does_not_retain_temporary_storage():
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits="iq1_s",
            block_sizes={-1: 256},
            backend="ggml",
            backend_extra_args={"search_impl": "auto"},
        )
    ).eval()
    quantizer.freeze_reconstruction_cache()
    weight = torch.randn(1, 256, dtype=torch.bfloat16)
    temporary = weight.float()
    storage_ref = weakref.ref(temporary.untyped_storage())

    quantizer(temporary)
    del temporary
    gc.collect()

    assert storage_ref() is None


def test_iq1_s_fake_quant_distinguishes_storage_offsets(monkeypatch):
    calls = 0
    original_quantize = iq1_s_module._quantize_iq1_s_packed

    def counting_quantize(weight):
        nonlocal calls
        calls += 1
        return original_quantize(weight)

    monkeypatch.setattr(iq1_s_module, "_quantize_iq1_s_packed", counting_quantize)
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits="iq1_s",
            block_sizes={-1: 256},
            backend="ggml",
            backend_extra_args={"search_impl": "auto"},
        )
    ).eval()
    quantizer.freeze_reconstruction_cache()
    storage = torch.randn(512)

    quantizer(storage[:256].reshape(1, 256))
    quantizer(storage[256:].reshape(1, 256))

    assert calls == 2
