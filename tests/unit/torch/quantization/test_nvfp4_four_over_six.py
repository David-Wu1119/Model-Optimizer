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

"""CPU tests for NVFP4 Four-Over-Six (4/6) adaptive weight scaling.

4/6 is weight-only: the ``four_over_six: True`` block_sizes flag selects the 256 FP8
normalization max (vs 448); the per-block M=6 vs M=4 choice is made by MSE weight
calibration (arXiv:2512.02010).
"""

from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizeConfig, QuantizerAttributeConfig, choices
from modelopt.torch.quantization.nn import NVFP4StaticQuantizer
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
from modelopt.torch.quantization.utils.numeric_utils import E2M1_MAX, E4M3_MAX, E4M3_MAX_46

BLOCK_SIZE = 16


class TestConstants:
    def test_fp8_and_e2m1_constants(self):
        assert E4M3_MAX == 448.0
        assert E4M3_MAX_46 == 256.0
        assert E2M1_MAX == 6.0


class TestScalingFactor2:
    def test_256_vs_448_denominator(self):
        # 4/6 selects the 256 FP8 normalization via the static quantizer path.
        global_amax = torch.tensor(2.0)
        q_default = SimpleNamespace(block_sizes={-1: BLOCK_SIZE}, global_amax=global_amax)
        q_46 = SimpleNamespace(
            block_sizes={-1: BLOCK_SIZE, "four_over_six": True}, global_amax=global_amax
        )
        wsf2_default = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(q_default)
        wsf2_46 = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(q_46)
        # wsf2 = global_amax / (6 * m_fp8); only m_fp8 differs (448 vs 256).
        assert torch.allclose(
            wsf2_46 / wsf2_default, torch.tensor(E4M3_MAX / E4M3_MAX_46), rtol=1e-6
        )


class TestRoundTripScales:
    def test_no_zero_or_nan_scales(self):
        torch.manual_seed(1)
        weight = torch.cat([torch.randn(4, BLOCK_SIZE), torch.full((4, BLOCK_SIZE), 1e-12)], dim=0)
        per_block_scale, _ = NVFP4QTensor.get_weights_scaling_factor(weight, BLOCK_SIZE)
        s = per_block_scale.float()
        assert torch.isfinite(s).all(), f"Non-finite 4/6 scales: {s.tolist()}"
        assert (s > 0).all(), f"Zero 4/6 scales: {s.tolist()}"


class TestNVFP4FourOverSixConfig:
    @staticmethod
    def _block_sizes(cfg, name):
        entry = next(e for e in cfg["quant_cfg"] if e["quantizer_name"] == name)
        return entry["cfg"]["block_sizes"]

    def test_weight_quantizer_is_static_with_four_over_six(self):
        bs = self._block_sizes(mtq.NVFP4_FOUR_OVER_SIX_CFG, "*weight_quantizer")
        assert bs.get("type") == "static"
        # Schema coerces the bool to int 1; the feature reads it truthily.
        assert bs.get("four_over_six")

    def test_input_quantizer_unchanged(self):
        bs = self._block_sizes(mtq.NVFP4_FOUR_OVER_SIX_CFG, "*input_quantizer")
        assert not bs.get("four_over_six", False)

    def test_registered_in_choices(self):
        assert "NVFP4_FOUR_OVER_SIX_CFG" in choices


class TestStaticQuantizerFourOverSixThreading:
    """NVFP4StaticQuantizer._fake_quantize threads fp8_max_for_normalization from the
    four_over_six flag: 256 when enabled, 448 otherwise.

    The per-block M=6/M=4 choice itself is made by MSE calibration.
    """

    @staticmethod
    def _make_static_quantizer(four_over_six: bool) -> NVFP4StaticQuantizer:
        block_sizes = {-1: BLOCK_SIZE, "type": "static", "scale_bits": (4, 3)}
        if four_over_six:
            block_sizes["four_over_six"] = True
        cfg = QuantizerAttributeConfig(num_bits=(2, 1), block_sizes=block_sizes)
        q = NVFP4StaticQuantizer(quant_attribute_cfg=cfg)
        q.amax = torch.full((1, 4), 0.5)
        q.global_amax = torch.tensor(2.0)
        return q

    def _captured_fp8_max(self, monkeypatch, four_over_six: bool) -> float:
        import modelopt.torch.quantization.nn.modules.tensor_quantizer as tqm

        captured = {}

        def spy(*args, **kwargs):
            # Call site: (inputs, amax, global_amax, quantize_block_scales,
            #             fp8_max_for_normalization, dtype, pass_through_bwd).
            # The 4/6 → 256 vs 448 selection happens before this call, so capturing the
            # threaded value is enough; return a passthrough to avoid the triton kernel
            # (unavailable on CPU) — this tests the threading, not the kernel.
            captured["fp8_max"] = args[4]
            return args[0]

        monkeypatch.setattr(tqm, "static_blockwise_fp4_fake_quant", spy)
        q = self._make_static_quantizer(four_over_six)
        q._fake_quantize(torch.randn(1, 4 * BLOCK_SIZE))
        return captured["fp8_max"]

    def test_four_over_six_threads_256(self, monkeypatch):
        assert self._captured_fp8_max(monkeypatch, four_over_six=True) == E4M3_MAX_46

    def test_default_threads_448(self, monkeypatch):
        assert self._captured_fp8_max(monkeypatch, four_over_six=False) == E4M3_MAX


class TestCompressUnsupported:
    """mtq.compress (TensorQuantizer._real_quantize) must reject 4/6: the per-block
    M=4/M=6 choice baked into amax by MSE calibration is not preserved by real quantization.
    """

    def test_real_quantize_raises_for_four_over_six(self):
        q = TestStaticQuantizerFourOverSixThreading._make_static_quantizer(four_over_six=True)
        with pytest.raises(NotImplementedError, match="Four-Over-Six"):
            q._real_quantize(torch.randn(1, 4 * BLOCK_SIZE))


# --------------------------------------------------------------------------------------
# The `four_over_six` calibration algorithm
# --------------------------------------------------------------------------------------

# How 4/6 was written before it had a name, copy-pasted across five shipped recipes.
# Every assertion below that `four_over_six` reproduces 4/6 is an assertion about *this*.
LEGACY_FOUR_OVER_SIX_STANZA = {
    "method": "mse",
    "fp8_scale_sweep": False,
    "start_multiplier": 1.0,
    "stop_multiplier": 1.5,
    "step_size": 0.5,
}

NVFP4_FOUR_OVER_SIX_ATTRS = {
    "num_bits": (2, 1),
    "block_sizes": {-1: BLOCK_SIZE, "type": "static", "scale_bits": (4, 3), "four_over_six": True},
}
NVFP4_STATIC_ATTRS = {
    "num_bits": (2, 1),
    "block_sizes": {-1: BLOCK_SIZE, "type": "static", "scale_bits": (4, 3)},
}


def _weight_only_quant_cfg(attrs):
    return [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*weight_quantizer", "enable": True, "cfg": attrs},
    ]


class TestFourOverSixAlgorithm:
    """`four_over_six` is a registered calibrate mode whose search grid is derived."""

    def test_mode_is_registered(self):
        from modelopt.torch.quantization.mode import (
            BaseCalibrateModeDescriptor,
            CalibrateModeRegistry,
        )

        name = BaseCalibrateModeDescriptor._get_mode_name("four_over_six")
        assert name == "four_over_six_calibrate"
        assert name in CalibrateModeRegistry

    def test_config_does_not_expose_the_search_grid(self):
        """The whole point of the name: the multipliers cannot be retyped, so they cannot drift."""
        from modelopt.torch.quantization.config import FourOverSixCalibConfig

        fields = FourOverSixCalibConfig.model_fields
        assert not {"start_multiplier", "stop_multiplier", "step_size", "fp8_scale_sweep"} & set(
            fields
        )

    def test_delegates_to_mse_with_the_two_46_candidates(self, monkeypatch):
        """The grid handed to MSE is the legacy stanza, and it yields exactly {M=6, M=4}.

        Driven by what ``four_over_six_calibrate`` actually passes, not by literals, so it
        fails if the derived grid ever stops being the two 4/6 candidates.
        """
        import modelopt.torch.quantization.model_calib as mc
        from modelopt.torch.quantization.calib import MseCalibrator

        captured = {}
        monkeypatch.setattr(mc, "mse_calibrate", lambda *a, **kw: captured.update(kw))
        mc.four_over_six_calibrate(torch.nn.Linear(4, 4))

        grid = {k: captured[k] for k in ("step_size", "start_multiplier", "stop_multiplier")}
        assert grid == {k: LEGACY_FOUR_OVER_SIX_STANZA[k] for k in grid}
        assert captured["fp8_scale_sweep"] is False
        # 1.0 keeps the M=6 range; 1.5 == 6/4 is the M=4 range. No third candidate.
        cal = MseCalibrator(amax=torch.ones(1), **grid)
        assert cal._generate_candidates(torch.device("cpu")).tolist() == [1.0, E2M1_MAX / 4.0]


def _reference_static_fp4(inputs, amax, global_amax, quantize_block_scales, fp8_max, dtype, ptb):
    """Deterministic pure-torch stand-in for the Triton static-NVFP4 kernel.

    Not an E2M1 emulation -- only the Triton kernel is stubbed, and both arms of the
    comparison below use this same stand-in. What matters is that the error it produces
    depends on ``amax``, so the MSE search has something to discriminate between the
    M=6 and M=4 candidates.
    """
    flat = inputs.reshape(amax.numel(), -1)
    scale = (amax.reshape(-1, 1).to(flat.dtype) / E2M1_MAX).clamp(min=1e-12)
    return (torch.round(flat / scale).clamp(-E2M1_MAX, E2M1_MAX) * scale).reshape(inputs.shape)


class TestFourOverSixIsTheLegacyStanzaOnCPU:
    """`algorithm: four_over_six` calibrates bit-identically to the stanza it replaces.

    This is the acceptance gate for renaming the three shipped 4/6 recipes: the name has
    to be a name, not a behaviour change. Runs on CPU by stubbing the Triton-only static
    NVFP4 kernel; ``TestFourOverSixIsTheLegacyStanzaOnCUDA`` in ``tests/gpu`` runs the same
    comparison against the real kernel.
    """

    @staticmethod
    def _calibrated_weight_amax(monkeypatch, algorithm):
        import modelopt.torch.quantization.nn.modules.tensor_quantizer as tqm

        monkeypatch.setattr(tqm, "static_blockwise_fp4_fake_quant", _reference_static_fp4)

        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(4 * BLOCK_SIZE, 2 * BLOCK_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * BLOCK_SIZE, 4 * BLOCK_SIZE),
        )
        data = [torch.randn(4, 4 * BLOCK_SIZE) for _ in range(2)]
        mtq.quantize(
            model,
            {
                "quant_cfg": _weight_only_quant_cfg(NVFP4_FOUR_OVER_SIX_ATTRS),
                "algorithm": algorithm,
            },
            lambda m: [m(d) for d in data],
        )
        return {
            name: q.amax.clone()
            for name, q in model.named_modules()
            if getattr(q, "amax", None) is not None
        }

    def test_matches_the_legacy_stanza_bit_identically(self, monkeypatch):
        legacy = self._calibrated_weight_amax(monkeypatch, LEGACY_FOUR_OVER_SIX_STANZA)
        named = self._calibrated_weight_amax(monkeypatch, "four_over_six")

        assert set(legacy) == set(named)
        # Per-block amax, not per-tensor -- otherwise the comparison proves nothing about 4/6.
        assert legacy and all(t.numel() > 1 for t in legacy.values())
        for name in legacy:
            assert torch.equal(legacy[name], named[name]), name

    def test_the_grid_actually_matters(self, monkeypatch):
        """Guard against a vacuous pass: a different grid must give a different answer."""
        named = self._calibrated_weight_amax(monkeypatch, "four_over_six")
        default_mse = self._calibrated_weight_amax(monkeypatch, "mse")
        assert any(not torch.equal(named[n], default_mse[n]) for n in named)


class TestFourOverSixCoordination:
    """Neither half of 4/6 is useful alone, so a config carrying only one is rejected."""

    def test_flag_requires_static_nvfp4(self):
        with pytest.raises(ValidationError, match="only supported on static NVFP4"):
            QuantizerAttributeConfig(
                num_bits=(2, 1),
                block_sizes={
                    -1: BLOCK_SIZE,
                    "type": "dynamic",
                    "scale_bits": (4, 3),
                    "four_over_six": True,
                },
            )

    def test_flag_requires_e2m1(self):
        with pytest.raises(ValidationError, match="only supported on static NVFP4"):
            QuantizerAttributeConfig(
                num_bits=(4, 3),
                block_sizes={
                    -1: BLOCK_SIZE,
                    "type": "static",
                    "scale_bits": (4, 3),
                    "four_over_six": True,
                },
            )

    def test_flag_requires_e4m3_scale_bits(self):
        with pytest.raises(ValidationError, match="only supported on static NVFP4"):
            QuantizerAttributeConfig(
                num_bits=(2, 1),
                block_sizes={-1: BLOCK_SIZE, "type": "static", "four_over_six": True},
            )

    def test_flag_accepts_the_exmy_string_spelling(self):
        """Recipe YAML arrives as tuples, but the Python API keeps whatever was written."""
        cfg = QuantizerAttributeConfig(
            num_bits="e2m1",
            block_sizes={
                -1: BLOCK_SIZE,
                "type": "static",
                "scale_bits": "e4m3",
                "four_over_six": True,
            },
        )
        assert cfg.block_sizes["four_over_six"]

    def test_flag_without_a_weight_scale_search_is_rejected(self):
        """`max` never makes the M=6/M=4 choice, so the 256 normalization buys nothing."""
        with pytest.raises(ValidationError, match="never searches weight scales"):
            QuantizeConfig(
                quant_cfg=_weight_only_quant_cfg(NVFP4_FOUR_OVER_SIX_ATTRS), algorithm="max"
            )

    @pytest.mark.parametrize(
        "algorithm",
        ["four_over_six", "mse", "local_hessian", LEGACY_FOUR_OVER_SIX_STANZA, ["awq_lite", "mse"]],
    )
    def test_flag_with_a_weight_scale_search_is_accepted(self, algorithm):
        QuantizeConfig(
            quant_cfg=_weight_only_quant_cfg(NVFP4_FOUR_OVER_SIX_ATTRS), algorithm=algorithm
        )

    @pytest.mark.parametrize(
        "algorithm",
        [
            {"method": "nvfp4_act_headroom", "weight_scale_algorithm": {"method": "four_over_six"}},
            {"method": "nvfp4_act_headroom", "weight_scale_algorithm": {"method": "mse"}},
            {"method": "lsq", "scale_algorithm": {"method": "mse"}},
        ],
    )
    def test_a_bundled_weight_scale_search_satisfies_the_flag(self, algorithm):
        """`nvfp4_act_headroom` and `lsq` run their weight-scale search one level down."""
        QuantizeConfig(
            quant_cfg=_weight_only_quant_cfg(NVFP4_FOUR_OVER_SIX_ATTRS), algorithm=algorithm
        )

    def test_a_bundled_max_does_not_satisfy_the_flag(self):
        with pytest.raises(ValidationError, match="never searches weight scales"):
            QuantizeConfig(
                quant_cfg=_weight_only_quant_cfg(NVFP4_FOUR_OVER_SIX_ATTRS),
                algorithm={
                    "method": "nvfp4_act_headroom",
                    "weight_scale_algorithm": {"method": "max"},
                },
            )

    def test_a_bundled_four_over_six_still_requires_the_flag(self):
        with pytest.raises(ValidationError, match="no enabled quant_cfg entry sets"):
            QuantizeConfig(
                quant_cfg=_weight_only_quant_cfg(NVFP4_STATIC_ATTRS),
                algorithm={
                    "method": "nvfp4_act_headroom",
                    "weight_scale_algorithm": {"method": "four_over_six"},
                },
            )

    def test_flag_on_a_disabled_entry_does_not_constrain_the_algorithm(self):
        cfg = _weight_only_quant_cfg(NVFP4_FOUR_OVER_SIX_ATTRS)
        cfg[-1]["enable"] = False
        QuantizeConfig(quant_cfg=cfg, algorithm="max")

    def test_algorithm_without_the_flag_is_rejected(self):
        """Searching M=4 while the scales are normalized by 448 encodes those blocks wrongly."""
        with pytest.raises(ValidationError, match="no enabled quant_cfg entry sets"):
            QuantizeConfig(
                quant_cfg=_weight_only_quant_cfg(NVFP4_STATIC_ATTRS), algorithm="four_over_six"
            )

    def test_plain_mse_is_unaffected_by_either_rule(self):
        QuantizeConfig(quant_cfg=_weight_only_quant_cfg(NVFP4_STATIC_ATTRS), algorithm="mse")

    def test_shipped_preset_uses_the_named_algorithm(self):
        assert mtq.NVFP4_FOUR_OVER_SIX_CFG["algorithm"] == "four_over_six"
        QuantizeConfig(**mtq.NVFP4_FOUR_OVER_SIX_CFG)


class TestCompressRejectsFourOverSixUpFront:
    """`mtq.compress` refuses before touching any weight, not partway through the model."""

    def test_compress_names_every_offending_quantizer(self, monkeypatch):
        import modelopt.torch.quantization.nn.modules.tensor_quantizer as tqm

        monkeypatch.setattr(tqm, "static_blockwise_fp4_fake_quant", _reference_static_fp4)

        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(4 * BLOCK_SIZE, 4 * BLOCK_SIZE),
            torch.nn.Linear(4 * BLOCK_SIZE, 4 * BLOCK_SIZE),
        )
        mtq.quantize(
            model,
            {
                "quant_cfg": _weight_only_quant_cfg(NVFP4_FOUR_OVER_SIX_ATTRS),
                "algorithm": "four_over_six",
            },
            lambda m: m(torch.randn(4, 4 * BLOCK_SIZE)),
        )
        with pytest.raises(
            NotImplementedError, match="does not support the quantization format"
        ) as excinfo:
            mtq.compress(model)
        # Up front and complete: both quantizers are named, so the user sees the whole
        # problem rather than whichever layer compression happened to reach first.
        assert "0.weight_quantizer" in str(excinfo.value)
        assert "1.weight_quantizer" in str(excinfo.value)

    def test_excluding_the_46_layers_still_compresses(self, monkeypatch):
        """The refusal is about what would actually be packed, not what the model contains."""
        import modelopt.torch.quantization.nn.modules.tensor_quantizer as tqm

        monkeypatch.setattr(tqm, "static_blockwise_fp4_fake_quant", _reference_static_fp4)

        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(4 * BLOCK_SIZE, 4 * BLOCK_SIZE))
        mtq.quantize(
            model,
            {
                "quant_cfg": _weight_only_quant_cfg(NVFP4_FOUR_OVER_SIX_ATTRS),
                "algorithm": "four_over_six",
            },
            lambda m: m(torch.randn(4, 4 * BLOCK_SIZE)),
        )
        mtq.compress(model, {"compress": {"default": False}})
