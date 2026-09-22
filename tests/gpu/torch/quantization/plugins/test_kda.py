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

import copy

import pytest
import torch

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.linear_attention import LinearAttentionConfig
from modelopt.torch.quantization.nn import TensorQuantizer

KimiDeltaAttention = pytest.importorskip("fla.layers.kda").KimiDeltaAttention


def _layer():
    return (
        KimiDeltaAttention(hidden_size=32, head_dim=16, num_heads=2, use_short_conv=True)
        .cuda()
        .train()
    )


def _forward(model, hidden):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(hidden)[0]


@pytest.mark.parametrize(
    "mode", ["state", "w", "prefill_fp8", "prefill_nvfp4", "arithmetic", "solve"]
)
@pytest.mark.timeout(180)
def test_fla_layer_qat_restore_and_optimizer(tmp_path, mode):
    torch.manual_seed(73)
    model = _layer()
    hidden = torch.randn(1, 73, 32, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        baseline = _forward(model, hidden)
    attributes = {"num_bits": (4, 3), "type": "dynamic", "axis": (0, 1, 2)}
    if mode == "prefill_nvfp4":
        attributes = {
            "num_bits": (2, 1),
            "type": "dynamic",
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        }
    cfg = {
        "quant_cfg": [{"quantizer_name": "*", "enable": False}],
        "algorithm": None,
        "linear_attention": [{"module_name": "*", "cfg": {"backend": "matmul"}}],
    }
    if mode in ("w", "prefill_fp8", "prefill_nvfp4"):
        cfg["quant_cfg"].append({"quantizer_name": "*kda_w_quantizer", "cfg": attributes})
    if mode.startswith("prefill"):
        cfg["quant_cfg"].append({"quantizer_name": "*linear_attn_sites.*", "cfg": attributes})
    if mode == "state":
        cfg["quant_cfg"].append(
            {"quantizer_name": "*kda_state_quantizer", "cfg": {**attributes, "axis": (0, 1)}}
        )
    if mode == "arithmetic":
        cfg["linear_attention"][0]["cfg"]["elementwise"] = {"value_residual": "bfloat16"}
    if mode == "solve":
        cfg["linear_attention"][0]["cfg"]["solve"] = {
            "method": "neumann",
            "degree": 3,
            "implementation": "triton",
        }
    mtq.quantize(model, cfg)
    output = _forward(model, hidden)
    assert torch.isfinite(output).all()
    enabled = [q for q in model.modules() if isinstance(q, TensorQuantizer) and q.is_enabled]
    policy = model.linear_attention_config
    for q in enabled:
        q.disable()
    model.linear_attention_config = LinearAttentionConfig()
    with torch.no_grad():
        torch.testing.assert_close(_forward(model, hidden), baseline, rtol=0, atol=0)
    for q in enabled:
        q.enable()
    model.linear_attention_config = policy
    checkpoint = tmp_path / "model.pth"
    mto.save(model, checkpoint)
    restored = mto.restore(_layer(), checkpoint)
    actual = _forward(restored, hidden)
    torch.testing.assert_close(actual, output, rtol=0, atol=0)
    assert restored.linear_attention_config == policy
    actual.float().square().mean().backward()
    assert restored.q_proj.weight.grad is not None
    for parameter in restored.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    if mode == "prefill_fp8":
        copied = copy.deepcopy(restored)
        calls = []
        hook = copied.kda_w_quantizer.register_forward_hook(lambda *args: calls.append(True))
        with torch.no_grad():
            torch.testing.assert_close(_forward(copied, hidden), actual, rtol=0, atol=0)
        hook.remove()
        assert calls, "a copied layer must use its own quantizer handles"
    before = restored.q_proj.weight.detach().clone()
    torch.optim.SGD(restored.parameters(), lr=0.1).step()
    assert not torch.equal(before, restored.q_proj.weight)
    restored.eval()
    with pytest.raises(NotImplementedError, match="chunk path"):
        _forward(restored, hidden[:, :1])
