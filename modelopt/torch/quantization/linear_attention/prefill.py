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

"""Batched differentiable GDN prefill with materialized numerical boundaries."""

import torch
import torch.nn.functional as F

from .config import LinearAttentionConfig
from .matmul import LinearAttentionMatmulSites, _validate_operand_quantizer
from .reference import _prepare, state_fp8_qdq_reference

__all__ = ["matmul_gdn"]


def matmul_gdn(
    q,
    k,
    v,
    g,
    beta,
    *,
    sites: LinearAttentionMatmulSites,
    policy: LinearAttentionConfig,
    w_quantizer,
    state_qdq=False,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    A_log=None,  # noqa: N803 - match the FLA kernel signature
    dt_bias=None,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    state_v_first=False,
    chunk_size=64,
    cp_context=None,
):
    """Execute all eight GDN prefill sites with autograd through their QDQ operators.

    This path batches chunk-local matmuls and carries states across chunks without
    detaching them. It uses FP32 working arithmetic (FP64 for double inputs) and
    casts outputs back to Q's dtype. It materializes the inverse and intermediates;
    it is a numerical-emulation backend, not the fused FLA implementation.
    """
    if policy.backend != "matmul" or chunk_size != policy.chunk_size:
        raise ValueError("matmul_gdn requires backend='matmul' and its configured chunk size")
    if cp_context is not None:
        raise NotImplementedError("GDN matmul emulation does not support context parallelism")
    sites.validate()
    _validate_operand_quantizer(w_quantizer, "gdn_w_quantizer")
    output_dtype = q.dtype
    dtype = torch.float64 if output_dtype == torch.float64 else torch.float32
    q, k, v, g, beta = (x.to(dtype) for x in (q, k, v, g, beta))
    if use_qk_l2norm_in_kernel:
        q, k = (x * (x.square().sum(-1, keepdim=True) + 1e-6).rsqrt() for x in (q, k))
    if use_gate_in_kernel:
        if A_log is None or dt_bias is None:
            raise ValueError("Fused GDN gate requires A_log and dt_bias")
        g = -A_log.to(dtype).exp() * F.softplus(g + dt_bias.to(dtype))
    if use_beta_sigmoid_in_kernel:
        beta = beta.sigmoid() * (2.0 if allow_neg_eigval else 1.0)
    if g.ndim != 3:
        raise ValueError("matmul_gdn requires scalar GDN log gates")
    initial_state = initial_state.to(dtype) if initial_state is not None else None
    boundaries = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    q, k, state, sequences = _prepare(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        boundaries,
        state_v_first,
    )
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    count = (max(end - start for _, start, end in sequences) + chunk_size - 1) // chunk_size
    padded_length = count * chunk_size
    heads = v.shape[2]

    def chunks(x):
        pieces = []
        for b, start, end in sequences:
            part = x[b, start:end]
            padding = part.new_zeros(padded_length - len(part), *part.shape[1:])
            pieces.append(torch.cat((part, padding)))
        packed = torch.stack(pieces)
        # [sequence, chunk, head, token, optional feature] -> batch matmuls over chunks.
        packed = packed.reshape(len(sequences), count, chunk_size, heads, *x.shape[3:])
        return packed.transpose(2, 3).reshape(-1, heads, chunk_size, *x.shape[3:])

    def arithmetic(name, x):
        target = policy.elementwise.get(name)
        if target is None:
            return x
        rounded = x.to(getattr(torch, target)).to(dtype)
        return x + (rounded - x).detach()

    def mm(name, lhs, rhs):
        return sites.matmul(name, lhs, rhs, policy, w_quantizer=w_quantizer)

    qc, kc, vc = (chunks(x) for x in (q, k, v))
    gc = arithmetic("gate_prefix", chunks(g).cumsum(-1))
    bc = chunks(beta).unsqueeze(-1)
    causal = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).tril()
    relative = (gc.unsqueeze(-1) - gc.unsqueeze(-2)).masked_fill(~causal, 0)
    decay = arithmetic("gate_exp", relative.exp())
    gate = arithmetic("gate_exp", gc.exp()).unsqueeze(-1)
    lower = (mm("key_interaction", bc * kc, kc) * decay).tril(-1)
    identity = torch.eye(chunk_size, device=q.device, dtype=dtype).expand_as(lower)
    inverse = torch.linalg.solve_triangular(
        identity + lower, identity, upper=False, unitriangular=True
    )
    u = mm("wy_value", inverse, (bc * vc).transpose(-1, -2))
    w = mm("wy_key", inverse, (bc * kc * gate).transpose(-1, -2))
    scores = (mm("output_score", qc * scale, kc) * decay).tril()
    weighted_keys = kc * arithmetic("gate_exp", (gc[..., -1:] - gc).exp()).unsqueeze(-1)

    def unbatch(x):
        return x.reshape(len(sequences), count, *x.shape[1:])

    qc, gate, gc, u, w, scores, weighted_keys = map(
        unbatch, (qc, gate, gc, u, w, scores, weighted_keys)
    )
    if state_qdq:
        state = state_fp8_qdq_reference(state, policy.state.block_v)
    outputs = []
    for c in range(count):
        # The legacy W handle is invoked once per state read and is never rerun in backward.
        updated = arithmetic(
            "value_residual", u[:, c] - mm("state_read", w[:, c], state.transpose(-1, -2))
        )
        memory = mm("output_state", qc[:, c] * scale * gate[:, c], state.transpose(-1, -2))
        local = mm("output_value", scores[:, c], updated.transpose(-1, -2))
        outputs.append(arithmetic("output_add", memory + local))
        decayed = arithmetic(
            "state_decay", state * arithmetic("gate_exp", gc[:, c, :, -1].exp())[..., None, None]
        )
        update = mm(
            "state_update", weighted_keys[:, c].transpose(-1, -2), updated.transpose(-1, -2)
        )
        next_state = arithmetic("state_add", decayed + update)
        if state_qdq:
            next_state = state_fp8_qdq_reference(next_state, policy.state.block_v)
        # Padded chunks must not introduce extra state writes for shorter packed sequences.
        active = torch.tensor(
            [c * chunk_size < end - start for _, start, end in sequences], device=q.device
        )
        state = torch.where(active[:, None, None, None], next_state, state)
    output = torch.cat(outputs, dim=2).transpose(1, 2)
    pieces = [output[n, : end - start] for n, (_, start, end) in enumerate(sequences)]
    output = torch.stack(pieces) if boundaries is None else torch.cat(pieces).unsqueeze(0)
    final = state.transpose(-1, -2) if state_v_first else state
    return output.to(output_dtype), final if output_final_state else None
