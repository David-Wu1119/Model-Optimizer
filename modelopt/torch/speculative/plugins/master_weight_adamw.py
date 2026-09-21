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

"""AdamW with fp32 master weights, for models trained in a lower precision.

The DFlash draft trains alongside a frozen bf16 target and is cast to match it, so AdamW
allocates its moments with ``zeros_like(p)`` in bf16 -- and bf16 cannot represent the
updates Adam's second moment accumulates. At ``beta2=0.999`` a single step changes ``v`` by
at most 0.1%, while the smallest change bf16 can represent near ``v`` is about 0.4%, so
every decrease rounds back to the same number, ``v`` can only grow, and since the update is
divided by ``sqrt(v)`` the effective step size shrinks on its own from step 1, at any
learning rate.

This keeps the model in its training dtype and holds the extra precision in the optimizer
instead: an fp32 master copy of each low-precision parameter, fp32 moments, the AdamW
update applied to the master, and the result copied back down. Nothing has to reconcile
dtypes at forward time, because the model never holds a dtype the rest of it does not.
"""

import torch
from torch.optim.adamw import adamw
from transformers import TrainerCallback

__all__ = ["MasterWeightAdamW", "VerifyMasterWeightsCallback"]

# State keys this optimizer adds or owns, all of which must stay fp32 across a resume.
_FP32_STATE_KEYS = ("master", "exp_avg", "exp_avg_sq", "max_exp_avg_sq")


class MasterWeightAdamW(torch.optim.AdamW):
    """AdamW that keeps an fp32 master copy of every non-fp32 parameter.

    A drop-in replacement: it subclasses ``AdamW``, so param groups, weight decay and any
    LR scheduler behave unchanged. Parameters that are already fp32 are updated in place
    with no master copy, so a mixed-dtype model pays memory only where it buys precision.

    The master and both moments live in ``self.state[p]``, which is what ``state_dict()``
    serialises, so HF ``Trainer``'s ``optimizer.pt`` carries them across a resume.
    """

    @torch.no_grad()
    def step(self, closure=None):
        """Run one AdamW step in fp32 and write the result back at the parameter's dtype.

        This is ``AdamW.step`` with one substitution: the tensors handed to the functional
        update are the fp32 masters rather than the parameters. Everything else -- the
        group options, the lazy state init, the update itself -- is torch's.

        Swapping ``p.data`` to the master and calling ``super().step()`` would be shorter,
        but it is silently wrong under FSDP2: for a ``DTensor`` parameter the assignment
        updates the wrapper's reported dtype while the local shard stays in the model's, so
        ``zeros_like(p)`` allocates the moments in bf16 after all.
        """
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            if group.get("fused"):
                raise ValueError(
                    "fused AdamW writes through to the parameters it is given, which is not "
                    "compatible with fp32 master weights. Use foreach instead."
                )
            amsgrad = group.get("amsgrad", False)
            downcast, targets, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, steps = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                needs_master = p.dtype != torch.float32
                if not state:
                    if needs_master:
                        state["master"] = p.detach().float().clone()
                    zeros = state.get("master", p).detach().float()
                    state["exp_avg"] = torch.zeros_like(zeros)
                    state["exp_avg_sq"] = torch.zeros_like(zeros)
                    state["step"] = torch.zeros((), dtype=torch.float32)
                if amsgrad and "max_exp_avg_sq" not in state:
                    state["max_exp_avg_sq"] = torch.zeros_like(state["exp_avg"])
                target = state["master"] if needs_master else p
                if needs_master:
                    downcast.append((p, target))
                targets.append(target)
                grads.append(p.grad if p.grad.dtype == torch.float32 else p.grad.float())
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                if amsgrad:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                steps.append(state["step"])

            if not targets:
                continue

            beta1, beta2 = group["betas"]
            adamw(
                targets,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group.get("maximize", False),
                foreach=group.get("foreach"),
                capturable=group.get("capturable", False),
                differentiable=group.get("differentiable", False),
            )
            for param, master in downcast:
                param.copy_(master)

        return loss

    def load_state_dict(self, state_dict):
        """Restore, without letting the base class round the fp32 state to the parameters.

        ``Optimizer.load_state_dict`` casts every floating-point state tensor to its
        parameter's dtype. For a bf16 model that silently rounds the master copy and both
        moments on every resume -- the exact loss this optimizer exists to avoid, and
        invisible, since training continues and the loss keeps falling. The incoming
        ``state_dict`` still holds the saved fp32 tensors, so they are put back afterwards.
        """
        super().load_state_dict(state_dict)
        params = [p for group in self.param_groups for p in group["params"]]
        for param_id, saved in state_dict["state"].items():
            if not isinstance(param_id, int) or param_id >= len(params):
                continue
            param = params[param_id]
            for key in _FP32_STATE_KEYS:
                value = saved.get(key)
                if isinstance(value, torch.Tensor):
                    self.state[param][key] = value.detach().clone().to(device=param.device)


class VerifyMasterWeightsCallback(TrainerCallback):
    """Fail loudly when master weights were asked for but the optimizer does not keep them.

    Wiring the optimizer is the caller's job, so a training loop that builds its own
    ``AdamW`` gets bf16 moments and no error -- the feature is simply absent, and the only
    symptom is a drafter that trains a little worse. Checked after the first step, which is
    when the moments exist.
    """

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        """Check the optimizer's moment dtypes once, at the end of the first step."""
        if state.global_step != 1 or optimizer is None:
            return control
        inner = getattr(optimizer, "optimizer", optimizer)  # unwrap accelerate
        dtypes = {
            value.dtype
            for param_state in inner.state.values()
            for key, value in param_state.items()
            if key in _FP32_STATE_KEYS and isinstance(value, torch.Tensor)
        }
        if dtypes and dtypes != {torch.float32}:
            raise RuntimeError(
                f"dflash_fp32_master_weights is set, but the optimizer's Adam moments are "
                f"{dtypes} rather than fp32, so the flag is doing nothing. The training loop "
                f"has to build {MasterWeightAdamW.__name__}; see "
                f"examples/speculative_decoding/eagle_utils.py for how the shipped one does it."
            )
        return control
