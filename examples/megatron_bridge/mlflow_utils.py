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

"""MLflow tracking for the Megatron-Bridge examples, mirroring ``examples/hf_ptq``.

The three scripts that write a checkpoint share one CLI and one experiment-name convention,
so a PTQ run, the QAD run that refines its checkpoint and the export that deploys it can be
found together on one server. How the run is opened differs:

``quantize.py`` and ``export_quantized_megatron_to_hf.py`` each produce a checkpoint in a
single pass, so they open their own run through
:func:`~modelopt.torch.utils.mlflow.track_run`.

``distill.py`` is a training loop, and Megatron-Bridge already logs to MLflow from inside it
(``LoggerConfig``) -- per-iteration metrics and the full resolved config, neither of which a
wrapper around ``main()`` can see. So it hands these settings to Megatron-Bridge rather than
opening a competing run, and adds only the provenance pointer, which Megatron-Bridge does
not write.

Every rank parses and validates the same flags, so a typo in the URI fails identically
everywhere instead of on one rank while the others wait in a collective.

Nothing here imports Megatron, so the tracking can be exercised without it.
"""

import argparse
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils.mlflow import (
    MlflowRunLogger,
    checkpoint_run_tags,
    log_active_run_experiment_json,
    resolved_recipe_texts,
    track_run,
)
from modelopt.torch.utils.mlflow import add_mlflow_args as _add_mlflow_args
from modelopt.torch.utils.mlflow import resolve_mlflow_args as _resolve_mlflow_args

# The tracking settings describe the destination rather than the work, and
# checkpoint_exported is the scripts' own bookkeeping.
_NON_PARAM_ARGS = frozenset(
    {
        "checkpoint_exported",
        "mlflow",
        "mlflow_experiment",
        "mlflow_log_checkpoints",
        "mlflow_required",
        "mlflow_run_name",
    }
)


def _name(path: str) -> str:
    return Path(path.rstrip("/")).name


@dataclass(frozen=True)
class Tool:
    """What distinguishes one script's tracking from its siblings'.

    *model* and *checkpoint* read the arguments naming this run's input and output, which is
    all the shared join keys need; *variant* names what the run did to that model, for the
    default experiment name.
    """

    name: str
    tracks: str
    variant_help: str
    variant: Callable[[argparse.Namespace], str]
    model: Callable[[argparse.Namespace], str]
    checkpoint: Callable[[argparse.Namespace], str]
    outputs: Callable[[argparse.Namespace], dict[str, Path]] = field(default=lambda args: {})


QUANTIZE = Tool(
    name="megatron_bridge_quantize",
    tracks=(
        "Track this run on an MLflow server (e.g. https://<your-mlflow-server>/), uploading "
        "the command, the resolved recipe, the run log and the quantizer summary, and "
        "writing .experiment.json into --export_megatron_path so the checkpoint names the "
        "run that produced it."
    ),
    variant_help="recipe name, or --quant_cfg if no --recipe",
    # ``or "none"``: neither flag is required by the parser, and a run that reaches
    # get_quant_config without one fails there rather than while being named.
    variant=lambda args: Path(args.recipe).stem if args.recipe else (args.quant_cfg or "none"),
    model=lambda args: args.hf_model_name_or_path,
    checkpoint=lambda args: args.export_megatron_path,
    outputs=lambda args: {
        "summary/quant_summary.txt": Path(args.export_megatron_path) / ".quant_summary.txt"
    },
)

EXPORT = Tool(
    name="megatron_bridge_export",
    tracks=(
        "Track this export on an MLflow server (e.g. https://<your-mlflow-server>/), "
        "uploading the command and the run log, and writing .experiment.json into "
        "--export_unified_hf_path so the deployable checkpoint names the run that wrote it."
    ),
    variant_help="the Megatron checkpoint's directory name",
    variant=lambda args: _name(args.megatron_path),
    model=lambda args: args.hf_model_name_or_path,
    checkpoint=lambda args: args.export_unified_hf_path,
)

DISTILL = Tool(
    name="megatron_bridge_distill",
    tracks=(
        "Track this run on an MLflow server (e.g. https://<your-mlflow-server>/) through "
        "Megatron-Bridge's own MLflow logging, which records the training metrics and the "
        "resolved config, and write .experiment.json into <output_dir>/checkpoints so the "
        "distilled checkpoint names the run that produced it."
    ),
    variant_help="the student Megatron checkpoint's directory name",
    variant=lambda args: (
        _name(args.student_megatron_path) if args.student_megatron_path else "bf16"
    ),
    model=lambda args: args.student_hf_path,
    checkpoint=lambda args: str(Path(args.output_dir) / "checkpoints"),
)


def add_mlflow_args(parser: argparse.ArgumentParser, tool: Tool) -> None:
    """Add the MLflow tracking flags for *tool*."""
    _add_mlflow_args(parser, tool.name, tracks=tool.tracks, variant_help=tool.variant_help)
    if tool is DISTILL:
        parser.add_argument(
            "--mlflow_log_checkpoints",
            action="store_true",
            help=(
                "Upload every saved checkpoint to the MLflow server as an artifact. Off by "
                "default, unlike Megatron-Bridge's own setting: a QAD checkpoint is tens to "
                "hundreds of GB, sent over HTTP on every save."
            ),
        )


def resolve_mlflow_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser, tool: Tool
) -> None:
    """Settle where tracking is configured from, and name the experiment."""
    _resolve_mlflow_args(
        args, parser, tool=tool.name, model=tool.model(args), variant=tool.variant(args)
    )


def _run_inputs(args: argparse.Namespace, tool: Tool) -> tuple[dict, dict]:
    """Params and start-time artifacts describing this run."""
    params = {k: v for k, v in vars(args).items() if k not in _NON_PARAM_ARGS}
    # The parallelism flags say how the run was laid out but not how many GPUs it took:
    # data parallelism is implicit in the launcher's world size.
    params["world_size"] = dist.size()
    return params, resolved_recipe_texts(getattr(args, "recipe", None))


def _describe(args: argparse.Namespace, tool: Tool) -> dict:
    """Everything the run uploads, gathered once -- reading the recipe twice would print a
    second "[load_recipe] loading:" line on every tracked run."""
    params, texts = _run_inputs(args, tool)
    return {
        "params": params,
        "tags": checkpoint_run_tags(tool.model(args), tool.checkpoint(args)),
        "texts": texts,
        "files": tool.outputs(args),
    }


@contextmanager
def mlflow_run(args: argparse.Namespace, tool: Tool) -> Iterator[None]:
    """Track this invocation for the duration of the block; see
    :func:`~modelopt.torch.utils.mlflow.track_run`.

    For the single-pass scripts. ``distill.py`` uses :func:`logger_kwargs` instead, so it
    never opens a run Megatron-Bridge would then compete with.
    """
    logger = MlflowRunLogger(
        args.mlflow or "",
        args.mlflow_experiment,
        run_name=args.mlflow_run_name,
        enabled=bool(args.mlflow) and dist.is_master(),
        required=args.mlflow_required,
    )
    with track_run(
        logger,
        tool.checkpoint(args),
        is_main=dist.is_master(),
        exported=lambda: args.checkpoint_exported,
        describe=lambda: _describe(args, tool),
    ):
        yield


def logger_kwargs(args: argparse.Namespace, tool: Tool = DISTILL) -> dict:
    """The MLflow half of Megatron-Bridge's ``LoggerConfig``, from these flags.

    Empty unless tracking was requested, so an untracked run hands Megatron-Bridge nothing:
    the fields below landed in ``LoggerConfig`` in Megatron-Bridge 0.6, and passing them
    unconditionally would break an untracked run on an older one for no reason. A tracked
    run on an older one still fails loudly, naming the field.
    """
    if not args.mlflow:
        return {}
    return {
        "mlflow_tracking_uri": args.mlflow,
        "mlflow_experiment": args.mlflow_experiment,
        "mlflow_run_name": args.mlflow_run_name,
        "mlflow_tags": checkpoint_run_tags(tool.model(args), tool.checkpoint(args)),
        "mlflow_log_artifacts": args.mlflow_log_checkpoints,
    }


def record_checkpoint_provenance(args: argparse.Namespace, tool: Tool = DISTILL) -> None:
    """Point the trained checkpoint at the run Megatron-Bridge opened for it.

    Called once training has written the checkpoint, from the rank that owns that run --
    Megatron-Bridge opens it on the *last* rank, not the first.
    """
    if not args.mlflow or not dist.is_last_process():
        return
    log_active_run_experiment_json(tool.checkpoint(args))
