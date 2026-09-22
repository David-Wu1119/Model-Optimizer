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

"""MLflow tracking for ``quantize.py``, mirroring ``examples/hf_ptq``.

Every rank parses and validates the same flags, so a typo in the URI fails identically
everywhere instead of on one rank while the others wait in a collective. Only the master rank
opens a run, so the log capture and the uploads happen once.

Nothing here imports Megatron, so the tracking can be exercised without it.
"""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils.mlflow import (
    MlflowRunLogger,
    checkpoint_run_tags,
    resolved_recipe_texts,
    track_run,
)
from modelopt.torch.utils.mlflow import add_mlflow_args as _add_mlflow_args
from modelopt.torch.utils.mlflow import resolve_mlflow_args as _resolve_mlflow_args

TOOL_NAME = "megatron_bridge_quantize"

# The tracking settings describe the destination rather than the quantization, and
# checkpoint_exported is this script's own bookkeeping.
_NON_PARAM_ARGS = frozenset(
    {
        "checkpoint_exported",
        "mlflow",
        "mlflow_experiment",
        "mlflow_required",
        "mlflow_run_name",
    }
)


def add_mlflow_args(parser: argparse.ArgumentParser) -> None:
    """Add the MLflow tracking flags."""
    _add_mlflow_args(
        parser,
        TOOL_NAME,
        tracks=(
            "Track this run on an MLflow server (e.g. https://<your-mlflow-server>/), "
            "uploading the command, the resolved recipe, the run log and the quantizer "
            "summary, and writing .experiment.json into --export_megatron_path so the "
            "checkpoint names the run that produced it."
        ),
        variant_help="recipe name, or --quant_cfg if no --recipe",
    )


def resolve_mlflow_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Settle where tracking is configured from, and name the experiment."""
    _resolve_mlflow_args(
        args,
        parser,
        tool=TOOL_NAME,
        model=args.hf_model_name_or_path,
        # ``or "none"``: neither flag is required by the parser, and the run that reaches
        # get_quant_config without one fails there rather than while being named.
        variant=Path(args.recipe).stem if args.recipe else (args.quant_cfg or "none"),
    )


def _run_inputs(args: argparse.Namespace) -> tuple[dict, dict]:
    """Params and start-time artifacts describing this PTQ run."""
    params = {k: v for k, v in vars(args).items() if k not in _NON_PARAM_ARGS}
    # The parallelism flags say how the run was laid out but not how many GPUs it took:
    # data parallelism is implicit in the launcher's world size.
    params["world_size"] = dist.size()
    return params, resolved_recipe_texts(args.recipe)


def _run_tags(args: argparse.Namespace) -> dict[str, str]:
    """This run's shared join keys, from the arguments that name its input and output."""
    return checkpoint_run_tags(args.hf_model_name_or_path, args.export_megatron_path)


def _run_outputs(args: argparse.Namespace) -> dict[str, Path]:
    """Summaries written beside the checkpoint, keyed by artifact path.

    Uploaded without the leading dot, which is awkward to browse in the MLflow UI. A missing
    entry is skipped: the summary is written by the master rank only once quantization
    has finished.
    """
    return {"summary/quant_summary.txt": Path(args.export_megatron_path) / ".quant_summary.txt"}


def _describe(args: argparse.Namespace) -> dict:
    """Everything the run uploads, gathered once -- reading the recipe twice would print a
    second "[load_recipe] loading:" line on every tracked run."""
    params, texts = _run_inputs(args)
    return {
        "params": params,
        "tags": _run_tags(args),
        "texts": texts,
        "files": _run_outputs(args),
    }


@contextmanager
def mlflow_run(args: argparse.Namespace) -> Iterator[None]:
    """Track this invocation for the duration of the block; see
    :func:`~modelopt.torch.utils.mlflow.track_run`."""
    logger = MlflowRunLogger(
        args.mlflow or "",
        args.mlflow_experiment,
        run_name=args.mlflow_run_name,
        enabled=bool(args.mlflow) and dist.is_master(),
        required=args.mlflow_required,
    )
    with track_run(
        logger,
        args.export_megatron_path,
        is_main=dist.is_master(),
        exported=lambda: args.checkpoint_exported,
        describe=lambda: _describe(args),
    ):
        yield
