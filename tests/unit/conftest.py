# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import contextlib
import os
import sys

import pytest

# Enforce no HuggingFace Hub network access for unit tests
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

with contextlib.suppress(ImportError):
    import huggingface_hub.constants as _hf_constants

    _hf_constants.HF_HUB_OFFLINE = True


@pytest.fixture(scope="session", autouse=True)
def _report_cpu_dispatch():
    """On Windows, record what torch decided the CPU can do.

    The job intermittently dies with 0xc000001d (STATUS_ILLEGAL_INSTRUCTION): a native module
    executing an opcode the host lacks. torch selects a vectorized kernel set at runtime, so what
    it chose -- compared against the CPU the workflow records before the run -- is the first thing
    to check. Reported from inside the test process because that is where the torch under test
    lives; the runner interpreter has only nox and uv.

    Windows-only and best-effort: elsewhere it is noise, and a diagnostic must never fail a run.
    """
    if sys.platform != "win32":
        return
    with contextlib.suppress(Exception):
        import torch

        print(
            f"\n[diag] torch {torch.__version__} "
            f"cpu_capability={torch.backends.cpu.get_cpu_capability()}",
            flush=True,
        )
