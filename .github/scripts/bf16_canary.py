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

"""Run the bf16 forward that 0xc000001d faults on, so a host can be tested without the suite.

Kept as a file rather than inlined into the workflow: a multi-line program inside a YAML `run:`
block has to fight the block scalar's indentation, and the first version of this canary silently
degraded to a bare `a @ b` that never reached oneDNN at all.
"""

import torch


def main() -> int:
    torch.set_grad_enabled(False)
    # The crashing test builds a 128-wide bf16 Linear; 512 is included because oneDNN selects a
    # kernel by shape as well as by ISA, and the small case may stay in a reference path.
    for n in (128, 512):
        layer = torch.nn.Linear(n, n, bias=False).eval().to(torch.bfloat16)
        x = torch.ones(8, n, dtype=torch.bfloat16)
        print(f"bf16 linear {n} ok {float(layer(x).sum())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
