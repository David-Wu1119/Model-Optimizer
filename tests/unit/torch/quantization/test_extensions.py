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

import pytest

import modelopt.torch.quantization.extensions as extensions


def test_ggml_extension_caches_strict_build_failure(monkeypatch):
    calls = []

    def getter():
        raise AssertionError("the helper uses the getter only as a cache namespace")

    def failing_loader(**kwargs):
        calls.append(kwargs["raise_if_failed"])
        if kwargs["raise_if_failed"]:
            raise RuntimeError("compiler failed")

    monkeypatch.setattr(extensions, "load_cpp_extension", failing_loader)

    assert extensions._get_ggml_ext(getter, "test", "TEST", False) is None
    with pytest.raises(RuntimeError, match="compiler failed"):
        extensions._get_ggml_ext(getter, "test", "TEST", True)
    with pytest.raises(RuntimeError, match="compiler failed"):
        extensions._get_ggml_ext(getter, "test", "TEST", True)
    assert extensions._get_ggml_ext(getter, "test", "TEST", False) is None

    assert calls == [False, True]
