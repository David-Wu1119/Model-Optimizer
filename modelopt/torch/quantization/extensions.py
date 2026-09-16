# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Module to load C++ / CUDA extensions."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from modelopt.torch.utils import load_cpp_extension

__all__ = [
    "get_cuda_ext",
    "get_cuda_ext_fp8",
    "get_cuda_ext_iq1_s",
    "get_cuda_ext_iq2_xs",
    "get_cuda_ext_mx",
    "precompile",
]

path = Path(__file__).parent
kernels_gemm = path.parent / "kernels" / "quantization" / "gemm"
kernels_ggml = path.parent / "kernels" / "quantization" / "ggml"
_NOT_BUILT = object()


def get_cuda_ext(raise_if_failed: bool = False):
    """Returns the cuda extension for tensor_quant."""
    if not hasattr(get_cuda_ext, "extension"):
        get_cuda_ext.extension = load_cpp_extension(  # type:ignore[attr-defined]
            name="modelopt_cuda_ext",
            sources=[kernels_gemm / "tensor_quant.cpp", kernels_gemm / "tensor_quant_gpu.cu"],
            cuda_version_specifiers=">=11",
            raise_if_failed=raise_if_failed,
        )
    return get_cuda_ext.extension  # type:ignore[attr-defined]


def get_cuda_ext_fp8(raise_if_failed: bool = False):
    """Returns the cuda extension for tensor_quant_fp8."""
    if not hasattr(get_cuda_ext_fp8, "extension"):
        get_cuda_ext_fp8.extension = load_cpp_extension(  # type:ignore[attr-defined]
            name="modelopt_cuda_ext_fp8",
            sources=[kernels_gemm / "tensor_quant_gpu_fp8.cu"],
            cuda_version_specifiers=">=11.8",
            fail_msg=(
                "CUDA extension for FP8 quantization could not be built and loaded, FP8 simulated"
                " quantization will not be available."
            ),
            raise_if_failed=raise_if_failed,
        )
    return get_cuda_ext_fp8.extension  # type:ignore[attr-defined]


def get_cuda_ext_mx(raise_if_failed: bool = False):
    """Returns the cuda extension for tensor_quant_mx."""
    if not hasattr(get_cuda_ext_mx, "extension"):
        get_cuda_ext_mx.extension = load_cpp_extension(  # type:ignore[attr-defined]
            name="modelopt_cuda_ext_mx",
            sources=[
                kernels_gemm / "tensor_quant_mx.cu",
            ],
            cuda_version_specifiers=">=11.8",
            fail_msg=(
                "CUDA extension for MX quantization could not be built and loaded, MX simulated"
                " quantization will not be available."
            ),
            extra_cuda_cflags=["--use_fast_math"],
            raise_if_failed=raise_if_failed,
        )
    return get_cuda_ext_mx.extension  # type:ignore[attr-defined]


def _get_ggml_ext(getter: Callable[..., Any], stem: str, label: str, raise_if_failed: bool):
    # A strict caller may follow an optional build that cached ``None``; retry so it fails loudly.
    extension = getattr(getter, "extension", _NOT_BUILT)
    if extension is _NOT_BUILT or (raise_if_failed and extension is None):
        extension = load_cpp_extension(
            name=f"modelopt_cuda_ext_{stem}",
            sources=[kernels_ggml / f"{stem}.cpp", kernels_ggml / f"{stem}.cu"],
            cuda_version_specifiers=">=11.8",
            fail_msg=f"{label} CUDA packing is unavailable; using the PyTorch reference encoder.",
            extra_cuda_cflags=["-O3"],
            raise_if_failed=raise_if_failed,
        )
        setattr(getter, "extension", extension)
    return extension


def get_cuda_ext_iq1_s(raise_if_failed: bool = False):
    """Return the GGML-compatible IQ1_S packing extension."""
    return _get_ggml_ext(get_cuda_ext_iq1_s, "iq1_s", "IQ1_S", raise_if_failed)


def get_cuda_ext_iq2_xs(raise_if_failed: bool = False):
    """Return the GGML-compatible IQ2_XS packing extension."""
    return _get_ggml_ext(get_cuda_ext_iq2_xs, "iq2_xs", "IQ2_XS", raise_if_failed)


def __getattr__(name):
    # Bare extension attributes are legacy compatibility aliases; new extensions expose getters.
    if name == "cuda_ext":
        return get_cuda_ext()
    elif name == "cuda_ext_fp8":
        return get_cuda_ext_fp8()
    elif name == "cuda_ext_mx":
        return get_cuda_ext_mx()
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")


def precompile():
    """Precompile the CUDA extensions."""
    print(get_cuda_ext())
    print(get_cuda_ext_fp8())
    print(get_cuda_ext_mx())
    print(get_cuda_ext_iq1_s())
    print(get_cuda_ext_iq2_xs())
