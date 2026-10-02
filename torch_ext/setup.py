# SPDX-License-Identifier: MIT
#
# Build for the barlink_sm86 torch extension.
#
#   python setup.py build_ext --inplace    # produces barlink_sm86/_C*.so
#
# Architecture list comes from TORCH_CUDA_ARCH_LIST (e.g. "8.6" for the
# RTX 3080); when unset, torch auto-detects from the visible GPUs.

import os
from setuptools import setup

from torch.utils.cpp_extension import CUDAExtension, BuildExtension

HERE = os.path.dirname(os.path.abspath(__file__))

setup(
    name="barlink-sm86",
    version="0.1.0",
    description="Dual-GPU BAR1 P2P write primitives for torch (route B)",
    ext_modules=[
        CUDAExtension(
            name="barlink_sm86._C",
            sources=[
                os.path.join(HERE, "barlink_sm86", "core.cu"),
                os.path.join(HERE, "barlink_sm86", "binding.cpp"),
            ],
            include_dirs=[os.path.join(HERE, "barlink_sm86")],
            extra_compile_args={
                # torch >= 2.14 headers use C++20 concepts (e.g.
                # at::symint::sizes in ATen/ExpandUtils.h). A user-supplied
                # -std=c++17 overrides torch's injected -std=c++20 and
                # silently mangles those template declarations ("expected
                # primary-expression before '>' token"). core.cu has no
                # torch headers and stays on c++14.
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": ["-O3", "-std=c++14"],
            },
            libraries=["cuda"],  # driver API
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
