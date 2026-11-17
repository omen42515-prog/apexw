import os
from setuptools import setup, find_packages

cmdclass = {}
ext_modules = []

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

    cuda_available = torch.cuda.is_available() or (CUDA_HOME is not None and os.path.exists(CUDA_HOME))
    force_cuda = os.environ.get("FORCE_CUDA", "0") == "1"
    disable_cuda = os.environ.get("DISABLE_CUDA_EXT", "0") == "1"

    if (cuda_available or force_cuda) and not disable_cuda:
        csrc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apexw", "csrc")
        cpp_src = os.path.join(csrc_dir, "apexw_cuda.cpp")
        cu_src = os.path.join(csrc_dir, "apexw_cuda_kernel.cu")
        if os.path.exists(cpp_src) and os.path.exists(cu_src):
            ext_modules.append(
                CUDAExtension(
                    name="apexw.apexw_cuda_ext",
                    sources=[
                        "apexw/csrc/apexw_cuda.cpp",
                        "apexw/csrc/apexw_cuda_kernel.cu",
                    ],
                    extra_compile_args={
                        "cxx": ["-O3"],
                        "nvcc": [
                            "-O3",
                            "-U__CUDA_NO_HALF_OPERATORS__",
                            "-U__CUDA_NO_HALF_CONVERSIONS__",
                            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                            "--expt-relaxed-constexpr",
                        ],
                    },
                )
            )
            cmdclass["build_ext"] = BuildExtension
except Exception:
    pass

setup(
    name="apexw",
    version="1.3.0",
    description="Memory-efficient fused AdamW and Muon optimizers for PyTorch",
    author="ApexW Contributors",
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
    package_data={
        "apexw": [
            "csrc/*.cpp",
            "csrc/*.cu",
            "csrc/*.h",
            "csrc/*.cuh",
        ],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=[
        "torch>=2.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
)
