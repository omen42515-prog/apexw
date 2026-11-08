from setuptools import setup, find_packages

setup(
    name="apexw",
    version="1.2.0",
    description="ApexW: Memory-Efficient Fused Optimizer for PyTorch — Cross-Hardware (CUDA, ROCm, Apple MPS, Google TPU)",
    author="ApexW Contributors",
    packages=find_packages(exclude=["tests*", "benchmarks*"]),
    include_package_data=True,
    package_data={
        "apexw": [
            "csrc/*.cpp",
            "csrc/*.cu",
            "csrc/*.h",
            "csrc/*.cuh",
        ],
    },
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
