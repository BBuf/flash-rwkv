import os
import sys
import sysconfig
import shutil
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def compile_rwkv5_cuda_kernel():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_dir = os.path.join(current_dir, "rwkv5_attention", "kernels")
    os.makedirs(kernel_dir, exist_ok=True)
    cuda_sources = [
        os.path.join(current_dir, "rwkv5", "wkv5_op.cpp"),
        os.path.join(current_dir, "rwkv5", "wkv5_kernel.cu")
    ]

    cuda_extension = CUDAExtension(
        name="flash_rwkv.wkv5_cuda",
        sources=cuda_sources,
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-O3",
                "-res-usage",
                "--maxrregcount=60",
                "--use_fast_math",
                "-Xptxas=-O3",
                "--extra-device-vectorization",
                "-D_N_=64"
            ]
        }
    )

    cuda_extension.build(cuda_extension)
    shutil.copy(cuda_extension.get_lib_name(), kernel_dir)


setup(
    name="flash_rwkv",
    version="0.1",
    packages=["flash_rwkv"],
    package_data={
        "flash_rwkv": ["kernels/*.so"]
    },
    cmdclass={
        "build_ext": BuildExtension
    },
    ext_modules=[],
    install_requires=[
        "torch>=1.13.0"
    ]
)

if __name__ == "__main__":
    compile_rwkv5_cuda_kernel()
    sys.exit(0)
