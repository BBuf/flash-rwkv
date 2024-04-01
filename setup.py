import os
import sys
import sysconfig
import shutil
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, load

def compile_rwkv5_cuda_kernel():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_dir = os.path.join(current_dir, "rwkv5_attention", "kernels")
    os.makedirs(kernel_dir, exist_ok=True)
    cuda_sources = [
        os.path.join(current_dir, "flash_rwkv/rwkv5", "wkv5_op.cpp"),
        os.path.join(current_dir, "flash_rwkv/rwkv5", "wkv5_kernel.cu"),
    ]

    extension = load(
        name="flash_rwkv_wkv5_cuda",
        sources=cuda_sources,
        extra_cflags=['-O3'],
        extra_cuda_cflags=["-O3", "-res-usage", "--maxrregcount=60", "--use_fast_math", "-Xptxas=-O3", "--extra-device-vectorization", "-D_N_=64"],
        verbose=True
    )

    so_file_name = extension.__file__

    shutil.copy(so_file_name, kernel_dir)
    print(f"Copied {so_file_name} to {kernel_dir}")

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
