import os
import sys
import sysconfig
import shutil
from setuptools import setup

setup(
    name="flash_rwkv",
    version="0.1",
    packages=["flash_rwkv"],
    ext_modules=[],
    install_requires=[
        "torch>=1.13.0"
    ]
)
