import os
import sys
import sysconfig
import shutil
from setuptools import setup, find_packages

setup(
    name="flash_rwkv",
    version="0.2.1",
    packages=find_packages(),
    package_data={
        'flash_rwkv': ['rwkv5/*.cu', 'rwkv5/*.cpp', 'rwkv6/*.cu', 'rwkv6/*.cpp'],
    },
    ext_modules=[],
    install_requires=[
        "torch>=1.13.0"
    ]
)
