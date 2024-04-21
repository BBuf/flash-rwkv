## FlashRWKV

### Motivation

There are several reasons for creating the standalone FlashRWKV package:

- [During the support for the RWKV5 model in the transformers](https://github.com/huggingface/transformers/pull/29095)，[ArthurZucker](https://github.com/ArthurZucker)suggested that RWKV5's custom CUDA kernel should be implemented independently. This way, there's no need to compile and install the CUDA kernel within the transformers library itself.
- When implementing custom RWKV5 and RWKV6 models within the Hugging Face community, I found that `.cu` and `.cpp` files uploaded to the Hugging Face model repository could not be automatically downloaded when using `trust_remote_code=True`. To compile and use cuda kernels, it was necessary to manually copy the missing files, which was very cumbersome. And I found this problem was also present with Qwen's chat model and so on.
- The [rwkv package](https://github.com/BlinkDL/ChatRWKV/tree/main/rwkv_pip_package ) provided in the official ChatRWKV library can compile kernels, but it is quite difficult to use externally. For example, it is not feasible to use that package when implementing RWKV5 on Hugging Face, and the compilation of the rwkv package in the ChatRWKV library does not support backward propagation. In contrast, using the FlashRWKV package allows for a completely independent and convenient extension to any model implementation such as RWKV5/RWKV6, and it supports backward propagation.
- Use the state-of-the-art RWKV CUDA kernel on Nvidia GPUs to accelerate fine-tuning training and inference, enhancing model efficiency.
- Previously, there was no experience in using Torch's C++ extension module for library creation, so this project also serves as a skill-building exercise.

### Installation and features

Simply use `pip install flash-rwkv` .

### How to use FlashRWKV

#### RWKV5
```python
from flash_rwkv import rwkv5_cuda_linear_attention

out = rwkv5_cuda_linear_attention(receptance, key, value, time_decay, time_first, state)
```

Here, receptance, key, value, time_decay, time_first, state are intermediate results generated by the RWKV Linear Attention module. The shape of these Tensors and their equivalent naive Python computation process can be seen in the[rwkv5 complete test file](tests/test_rwkv5_linear_attention.py)。

### Why Flash

The CUDA kernel used here is the optimal version we manually implemented in [RWKV-CUDA](https://github.com/BlinkDL/RWKV-CUDA). Compared to the simple Hugging Face implementation or naive CUDA kernels, it offers significant acceleration in both forward and backward operations. Detailed benchmarks will be posted here later, and we will also explore new optimization opportunities.


### Changelog

- [x] Released `v0.2.0` on 2024.4.6, supporting the RWKV5 model, and providing the `rwkv5_cuda_linear_attention` API.

### Plan

- [ ] Operator and end2end model benchmarking.
- [ ] Integration of this library's operators with Hugging Face's RWKV models.
- [ ] Support for RWKV6.
- [ ] Continue optimize kernel.




