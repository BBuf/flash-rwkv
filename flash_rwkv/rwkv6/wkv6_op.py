import os
import torch
from pathlib import Path
from torch.utils.cpp_extension import load as load_kernel

rwkv6_cuda_kernel = None

def load_wkv6_cuda_kernel():
    global rwkv6_cuda_kernel
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_sources = [
        os.path.join(current_dir, "wkv6_op.cpp"),
        os.path.join(current_dir, "wkv6_kernel.cu"),
    ]

    if rwkv6_cuda_kernel is None:
        flags = [
            "-res-usage",
            "--maxrregcount 60",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-D_N_={64}",
            ]
        rwkv6_cuda_kernel = load_kernel(
            name="flash_rwkv_5",
            sources=cuda_sources,
            verbose=True,
            extra_cuda_cflags=flags,
        )
    return


class Rwkv6LinearAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, receptance, key, value, time_decay, time_first, state):
        with torch.no_grad():
            assert state.dtype == torch.float32
            batch, seq_length, hidden_size = key.shape
            num_heads = time_first.shape[0] // 64
            ctx.batch = batch
            ctx.seq_length = seq_length
            ctx.hidden_size = hidden_size
            ctx.num_heads = num_heads
            e_time_decay = (-torch.exp(time_decay.float())).contiguous()
            # ee_time_decay = (torch.exp(e_time_decay)).contiguous()
            # assert ee_time_decay.dtype == torch.float32
            # ctx.save_for_backward(receptance, key, value, ee_time_decay, e_time_decay, time_first)
            out = torch.empty(
                (batch, seq_length, num_heads, hidden_size // num_heads),
                device=receptance.device,
                dtype=receptance.dtype,
                memory_format=torch.contiguous_format,
            )
            state = state.clone()
            if receptance.dtype == torch.bfloat16:
                rwkv6_cuda_kernel.forward_bf16(
                    batch,
                    seq_length,
                    hidden_size,
                    num_heads,
                    state,
                    receptance,
                    key,
                    value,
                    e_time_decay,
                    time_first,
                    out,
                )
            elif receptance.dtype == torch.float16:
                rwkv6_cuda_kernel.forward_fp16(
                    batch,
                    seq_length,
                    hidden_size,
                    num_heads,
                    state,
                    receptance,
                    key,
                    value,
                    e_time_decay,
                    time_first,
                    out,
                )
            elif receptance.dtype == torch.float32:
                rwkv6_cuda_kernel.forward_fp32(
                    batch,
                    seq_length,
                    hidden_size,
                    num_heads,
                    state,
                    receptance,
                    key,
                    value,
                    e_time_decay,
                    time_first,
                    out,
                )
            return out, state

    @staticmethod
    def backward(ctx, gout, gstate):
        with torch.no_grad():
            return (None, None, None, None, None, None)

load_wkv6_cuda_kernel()

def rwkv6_cuda_linear_attention(receptance, key, value, time_decay, time_first, state):
    return Rwkv6LinearAttention.apply(receptance, key, value, time_decay, time_first, state)
