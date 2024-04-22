import os
import torch
from pathlib import Path
from torch.utils.cpp_extension import load as load_kernel

rwkv5_cuda_kernel = None

def load_wkv5_cuda_kernel():
    global rwkv5_cuda_kernel
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_sources = [
        os.path.join(current_dir, "wkv5_op.cpp"),
        os.path.join(current_dir, "wkv5_kernel.cu"),
    ]

    if rwkv5_cuda_kernel is None:
        flags = [
            "-res-usage",
            "--maxrregcount 60",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-D_N_={64}",
            ]
        rwkv5_cuda_kernel = load_kernel(
            name="flash_rwkv_5",
            sources=cuda_sources,
            verbose=True,
            extra_cuda_cflags=flags,
        )
    return


class Rwkv5LinearAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, receptance, key, value, time_decay, time_first, state):
        with torch.no_grad():
            assert state.dtype == torch.float32
            batch, seq_length, hidden_size = key.shape
            num_heads = time_decay.shape[0] // 64
            ctx.batch = batch
            ctx.seq_length = seq_length
            ctx.hidden_size = hidden_size
            ctx.num_heads = num_heads
            e_time_decay = (-torch.exp(time_decay.float())).contiguous()
            ee_time_decay = (torch.exp(e_time_decay)).contiguous()
            assert ee_time_decay.dtype == torch.float32
            ctx.save_for_backward(receptance, key, value, ee_time_decay, e_time_decay, time_first)
            out = torch.empty(
                (batch, seq_length, hidden_size),
                device=receptance.device,
                dtype=receptance.dtype,
                memory_format=torch.contiguous_format,
            )
            state = state.clone()
            if receptance.dtype == torch.bfloat16:
                rwkv5_cuda_kernel.forward_bf16(
                    batch,
                    seq_length,
                    hidden_size,
                    num_heads,
                    state,
                    receptance,
                    key,
                    value,
                    ee_time_decay,
                    time_first,
                    out,
                )
            elif receptance.dtype == torch.float16:
                rwkv5_cuda_kernel.forward_fp16(
                    batch,
                    seq_length,
                    hidden_size,
                    num_heads,
                    state,
                    receptance,
                    key,
                    value,
                    ee_time_decay,
                    time_first,
                    out,
                )
            elif receptance.dtype == torch.float32:
                rwkv5_cuda_kernel.forward_fp32(
                    batch,
                    seq_length,
                    hidden_size,
                    num_heads,
                    state,
                    receptance,
                    key,
                    value,
                    ee_time_decay,
                    time_first,
                    out,
                )
            return out, state

    @staticmethod
    def backward(ctx, gout, gstate):
        with torch.no_grad():
            batch = ctx.batch
            seq_length = ctx.seq_length
            hidden_size = ctx.hidden_size
            num_heads = ctx.num_heads
            receptance, key, value, ee_time_decay, e_time_decay, time_first = ctx.saved_tensors

            global_shape = (batch, seq_length, hidden_size)

            g_receptance = torch.empty(
                global_shape,
                device=gout.device,
                requires_grad=False,
                dtype=receptance.dtype,
                memory_format=torch.contiguous_format,
            )
            g_key = torch.empty(
                global_shape,
                device=gout.device,
                requires_grad=False,
                dtype=receptance.dtype,
                memory_format=torch.contiguous_format,
            )
            g_value = torch.empty(
                global_shape,
                device=gout.device,
                requires_grad=False,
                dtype=receptance.dtype,
                memory_format=torch.contiguous_format,
            )
            g_time_decay = torch.empty(
                (batch, hidden_size),
                device=gout.device,
                requires_grad=False,
                dtype=receptance.dtype,
                memory_format=torch.contiguous_format,
            )
            g_time_first = torch.empty(
                (batch, hidden_size),
                device=gout.device,
                requires_grad=False,
                dtype=receptance.dtype,
                memory_format=torch.contiguous_format,
            )
            if receptance.dtype == torch.bfloat16:
                rwkv5_cuda_kernel.backward_bf16(
                    batch,
                    seq_length,
                    hidden_size,
                    num_heads,
                    receptance,
                    key,
                    value,
                    ee_time_decay,
                    e_time_decay,
                    time_first,
                    gout,
                    g_receptance,
                    g_key,
                    g_value,
                    g_time_decay,
                    g_time_first,
                )
            elif receptance.dtype == torch.float16:
                rwkv5_cuda_kernel.backward_fp16(
                    batch,
                    seq_length,
                    hidden_size,
                    num_heads,
                    receptance,
                    key,
                    value,
                    ee_time_decay,
                    e_time_decay,
                    time_first,
                    gout,
                    g_receptance,
                    g_key,
                    g_value,
                    g_time_decay,
                    g_time_first,
                )
            elif receptance.dtype == torch.float32:
                rwkv5_cuda_kernel.backward_fp32(
                    batch,
                    seq_length,
                    hidden_size,
                    num_heads,
                    receptance,
                    key,
                    value,
                    ee_time_decay,
                    e_time_decay,
                    time_first,
                    gout,
                    g_receptance,
                    g_key,
                    g_value,
                    g_time_decay,
                    g_time_first,
                )
            head_size = hidden_size // num_heads
            g_time_decay = torch.sum(g_time_decay, 0).flatten()
            g_time_first = torch.sum(g_time_first, 0).flatten()
            return (g_receptance, g_key, g_value, g_time_decay, g_time_first, None)

load_wkv5_cuda_kernel()

def rwkv5_cuda_linear_attention(receptance, key, value, time_decay, time_first, state):
    return Rwkv5LinearAttention.apply(receptance, key, value, time_decay, time_first, state)
