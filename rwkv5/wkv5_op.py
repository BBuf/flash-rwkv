from pathlib import Path
import torch

KERNEL_PATH = Path(__file__).parent / "kernels" / "wkv5_cuda_kernel.so"

rwkv5_cuda_kernel = None

def load_wkv5_cuda_kernel(head_size):
    global rwkv5_cuda_kernel

    if rwkv5_cuda_kernel is None:
        print(f"Loading pre-compiled CUDA kernel for RWKV5 at head size of {head_size}.")
        rwkv5_cuda_kernel = torch.ops.load_library(str(KERNEL_PATH))

    return


class Rwkv5LinearAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, receptance, key, value, time_decay, time_first, state):
        with torch.no_grad():
            assert receptance.dtype == torch.bfloat16
            assert key.dtype == torch.bfloat16
            assert value.dtype == torch.bfloat16
            assert time_decay.dtype == torch.bfloat16
            assert time_first.dtype == torch.bfloat16
            assert state.dtype == torch.float32
            batch, seq_length, hidden_size = key.shape
            num_heads = time_decay.shape[0]
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
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )
            state = state.clone()
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
            return out, state

    @staticmethod
    def backward(ctx, gout):
        with torch.no_grad():
            assert gout.dtype == torch.bfloat16
            batch = ctx.batch
            seq_length = ctx.seq_length
            hidden_size = ctx.hidden_size
            num_heads = ctx.num_heads
            receptance, key, value, ee_time_decay, e_time_decay, time_first = ctx.saved_tensors

            global_shape = (batch, seq_length, hidden_size)

            # TODO dtype should not be forced here IMO
            greceptance = torch.empty(
                global_shape,
                device=gout.device,
                requires_grad=False,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )
            g_key = torch.empty(
                global_shape,
                device=gout.device,
                requires_grad=False,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )
            g_value = torch.empty(
                global_shape,
                device=gout.device,
                requires_grad=False,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )
            g_time_decay = torch.empty(
                (batch, hidden_size),
                device=gout.device,
                requires_grad=False,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )
            g_time_first = torch.empty(
                (batch, hidden_size),
                device=gout.device,
                requires_grad=False,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )
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
                greceptance,
                g_key,
                g_value,
                g_time_decay,
                g_time_first,
            )
            head_size = hidden_size // num_heads
            g_time_decay = torch.sum(g_time_decay, 0).view(num_heads, head_size)
            g_time_first = torch.sum(g_time_first, 0).view(num_heads, head_size)
            return (None, None, None, None, greceptance, g_key, g_value, g_time_decay, g_time_first)
