import sys
import torch
from fla.ops.rwkv6.chunk import chunk_rwkv6
from fla.ops.rwkv6.recurrent_fuse import fused_recurrent_rwkv6

def hf_rwkv6_linear_attention_cpu(receptance, key, value, time_decay, time_first, state):
    # For CPU fallback. Will be slower and probably take more memory than the custom CUDA kernel if not executed
    # within a torch.no_grad.
    batch, seq_length, _ = receptance.shape
    num_heads, head_size = time_first.shape
    key = key.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2).transpose(-2, -1)
    value = value.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2)
    receptance = receptance.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2)
    time_decay = torch.exp(-torch.exp(time_decay.float())).view(batch, seq_length, num_heads, head_size).permute(0, 2, 3, 1)
    time_first = time_first.float().reshape(-1, 1, 1).reshape(num_heads, -1, 1)
    out = torch.zeros_like(key).reshape(batch, seq_length, num_heads, head_size)

    for current_index in range(seq_length):
        current_receptance = receptance[:, :, current_index:current_index+1, :]
        current_key = key[:, :, :, current_index:current_index+1]
        current_value = value[:, :, current_index:current_index+1, :]
        current_time_decay = time_decay[:, :, :, current_index:current_index+1]
        attention_output = current_key @ current_value
        out[:, current_index] = (current_receptance @ (time_first * attention_output + state)).squeeze(2)
        with torch.no_grad():
            state = attention_output + current_time_decay * state

    return out, state



if __name__ == "__main__":
    mode = sys.argv[1]
    B = 1
    H = 32
    L = 54
    D = 64
    HIDDEN_SIZE = H * D
    dtype = torch.float32
    
    if mode == 'hf':
        profile_path = '/mnt/data/cangshui/bbuf/rwkv_profile_result/hf/'
    elif mode == 'recurrent':
        profile_path = '/mnt/data/cangshui/bbuf/rwkv_profile_result/recurrent/'
    elif mode == 'chunk':
        profile_path = '/mnt/data/cangshui/bbuf/rwkv_profile_result/chunk/'
    else:
        raise NotImplementedError
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_path, worker_name='worker0'),
        record_shapes=True,
        profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
        with_stack=True
    ) as p:
        for i in range(10):
            q = (torch.randn(B, L, HIDDEN_SIZE).cuda().to(torch.float16)).requires_grad_(True)
            k = (torch.randn(B, L, HIDDEN_SIZE).cuda().to(torch.float16)).requires_grad_(True)
            v = torch.randn(B, L, HIDDEN_SIZE).cuda().to(torch.float16).requires_grad_(True)
            w = torch.nn.functional.logsigmoid(torch.randn(B, L, HIDDEN_SIZE)).cuda().to(torch.float32).requires_grad_(True)
            u = (torch.randn(H, D).cuda().to(torch.float16)).requires_grad_(True)
            state = (torch.randn(B, H, D, D).cuda().to(torch.float32)).requires_grad_(True)
            if mode == 'hf':
                o1, state1 = hf_rwkv6_linear_attention_cpu(q, k, v, w, u, state)
            else:
                batch, seq_length, _ = q.shape
                num_heads, head_size = u.shape
                k = k.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2) # B, T, H, K -> B, H, T, K
                v = v.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2) # B, T, H, K - > B, H, T, V
                q = q.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2) # B, H, T, K
                w = -torch.exp(w.float()).view(batch, seq_length, num_heads, head_size).permute(0, 2, 1, 3) # B, T, H, K -> B, H, T, K
                u = u.float().reshape(num_heads, head_size) # H, K

                if mode == 'recurrent':
                    o2, state2 = fused_recurrent_rwkv6(q, k, v, w, u, initial_state=state, scale=1.0, output_final_state=True)
                elif mode == 'chunk':
                    o4, state4 = chunk_rwkv6(q, k, v, w, u, initial_state=state, scale=1.0, output_final_state=True)
            p.step()
