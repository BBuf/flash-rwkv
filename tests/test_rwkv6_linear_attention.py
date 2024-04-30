import torch
from fla.ops.rwkv6.chunk import chunk_rwkv6
from fla.ops.rwkv6.recurrent_fuse import fused_recurrent_rwkv6

def naive_recurrent_rwkv6(
    q,
    k,
    v,
    w,
    u,
    initial_state=None,
    output_final_state=False
):
    orig_dtype = q.dtype
    q, k, v, w, u = map(lambda x: x.float(), (q, k, v, w, u))
    batch_size, n_heads, seq_len, d_head_k = q.shape
    _, _, _, d_head_v = v.shape
    h = torch.zeros(batch_size, n_heads, d_head_k, d_head_v, dtype=torch.float32, device=q.device)
    o = torch.zeros_like(v)

    if initial_state is not None:
        h += initial_state

    for i in range(seq_len):
        q_i = q[:, :, i, :]
        k_i = k[:, :, i]
        v_i = v[:, :, i, :]
        w_i = w[:, :, i].exp()
        kv_i = k_i[..., None] * v_i[..., None, :]
        o_i = (h + u[None, ..., None] * kv_i) * q_i[..., None]
        o[:, :, i] = o_i.sum(-2)
        h = h * w_i[..., None] + kv_i
    return o.to(orig_dtype)


def naive_recurrent_rwkv6_bwd(
    q,
    k,
    v,
    w,
    u,
    o,
    do,
    initial_state=None,
    output_final_state=False
):
    q, k, v, w, u, o, do = map(lambda x: x.float(), (q, k, v, w, u, o, do))
    batch_size, n_heads, seq_len, d_head_k = q.shape
    _, _, _, d_head_v = v.shape
    h = torch.zeros(batch_size, n_heads, d_head_k, d_head_v, dtype=torch.float32, device=q.device)
    dq = torch.zeros_like(q)
    dq_aux = torch.zeros_like(q)

    if initial_state is not None:
        h += initial_state

    for i in range(seq_len):
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        w_i = w[:, :, i].exp()
        kv_i = k_i[..., None] * v_i[..., None, :]
        h_i = (h + u[None, ..., None] * kv_i)
        dq_i = (do[:, :, i, None, :] * h_i).sum(-1)
        dq_aux_i = (do[:, :, i, None, :] * h).sum(-1)
        dq[:, :, i] = dq_i
        dq_aux[:, :, i] = dq_aux_i
        h = h * w_i[..., None] + kv_i

    du = torch.zeros_like(u)
    dh = torch.zeros_like(h)
    dk = torch.zeros_like(k)
    dk_aux = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    for i in range(seq_len-1, -1, -1):
        d_kv_i = do[:, :, i, None, :] * q[:, :, i, :, None]
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        du_i = (d_kv_i * k_i[..., None] * v_i[..., None, :]).sum(-1)
        du += du_i
        dk_i = (dh * v_i[..., None, :]).sum(-1)
        dk_aux[:, :, i] = dk_i
        dk_i += (d_kv_i * u[None, ..., None] * v_i[..., None, :]).sum(-1)
        dv_i = (d_kv_i * u[None, ..., None] * k_i[..., None]).sum(-2)
        dv_i += (dh * k_i[..., None]).sum(-2)

        dk[:, :, i] = dk_i
        dv[:, :, i] = dv_i
        dh = dh * w[:, :, i, :, None].exp() + d_kv_i

    # dw = q * dq_aux - k * dk_aux
    dw = torch.zeros_like(w)
    for i in range(seq_len-2, -1, -1):
        dw[:, :, i] = dw[:, :, i+1] + dq_aux[:, :, i+1] * q[:, :, i+1] - dk_aux[:, :, i] * k[:, :, i]

    return dq, dk, dv, dw, du

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
    B = 1
    H = 32
    L = 54
    D = 64
    HIDDEN_SIZE = H * D
    dtype = torch.float32
    # HF Impl Input
    q = (torch.randn(B, L, HIDDEN_SIZE).cuda().to(torch.float16)).requires_grad_(True)
    k = (torch.randn(B, L, HIDDEN_SIZE).cuda().to(torch.float16)).requires_grad_(True)
    v = torch.randn(B, L, HIDDEN_SIZE).cuda().to(torch.float16).requires_grad_(True)
    w = torch.nn.functional.logsigmoid(torch.randn(B, L, HIDDEN_SIZE)).cuda().to(torch.float32).requires_grad_(True)
    u = (torch.randn(H, D).cuda().to(torch.float16)).requires_grad_(True)
    state = (torch.randn(B, H, D, D).cuda().to(torch.float32)).requires_grad_(True)
    
    o1, state1 = hf_rwkv6_linear_attention_cpu(q, k, v, w, u, state)
    
    # FLA Impl Input
    batch, seq_length, _ = q.shape
    num_heads, head_size = u.shape
    k = k.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2) # B, T, H, K -> B, H, T, K
    v = v.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2) # B, T, H, K - > B, H, T, V
    q = q.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2) # B, H, T, K
    w = -torch.exp(w.float()).view(batch, seq_length, num_heads, head_size).permute(0, 2, 1, 3) # B, T, H, K -> B, H, T, K
    u = u.float().reshape(num_heads, head_size) # H, K

    o2 = naive_recurrent_rwkv6(q, k, v, w, u, initial_state=state).transpose(1, 2)
    assert o1.allclose(o2.to(torch.float32), 0, 1e-1), breakpoint()
    
    o3, state3 = fused_recurrent_rwkv6(q, k, v, w, u, initial_state=state, scale=1.0, output_final_state=True)
    assert o1.allclose(o3.transpose(1, 2).to(torch.float32), 0, 1e-1), breakpoint()
    assert state1.allclose(state3.to(torch.float32), 0, 1e-1), breakpoint()

    o4, state4 = chunk_rwkv6(q, k, v, w, u, initial_state=state, scale=1.0, output_final_state=True)
    assert o1.allclose(o4.transpose(1, 2).to(torch.float32), 0, 1e-1), breakpoint()
    assert state1.allclose(state4.to(torch.float32), 0, 1e-1), breakpoint()
