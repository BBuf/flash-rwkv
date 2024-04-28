import torch

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

    batch, seq_length, _ = q.shape
    num_heads, head_size = u.shape
    k = k.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2) # B, T, H, K -> B, H, T, K
    v = v.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2) # B, T, H, K - > B, H, T, V
    q = q.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2) # B, H, T, K
    w = torch.exp(-torch.exp(w.float())).view(batch, seq_length, num_heads, head_size).permute(0, 2, 1, 3) # B, T, H, K -> B, H, T, K
    u = u.float().reshape(num_heads, head_size) # H, K

    # batch_size, n_heads, seq_len, d_head_k = q.shape
    # _, _, _, d_head_v = v.shape
    h = torch.zeros(batch, num_heads, head_size, head_size, dtype=torch.float32, device=q.device)
    o = torch.zeros_like(v)

    if initial_state is not None:
        h += initial_state

    for i in range(seq_length):
        q_i = q[:, :, i, :]
        k_i = k[:, :, i]
        v_i = v[:, :, i, :]
        w_i = w[:, :, i].exp()
        kv_i = k_i[..., None] * v_i[..., None, :]
        o_i = (h + u[None, ..., None] * kv_i) * q_i[..., None]
        o[:, :, i] = o_i.sum(-2)
        h = h * w_i[..., None] + kv_i
    return o.to(orig_dtype)

def rwkv6_linear_attention_cpu(receptance, key, value, time_decay, time_first, state):
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
    q = (torch.randn(B, L, HIDDEN_SIZE).cuda().to(torch.float16)).requires_grad_(True)
    k = (torch.randn(B, L, HIDDEN_SIZE).cuda().to(torch.float16)).requires_grad_(True)
    v = torch.randn(B, L, HIDDEN_SIZE).cuda().to(torch.float16).requires_grad_(True)
    w = torch.nn.functional.logsigmoid(torch.randn(B, L, HIDDEN_SIZE)).cuda().to(torch.float32).requires_grad_(True)
    u = (torch.randn(H, D).cuda().to(torch.float16)).requires_grad_(True)
    state = (torch.randn(B, H, D, D).cuda().to(torch.float32)).requires_grad_(True)
    
    o1 = naive_recurrent_rwkv6(q, k, v, w, u, initial_state=state).transpose(1, 2)
    print('o1.shape: ', o1.shape)
    print('o1 first: ', o1.cpu().detach().numpy().flatten()[:10])
    print('o1 end: ', o1.cpu().detach().numpy().flatten()[-10:])
    o2, _ = rwkv6_linear_attention_cpu(q, k, v, w, u, state)
    print('o2.shape: ', o2.shape)
    print('o2 first: ', o2.cpu().detach().numpy().flatten()[:10])
    print('o2 end: ', o2.cpu().detach().numpy().flatten()[-10:])

    
