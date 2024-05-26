import torch
from einops import rearrange

def naive_linear_attn(q, k, v):
    q = q * (q.shape[-1] ** -0.5)
    scores = torch.matmul(q, k.transpose(-1, -2))
    mask = torch.triu(torch.ones(scores.shape[-2], scores.shape[-1], device=q.device), diagonal=1)
    scores = scores.masked_fill(mask.bool(), float(0))
    output = torch.matmul(scores, v)
    return output

def torch_chunk_linear_attn(q, k, v, chunk_size=64):
    q = rearrange(q, 'b h (n c) d -> b h n c d', c = chunk_size) * (q.shape[-1] **-0.5)
    k = rearrange(k, 'b h (n c) d -> b h n c d', c = chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c = chunk_size)
    kv = k.transpose(-1, -2) @ v
    kv = kv.cumsum(2)
    kv = torch.cat([
        torch.zeros_like(kv[:, :, :1]),
        kv[:, :, :-1]
    ], dim=2)
    inter = q @ kv # (b, h, n, c, d) @ (b, h, n, d, d) -> (b, h, n, c, d)
    intra = ((q @ k.transpose(-1, -2)).masked_fill_(torch.triu(torch.ones(chunk_size, chunk_size, dtype=bool, device=q.device), diagonal=1), 0)) @ v
    o = inter + intra
    return rearrange(o, 'b h n c d -> b h (n c) d')


if __name__ == "__main__":
    B = 4
    H = 4
    L = 1024
    D = 100
    dtype = torch.float32
    require_grad = True
    q = (torch.randn(B, H, L, D).to(dtype)).requires_grad_(require_grad)
    k = (torch.randn(B, H, L, D).to(dtype)).requires_grad_(require_grad)
    v = torch.randn(B, H, L, D).to(dtype).requires_grad_(require_grad)
    o1 = torch_chunk_linear_attn(q, k, v)
    o2 = naive_linear_attn(q, k, v)
    print('o1: ', o1.sum())
    print('o2: ', o2.sum())
