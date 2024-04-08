import torch
import unittest
import flash_rwkv
from flash_rwkv import rwkv5_cuda_linear_attention

# copy from https://github.com/BlinkDL/RWKV-CUDA/blob/main/wkv5/run.py#L47-L65
def RUN_FORMULA_1(B, T, C, H, r, k, v, w, u):
    N = C // H
    r = r.view(B, T, H, N)
    k = k.view(B, T, H, N)
    v = v.view(B, T, H, N)
    w = w.view(H, N)
    u = u.view(H, N)
    out = torch.zeros((B, T, H, N), device="cuda")

    for b in range(B):
        for h in range(H):
            for t in range(T):
                for i in range(N):
                    for j in range(N):
                        for tt in range(t+1):
                            ww = u[h,j] if (tt == t) else w[h,j] ** (t - tt - 1)
                            out[b,t,h,i] += r[b,t,h,j] * ww * k[b,tt,h,j] * v[b,tt,h,i]

    return out.view(B, T, C)

# copy from https://github.com/BlinkDL/RWKV-CUDA/blob/main/wkv5/run.py#L112C1-L174C29
def RUN_FORMULA_2(B, T, C, H, r, k, v, w, u):
    N = C // H
    r = r.flatten().contiguous() # BTHN
    k = k.flatten().contiguous() # BTHN
    v = v.flatten().contiguous() # BTHN
    w = w.flatten().contiguous() # HN
    u = u.flatten().contiguous() # HN
    out = torch.zeros(B*T*C, device="cpu").contiguous()

    # kernel for v1/v1a/v1b
    for b in range(B):
        for h in range(H):
            state = torch.zeros(N*N, device="cpu").contiguous()
            for t in range(T):

                _o0 = b*H*T*N + t*H*N + h*N
                _o1 = h*N

                for _i in range(N):

                    i = _o0 + _i
                    
                    for _j in range(N):
                        
                        j = _o0 + _j
                        m = _o1 + _j
                        ij = _i * N + _j

                        x = k[j] * v[i]
                        s = state[ij]
                        
                        out[i] += r[j] * (u[m] * x + s)
                        state[ij] = s * w[m] + x

    return out.view(B, T, C)

# copy from https://github.com/BlinkDL/RWKV-CUDA/blob/main/wkv5/run.py#L176-L215
def RUN_BACKWARD_1(B, T, C, H, gy, r, k, v, __w, u):
    N = C // H
    gy = gy.view(B, T, H, N)
    r = r.view(B, T, H, N)
    k = k.view(B, T, H, N)
    v = v.view(B, T, H, N)
    _w = -torch.exp(__w).view(H, N)
    u = u.view(H, N)
    w = torch.exp(_w)

    gr = torch.zeros((B, T, H, N), device="cuda")
    gk = torch.zeros((B, T, H, N), device="cuda")
    gv = torch.zeros((B, T, H, N), device="cuda")
    gw = torch.zeros((H, N), device="cuda")
    gu = torch.zeros((H, N), device="cuda")

    for b in range(B):
        for h in range(H):
            for i in range(N):
                for t in range(T):
                    for j in range(N):

                        for tt in range(t+1):
                            ww = u[h,i] if (tt == t) else w[h,i] ** (t - tt - 1)
                            gr[b,t,h,i] += ww * k[b,tt,h,i] * v[b,tt,h,j] * gy[b,t,h,j]

                        for tt in range(t,T):
                            ww = u[h,i] if (tt == t) else w[h,i] ** (tt - t - 1)
                            gk[b,t,h,i] += r[b,tt,h,i] * ww * v[b,t,h,j] * gy[b,tt,h,j]

                            ww = u[h,j] if (tt == t) else w[h,j] ** (tt - t - 1)
                            gv[b,t,h,i] += r[b,tt,h,j] * ww * k[b,t,h,j] * gy[b,tt,h,i]

                        gu[h,i] += r[b,t,h,i] * k[b,t,h,i] * v[b,t,h,j] * gy[b,t,h,j]

                        for tt in range(t-1):
                            ww = (t-tt-1) * _w[h,i] * (w[h,i] ** (t - tt - 1))
                            gw[h,i] += r[b,t,h,i] * ww * k[b,tt,h,i] * v[b,tt,h,j] * gy[b,t,h,j]

    return gr.view(B, T, C), gk.view(B, T, C), gv.view(B, T, C), gw.view(C), gu.view(C)

class TestRwkv5LinearAttention(unittest.TestCase):
    def test_rwkv5_linear_attention_prefill_forward(self):
        batch_size = 4
        seq_length = 16
        hidden_size = 128
        num_heads = 2
        head_size = 64

        receptance = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float32)
        key = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float32)
        value = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float32)
        time_decay = torch.randn(hidden_size, dtype=torch.float32)
        time_first = torch.randn(hidden_size, dtype=torch.float32)
        state = torch.zeros(batch_size, num_heads, head_size, head_size, dtype=torch.float32)

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            receptance = receptance.to(dtype)
            key = key.to(dtype)
            value = value.to(dtype)
            time_decay = time_decay.to(dtype)
            time_first = time_first.to(dtype)

            out_cpu = RUN_FORMULA_2(batch_size, seq_length, hidden_size, num_heads, receptance, key, value, torch.exp(-torch.exp(time_decay)), time_first)
            out_cuda = rwkv5_cuda_linear_attention(receptance.to("cuda"), key.to("cuda"), value.to("cuda"), time_decay.to("cuda"), time_first.to("cuda"), state.to("cuda"))

            self.assertTrue(torch.allclose(out_cpu.to(dtype), out_cuda.cpu(), rtol=0.1, atol=0.1))
            print(f'test_rwkv5_linear_attention_prefill_forward passed dtype: {dtype}')

    def test_rwkv5_linear_attention_decode_forward(self):
        batch_size = 1
        seq_length = 1
        hidden_size = 2048
        num_heads = 32
        head_size = 64

        receptance = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float32)
        key = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float32)
        value = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.float32)
        time_decay = torch.randn(hidden_size, dtype=torch.float32)
        time_first = torch.randn(hidden_size, dtype=torch.float32)
        state = torch.zeros(batch_size, num_heads, head_size, head_size, dtype=torch.float32)

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            receptance = receptance.to(dtype)
            key = key.to(dtype)
            value = value.to(dtype)
            time_decay = time_decay.to(dtype)
            time_first = time_first.to(dtype)

            out_cpu = RUN_FORMULA_2(batch_size, seq_length, hidden_size, num_heads, receptance, key, value, torch.exp(-torch.exp(time_decay)), time_first)
            out_cuda = rwkv5_cuda_linear_attention(receptance.to("cuda"), key.to("cuda"), value.to("cuda"), time_decay.to("cuda"), time_first.to("cuda"), state.to("cuda"))

            self.assertTrue(torch.allclose(out_cpu.to(dtype), out_cuda.cpu(), rtol=0.1, atol=0.1))
            print(f'test_rwkv5_linear_attention_decode_forward passed dtype: {dtype}')

    def test_rwkv5_linear_attention_backward(self):
        batch_size = 1
        seq_length = 2
        hidden_size = 64
        num_heads = 1
        head_size = 64

        def LOSS(y):
            return ((y * y) - torch.tanh(y)).sum()
        
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            receptance = torch.randn(batch_size, seq_length, hidden_size, dtype=dtype, device="cuda").uniform_(-1, 1).requires_grad_(True)
            key = torch.randn(batch_size, seq_length, hidden_size, dtype=dtype, device="cuda").uniform_(-1, 1).requires_grad_(True)
            value = torch.randn(batch_size, seq_length, hidden_size, dtype=dtype, device="cuda").uniform_(-1, 1).requires_grad_(True)
            time_decay = torch.randn(hidden_size, dtype=dtype, device="cuda").uniform_(-1, 1).requires_grad_(True)
            time_first = torch.randn(hidden_size, dtype=dtype, device="cuda").uniform_(-1, 1).requires_grad_(True)
            state = torch.zeros(batch_size, num_heads, head_size, head_size, dtype=torch.float32, device="cuda")

            out_formula1 = RUN_FORMULA_1(batch_size, seq_length, hidden_size, num_heads, receptance, key, value, torch.exp(-torch.exp(time_decay)), time_first)
            yy = out_formula1.clone().detach().requires_grad_(True)
            
            LOSS(yy).backward()
            g_formula1 = yy.grad.data.clone()
            LOSS(out_formula1).backward()
            g_receptance_0 = receptance.grad.data.clone()
            g_key_0 = key.grad.data.clone()
            g_value_0 = value.grad.data.clone()
            g_time_decay_0 = time_decay.grad.data.clone()
            g_time_first_0 = time_first.grad.data.clone()
            receptance.grad.data.zero_()
            key.grad.data.zero_()
            value.grad.data.zero_()
            time_decay.grad.data.zero_()
            time_first.grad.data.zero_()

            g_receptance_1, g_key_1, g_value_1,g_time_decay_1, g_time_first_1 = RUN_BACKWARD_1(batch_size, seq_length, hidden_size, num_heads, g_out_cpu, receptance, key, value, time_decay, time_first)
            self.assertTrue(torch.allclose(g_receptance_0.cpu().to(dtype), g_receptance_1.cpu().to(dtype), rtol=0.1, atol=0.1))
            self.assertTrue(torch.allclose(g_key_0.cpu().to(dtype), g_key_1.cpu().to(dtype), rtol=0.1, atol=0.1))
            self.assertTrue(torch.allclose(g_value_0.cpu().to(dtype), g_value_1.cpu().to(dtype), rtol=0.1, atol=0.1))
            self.assertTrue(torch.allclose(g_time_decay_0.cpu().to(dtype), g_time_decay_1.cpu().to(dtype), rtol=0.1, atol=0.1))
            self.assertTrue(torch.allclose(g_time_first_0.cpu().to(dtype), g_time_first_1.cpu().to(dtype), rtol=0.1, atol=0.1))

            receptance.grad.data.zero_()
            key.grad.data.zero_()
            value.grad.data.zero_()
            time_decay.grad.data.zero_()
            time_first.grad.data.zero_()

            out_cuda = rwkv5_cuda_linear_attention(receptance, key, value, time_decay, time_first, state)
            LOSS(out_cuda).backward()

            self.assertTrue(torch.allclose(out_formula1.cpu().to(dtype), out_cuda.cpu().to(dtype), rtol=0.1, atol=0.1))
            self.assertTrue(torch.allclose(g_receptance_0.cpu().to(dtype), receptance.grad.data.clone().cpu().to(dtype), rtol=0.1, atol=0.1))
            self.assertTrue(torch.allclose(g_key_0.cpu().to(dtype), key.grad.data.clone().cpu().to(dtype), rtol=0.1, atol=0.1))
            self.assertTrue(torch.allclose(g_value_0.cpu().to(dtype), value.grad.data.clone().cpu().to(dtype), rtol=0.1, atol=0.1))
            self.assertTrue(torch.allclose(g_time_decay_0.cpu().to(dtype), time_decay.grad.data.clone().cpu().to(dtype), rtol=0.1, atol=0.1))
            self.assertTrue(torch.allclose(g_time_first_0.cpu().to(dtype), time_first.grad.data.clone().cpu().to(dtype), rtol=0.1, atol=0.1))
            print(f'test_rwkv5_linear_attention_backward passed dtype: {dtype}')

if __name__ == '__main__':
    unittest.main()
