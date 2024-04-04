import torch
import unittest
import flash_rwkv
from flash_rwkv import Rwkv5LinearAttention

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

class TestRwkv5LinearAttention(unittest.TestCase):
    def test_rwkv5_linear_attention(self):
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
            out_cuda, state_cuda = Rwkv5LinearAttention.apply(receptance.to("cuda"), key.to("cuda"), value.to("cuda"), time_decay.to("cuda"), time_first.to("cuda"), state.to("cuda"))

            if dtype == torch.float32:
                self.assertTrue(torch.allclose(out_cpu.to(dtype), out_cuda.cpu()))
            else:
                self.assertTrue(torch.allclose(out_cpu.to(dtype), out_cuda.cpu(), rtol=0.1, atol=0.1))
            print(f'passed dtype: {dtype}')

if __name__ == '__main__':
    unittest.main()
