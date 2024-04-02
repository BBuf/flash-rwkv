import torch
import unittest
import flash_rwkv
from flash_rwkv import Rwkv5LinearAttention

def rwkv5_linear_attention_cpu(receptance, key, value, time_decay, time_first, state):
    # For CPU fallback. Will be slower and probably take more memory than the custom CUDA kernel if not executed
    # within a torch.no_grad.
    batch, seq_length, hidden_size = receptance.shape
    num_heads, head_size = time_first.shape
    key = key.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2).transpose(-2, -1)
    value = value.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2)
    receptance = receptance.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2)
    time_decay = torch.exp(-torch.exp(time_decay.float())).reshape(-1, 1, 1).reshape(num_heads, -1, 1)
    time_first = time_first.float().reshape(-1, 1, 1).reshape(num_heads, -1, 1)
    out = torch.zeros_like(key).reshape(batch, seq_length, num_heads, head_size)

    for current_index in range(seq_length):
        current_receptance = receptance[:, :, current_index:current_index+1, :]
        current_key = key[:, :, :, current_index:current_index+1]
        current_value = value[:, :, current_index:current_index+1, :]
        attention_output = current_key @ current_value
        out[:, current_index] = (current_receptance @ (time_first * attention_output + state)).squeeze(2)
        with torch.no_grad():
            state = attention_output + time_decay * state

    return out.reshape(batch, seq_length, -1), state

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
        time_decay = torch.randn(num_heads, head_size, dtype=torch.float32)
        time_first = torch.randn(num_heads, head_size, dtype=torch.float32)
        state = torch.randn(batch_size, num_heads, head_size, head_size, dtype=torch.float32)

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            receptance = receptance.to(dtype)
            key = key.to(dtype)
            value = value.to(dtype)
            time_decay = time_decay.to(dtype)
            time_first = time_first.to(dtype)
            state = state.to(dtype)

            out_cpu, state_cpu = rwkv5_linear_attention_cpu(receptance, key, value, time_decay, time_first, state)
            out_cuda, state_cuda = Rwkv5LinearAttention.apply(receptance.to("cuda"), key.to("cuda"), value.to("cuda"), time_decay.to("cuda"), time_first.to("cuda"), state.to("cuda"))

            print('out_cpu.shape: ', out_cpu.shape)
            print('out_cuda.shape: ', out_cuda.shape)
            print('output_cpu: ', out_cpu.numpy().flatten()[:20])
            print('out_cuda: ', out_cuda.cpu().numpy().flatten()[:20])
            self.assertTrue(torch.allclose(out_cpu, out_cuda.cpu(), atol=1e-3, rtol=1e-3))
            self.assertTrue(torch.allclose(state_cpu, state_cuda.cpu(), atol=1e-3, rtol=1e-3))

if __name__ == '__main__':
    unittest.main()
