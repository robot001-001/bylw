import torch
import time
from typing import Tuple

# ==========================================
# 1. 导入你的模块
# ==========================================
from hstu_bsa_triton_v2 import HSTU_BSA_Triton

def generate_hstu_bsa_inputs(
    batch_size: int, 
    max_seq_len: int, 
    num_heads: int, 
    dim: int,
    device
):
    """
    专门为 HSTU_BSA_Triton 生成输入数据
    新增: g_cmp, g_slc
    """
    # 1. 生成 Jagged 序列长度
    lengths = torch.randint(1, max_seq_len + 1, (batch_size,), device=device)
    x_offsets = torch.cat([torch.tensor([0]).to(device), torch.cumsum(lengths, dim=0).to(device)]).to(torch.int32)
    total_L = x_offsets[-1].item()
    
    # 2. 构造 Q, K, V (Float16)
    dtype = torch.float32
    q = torch.randn(total_L, num_heads, dim, device=device, dtype=dtype)
    k = torch.randn(total_L, num_heads, dim, device=device, dtype=dtype)
    v = torch.randn(total_L, num_heads, dim, device=device, dtype=dtype)
    
    # 3. [新增] 构造 Gates (g_cmp, g_slc)
    # 假设 Gate 的形状是 (Total_L, H) 或者 (Total_L, H, 1)
    # 根据你的代码: g_cmp = g_cmp.unsqueeze(-1) 可知输入可以是 2D
    g_cmp = torch.sigmoid(torch.randn(total_L, num_heads, device=device, dtype=dtype))
    g_slc = torch.sigmoid(torch.randn(total_L, num_heads, device=device, dtype=dtype))
    
    return q, k, v, g_cmp, g_slc, x_offsets

def speed_exp(Bsize, max_seq_len, num_heads, emb_dim):
    device = 'cuda:0'
    if not torch.cuda.is_available():
        print("❌ 未检测到 GPU")
        return

    model = HSTU_BSA_Triton(block_size=64, block_counts=4).to(device)

    q, k, v, g_cmp, g_slc, x_offsets = generate_hstu_bsa_inputs(
        Bsize, max_seq_len, num_heads, emb_dim, device
    )

    for _ in range(5):
        _ = model(q, k, v, g_cmp, g_slc, x_offsets)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    base_memory = torch.cuda.memory_allocated()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    ret = model(q, k, v, g_cmp, g_slc, x_offsets)
    end_event.record()
    
    torch.cuda.synchronize()
    
    peak_memory = torch.cuda.max_memory_allocated()
    kernel_memory_cost = peak_memory - base_memory
    time_cost_ms = start_event.elapsed_time(end_event)
    
    print(f'seq_len: {max_seq_len}')
    print(f"Time: {time_cost_ms:.3f} ms")
    print(f"Base Memory (Inputs): {base_memory / 1024**2:.2f} MB")
    print(f"Peak Memory (Total):  {peak_memory / 1024**2:.2f} MB")
    print(f"Kernel Overhead:      {kernel_memory_cost / 1024**2:.2f} MB")
    print("-" * 30)
        

if __name__ == "__main__":
    for seq_len in range(128, 1024*8+1, 128):
        time_cost = speed_exp(32, seq_len, 8, 256)