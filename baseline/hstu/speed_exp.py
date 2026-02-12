from generative_recommenders.ops.hstu_attention import hstu_mha
from generative_recommenders.ops.triton.triton_hstu_attention import (
    triton_cached_hstu_mha,
    triton_hstu_mha,
)

import time
from typing import Tuple

import torch



def generate_random_jagged_qkv(
    batch_size: int, 
    max_seq_len: int, 
    num_heads: int, 
    dim: int,
    device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    lengths = torch.randint(1, max_seq_len + 1, (batch_size,), device=device)
    seq_offsets = torch.cat([torch.tensor([0]).to(device), torch.cumsum(lengths, dim=0).to(device)]).to(torch.int32)
    total_L = seq_offsets[-1].item()
    q = torch.randn(total_L, num_heads, dim).to(device)
    k = torch.randn(total_L, num_heads, dim).to(device)
    v = torch.randn(total_L, num_heads, dim).to(device)
    return q, k, v, seq_offsets, max_seq_len


def speed_exp(Bsize, max_seq_len, num_heads, emb_dim):
    ALPHA = 1.0 / (emb_dim ** 0.5)
    # 1. 准备数据
    q, k, v, seq_offsets, max_seq_len = generate_random_jagged_qkv(Bsize, max_seq_len, num_heads, emb_dim, 'cuda:0')
    
    # 2. 预热 (Warmup)
    # GPU 需要预热才能达到稳定状态，否则第一次运行会包含各种初始化开销
    for _ in range(5):
        _ = triton_hstu_mha(N=max_seq_len, alpha=ALPHA, q=q, k=k, v=v, seq_offsets=seq_offsets)
    torch.cuda.synchronize() # 等待预热完成

    # ================= 显存测试开始 =================
    # 3. 重置峰值显存统计
    torch.cuda.reset_peak_memory_stats()
    # 记录当前的显存占用（仅仅是输入数据 q,k,v 占用的显存）
    base_memory = torch.cuda.memory_allocated()
    
    # 4. 运行推理 (为了测速准确，建议使用 CUDA Event)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    ret = triton_hstu_mha(N=max_seq_len, alpha=ALPHA, q=q, k=k, v=v, seq_offsets=seq_offsets)
    end_event.record()
    
    # 5. 等待 GPU以此确保计时和显存统计准确
    torch.cuda.synchronize()
    
    # 6. 获取统计数据
    peak_memory = torch.cuda.max_memory_allocated() # 运行过程中的历史峰值
    kernel_memory_cost = peak_memory - base_memory  # 仅计算 Kernel 运行产生的额外显存（中间激活值等）
    
    time_cost_ms = start_event.elapsed_time(end_event) # CUDA Event 返回的是毫秒
    
    print(f"Time: {time_cost_ms:.3f} ms")
    print(f"Base Memory (Inputs): {base_memory / 1024**2:.2f} MB")
    print(f"Peak Memory (Total):  {peak_memory / 1024**2:.2f} MB")
    print(f"Kernel Overhead:      {kernel_memory_cost / 1024**2:.2f} MB")
    print("-" * 30)

    return time_cost_ms / 1000.0 # 返回秒


if __name__ == "__main__":
    while 1:
        time_cost = speed_exp(32, 256, 8, 512)