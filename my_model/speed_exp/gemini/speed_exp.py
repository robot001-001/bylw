import torch
import time
from typing import Tuple

# ==========================================
# 1. 导入你的模块
# ==========================================
try:
    # 假设你的文件名叫 hstu_bsa_triton_v2.py
    from hstu_bsa_triton_v2 import HSTU_BSA_Triton
except ImportError:
    print("❌ 错误: 找不到 hstu_bsa_triton_v2.py，请检查文件名！")
    exit(1)

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
    dtype = torch.float16
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

    print(f"\n📊 测试配置: [B={Bsize}, L={max_seq_len}, H={num_heads}, D={emb_dim}]")

    # 1. 实例化模型 (根据你的 __init__)
    try:
        # 这里你可以调整 block_size 和 block_counts
        model = HSTU_BSA_Triton(block_size=32, block_counts=4).to(device)
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return

    # 2. 准备数据
    try:
        q, k, v, g_cmp, g_slc, x_offsets = generate_hstu_bsa_inputs(
            Bsize, max_seq_len, num_heads, emb_dim, device
        )
    except RuntimeError as e:
        print(f"❌ 显存不足 (OOM) 无法生成数据: {e}")
        return

    # 3. 预热 (Warmup)
    print("   🔥 正在预热...")
    try:
        for _ in range(5):
            # [关键修改] 使用新的参数列表调用 forward
            _ = model(q, k, v, g_cmp, g_slc, x_offsets)
        torch.cuda.synchronize()
    except RuntimeError as e:
        print(f"❌ 预热失败 (可能参数不对或 OOM): {e}")
        return

    # 4. 性能测试
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    
    try:
        start_evt.record()
        # === 核心调用 ===
        ret = model(q, k, v, g_cmp, g_slc, x_offsets)
        # ===============
        end_evt.record()
        
        torch.cuda.synchronize()
        
        elapsed_ms = start_evt.elapsed_time(end_evt)
        peak_mem = torch.cuda.max_memory_allocated()
        kernel_overhead = (peak_mem - base_mem) / 1024**2
        
        print(f"   ✅ 完成!")
        print(f"      - 耗时: {elapsed_ms:.3f} ms")
        print(f"      - 显存开销 (Overhead): {kernel_overhead:.2f} MB")
        
    except Exception as e:
        print(f"❌ 运行崩溃: {e}")

if __name__ == "__main__":
    configs = [
        (32, 256, 8, 128),
        (32, 256, 8, 256),
        (32, 256, 8, 512), # 大 Dim 测试
    ]

    for (B, L, H, D) in configs:
        speed_exp(B, L, H, D)