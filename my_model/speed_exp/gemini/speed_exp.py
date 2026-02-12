import torch
import time
from typing import Tuple

# ==========================================
# ✅ 1. 导入你的新模块
# ==========================================
try:
    from hstu_bsa_triton_v2 import HSTU_BSA_Triton
except ImportError:
    print("❌ 错误: 找不到 hstu_bsa_triton_v2.py，请确保文件在同一目录下！")
    exit(1)

def generate_random_jagged_qkv(
    batch_size: int, 
    max_seq_len: int, 
    num_heads: int, 
    dim: int,
    device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    生成符合 HSTU 格式的 Jagged Tensor 数据
    注意：为了性能和显存优化，这里默认生成 float16 数据
    """
    # 随机生成序列长度
    lengths = torch.randint(1, max_seq_len + 1, (batch_size,), device=device)
    # 生成 Offset
    seq_offsets = torch.cat([torch.tensor([0]).to(device), torch.cumsum(lengths, dim=0).to(device)]).to(torch.int32)
    total_L = seq_offsets[-1].item()
    
    # 构造 Q, K, V (使用 float16 以节省显存并符合 Triton 最佳实践)
    dtype = torch.float16 
    q = torch.randn(total_L, num_heads, dim, device=device, dtype=dtype)
    k = torch.randn(total_L, num_heads, dim, device=device, dtype=dtype)
    v = torch.randn(total_L, num_heads, dim, device=device, dtype=dtype)
    
    return q, k, v, seq_offsets, max_seq_len

def speed_exp(Bsize, max_seq_len, num_heads, emb_dim):
    device = 'cuda:0'
    if not torch.cuda.is_available():
        print("❌ 错误: 未检测到 GPU")
        return

    ALPHA = 1.0 / (emb_dim ** 0.5)
    
    print(f"\n📊 正在测试配置: [Batch={Bsize}, SeqLen={max_seq_len}, Heads={num_heads}, Dim={emb_dim}]")
    
    # 1. 实例化模型
    # 假设 HSTU_BSA_Triton 是一个 nn.Module 或类，不需要参数初始化，或者参数在 forward 中
    try:
        model = HSTU_BSA_Triton().to(device)
        # 如果它是纯函数封装，不需要 .to(device)，但这行通常不会报错
    except Exception as e:
        # 如果它不是类而是函数，直接赋值
        model = HSTU_BSA_Triton

    # 2. 准备数据
    try:
        q, k, v, seq_offsets, max_seq_len = generate_random_jagged_qkv(
            Bsize, max_seq_len, num_heads, emb_dim, device
        )
    except RuntimeError as e:
        print(f"❌ 数据生成阶段显存不足 (OOM): {e}")
        return

    # 3. 预热 (Warmup) - 触发 Triton 编译
    print("   🔥 正在预热 (Autotuning Kernel)...")
    try:
        # 预热 5 次
        for _ in range(5):
            # 假设调用方式与之前一致。如果你的 forward 参数不同，请在这里修改
            _ = model(
                N=max_seq_len, 
                alpha=ALPHA, 
                q=q, 
                k=k, 
                v=v, 
                seq_offsets=seq_offsets
            )
        torch.cuda.synchronize()
    except RuntimeError as e:
        if "out of memory" in str(e) or "shared memory" in str(e):
            print(f"❌ 预热失败: 显存/共享内存不足 (OOM)。请尝试减小 BLOCK_M。")
            print(f"   错误详情: {e}")
        else:
            print(f"❌ 预热运行时错误: {e}")
        return

    # ================= 核心测试区 =================
    
    # 重置显存统计
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    
    # 初始化计时器
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    
    try:
        start_evt.record()
        # === 执行推理 ===
        ret = model(
            N=max_seq_len, 
            alpha=ALPHA, 
            q=q, 
            k=k, 
            v=v, 
            seq_offsets=seq_offsets
        )
        # ===============
        end_evt.record()
        
        # 等待 GPU 完成
        torch.cuda.synchronize()
        
        # 计算结果
        elapsed_ms = start_evt.elapsed_time(end_evt)
        peak_mem = torch.cuda.max_memory_allocated()
        kernel_overhead = (peak_mem - base_mem) / 1024**2
        
        print(f"   ✅ 测试成功!")
        print(f"      - 耗时: {elapsed_ms:.3f} ms")
        print(f"      - 基础显存 (Input): {base_mem / 1024**2:.2f} MB")
        print(f"      - 峰值显存 (Total): {peak_mem / 1024**2:.2f} MB")
        print(f"      - Kernel额外开销:   {kernel_overhead:.2f} MB")
        
    except Exception as e:
        print(f"❌ 运行测试时崩溃: {e}")

if __name__ == "__main__":
    # 在这里定义你想测试的所有配置
    # 格式: (Batch, SeqLen, Heads, Dim)
    configs_to_test = [
        (32, 256, 8, 64),   # 小 Dim，基准测试
        (32, 256, 8, 128),  # 常规 Dim
        (32, 256, 8, 256),  # 中等 Dim (注意 Shared Memory)
        (32, 256, 8, 512),  # 大 Dim (如果在 Config 中没把 BLOCK_M 设为 16，这里可能会挂)
    ]

    print("🚀 开始 HSTU_BSA_Triton 性能测试...")
    for (B, L, H, D) in configs_to_test:
        speed_exp(B, L, H, D)
        # 稍微暂停一下释放资源
        time.sleep(0.5)