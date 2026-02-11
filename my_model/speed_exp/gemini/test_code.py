import torch
import torch.nn as nn
import math
from hstu_bsa_triton import HSTU_BSA_Triton

# 假设上一条回复的代码保存在 hstu_bsa_impl.py 中
# from hstu_bsa_impl import HSTU_BSA_Triton
# 这里为了演示直接使用类定义 (请确保上面的 HSTU_BSA_Triton 类定义在当前作用域或已导入)

# ==========================================
# 1. 模拟辅助模块：Gate Model
# ==========================================
class MockGateModel(nn.Module):
    """
    模拟 HSTU 中的 Gate 生成网络。
    输入: Padded Tensor [B, N, H, D]
    输出: g_cmp, g_slc, g_swa (形状均为 [B, N, H])
    """
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, x):
        # x: [B, N, H, D]
        B, N, H, D = x.shape
        # 这里随机生成 gate score，实际模型中会有 Linear 层
        # 使用 sigmoid 确保范围在 (0, 1) 用于门控
        g_cmp = torch.rand(B, N, H, device=x.device)
        g_slc = torch.rand(B, N, H, device=x.device)
        g_swa = torch.rand(B, N, H, device=x.device)
        return g_cmp, g_slc, g_swa

# ==========================================
# 2. 数据构造工具：生成 Jagged Tensor
# ==========================================
def generate_jagged_data(batch_size, max_seq_len, num_heads, head_dim, device='cuda'):
    """
    生成符合 FBGEMM/HSTU 标准的 Jagged 数据
    """
    # 1. 随机生成每个样本的序列长度
    import random
    seq_lengths = [random.randint(32, max_seq_len) for _ in range(batch_size)]
    print(f"生成的 Batch 序列长度: {seq_lengths}")

    # 2. 构造 offsets (类似于 CSR 格式的行偏移)
    # x_offsets: [0, len1, len1+len2, ..., total_tokens]
    offsets = [0]
    for l in seq_lengths:
        offsets.append(offsets[-1] + l)
    
    x_offsets = torch.tensor(offsets, dtype=torch.long, device=device)
    total_tokens = offsets[-1]

    # 3. 构造展平的输入张量 [Total_Tokens, H, D]
    # 在 HSTU 中，Q/K/V 通常是直接展平的
    q = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=torch.float32)
    u = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=torch.float32) # 用于最后的残差/归一化

    return q, k, v, u, x_offsets

# ==========================================
# 3. 主运行逻辑
# ==========================================
def run_hstu_bsa_example():
    # 检查环境 (Triton 需要 GPU)
    if not torch.cuda.is_available():
        print("错误: 未检测到 CUDA 设备，Triton 无法运行。")
        return
    
    device = "cuda"
    torch.manual_seed(42)

    # --- 超参数设置 ---
    BATCH_SIZE = 4
    MAX_SEQ_LEN = 128
    NUM_HEADS = 8
    HEAD_DIM = 64
    BLOCK_SIZE = 32   # 压缩块大小
    BLOCK_COUNTS = 4  # TopK 选取的块数

    print(f"--- 初始化 HSTU_BSA 模型 (BlockSize={BLOCK_SIZE}, TopK={BLOCK_COUNTS}) ---")
    
    # 1. 实例化模型和 Gate
    # 注意：你需要确保 HSTU_BSA_Triton 类在上下文中可用
    model = HSTU_BSA_Triton(block_size=BLOCK_SIZE, block_counts=BLOCK_COUNTS).to(device)
    gate_model = MockGateModel(num_heads=NUM_HEADS).to(device)

    # 2. 准备数据
    print("--- 生成 Jagged 输入数据 ---")
    q, k, v, u, x_offsets = generate_jagged_data(BATCH_SIZE, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM, device)
    
    print(f"输入 Q 形状 (Flattened): {q.shape}")
    print(f"Offsets 形状: {x_offsets.shape}")

    # 3. 前向传播
    print("--- 开始前向传播 (Triton Kernel) ---")
    try:
        # 调用模型
        # invalid_attn_mask 在此实现中通过 offsets 隐式处理，这里传 None
        output, padded_q, padded_k = model(
            q, k, v, u, 
            x_offsets, 
            gate_model, 
            invalid_attn_mask=None
        )

        print("\n--- 运行成功! ---")
        print(f"输出 Output 形状: {output.shape}")
        
        # 4. 简单的输出检查
        if torch.isnan(output).any():
            print("警告: 输出包含 NaN!")
        else:
            print(f"输出均值: {output.mean().item():.4f}")
            print(f"输出方差: {output.var().item():.4f}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n运行出错: {e}")

if __name__ == "__main__":
    run_hstu_bsa_example()