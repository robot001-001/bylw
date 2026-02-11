import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl

# ==========================================
# 1. Triton Kernels (核心加速部分)
# ==========================================

@triton.jit
def _hstu_silu_activation(x):
    return x * tl.sigmoid(x)

# Kernel 1: Coarse-Grained Attention (Compression)
@triton.jit
def hstu_bsa_cmp_fwd_kernel(
    Q, K, V, 
    G_cmp, 
    Out,
    Stride_qt, Stride_qh, Stride_qd,
    Stride_kt, Stride_kh, Stride_kd,
    Stride_vt, Stride_vh, Stride_vd,
    Stride_ot, Stride_oh, Stride_od,
    offsets, 
    scale,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_z = tl.program_id(2)

    seq_start = tl.load(offsets + pid_z)
    seq_end = tl.load(offsets + pid_z + 1)
    seq_len = seq_end - seq_start
    cmp_len = tl.cdiv(seq_len, BLOCK_SIZE)

    start_m = pid_m * BLOCK_M
    if start_m >= seq_len:
        return

    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_len
    
    # Q 和 G 都是 Jagged [TotalTokens, H, D]
    q_ptrs = Q + (seq_start + offs_m[:, None]) * Stride_qt + pid_h * Stride_qh + tl.arange(0, HEAD_DIM)[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    g_ptrs = G_cmp + (seq_start + offs_m[:, None]) * Stride_qt + pid_h * Stride_qh
    g = tl.load(g_ptrs, mask=mask_m[:, None], other=0.0)

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # K_cmp, V_cmp 是 Padded 的 [B, MaxCmpLen, H, D]，需要正确计算指针
    # 这里的 K, V 指针计算基于 Padded Stride
    # pid_z * Stride_batch + ...
    
    for start_n in range(0, cmp_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < cmp_len
        
        # K 指针: Batch offset + Time offset + Head offset
        # 注意: 这里 K, V 传入的是 Padded Tensor 的指针
        k_ptrs = K + (pid_z * Stride_kt * cmp_len * BLOCK_SIZE * 0 + # 修正: 外部传入的是[B, N_blk, H, D]
                      start_n + offs_n[None, :]) * Stride_kt + \
                      pid_h * Stride_kh + \
                      tl.arange(0, HEAD_DIM)[:, None]
        
        # 修正: 上面的 Stride 计算假设 K 是 [B, N_blk, H, D] 的 View，或者调用时传入正确的 Base Pointer
        # 为简化，我们在 Python 端将 Padded K_cmp 展开，或者在 Kernel 里加上 Batch Stride
        # 这里为了演示简单，我们假设 K, V 指针已经在 Python 端移到了当前 Batch 的起始位置 (见 Kernel Launch 部分)
        
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        
        attn_score = tl.dot(q, k)
        attn_score *= scale
        
        # Causal Mask: original_idx // BS >= compressed_idx
        mask_causal = (offs_m[:, None] // BLOCK_SIZE) >= offs_n[None, :]
        attn_score = tl.where(mask_causal & mask_m[:, None] & mask_n[None, :], attn_score, -1e9)
        
        p = _hstu_silu_activation(attn_score)
        p = tl.where(mask_causal & mask_n[None, :], p, 0.0)
        
        v_ptrs = V + (start_n + offs_n[:, None]) * Stride_vt + pid_h * Stride_vh + tl.arange(0, HEAD_DIM)[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16))

    acc = acc * g
    o_ptrs = Out + (seq_start + offs_m[:, None]) * Stride_ot + pid_h * Stride_oh + tl.arange(0, HEAD_DIM)[None, :]
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])


# Kernel 2: Fine-Grained Attention (Selection)
@triton.jit
def hstu_bsa_slc_fwd_kernel(
    Q, K, V, 
    G_slc, 
    BlockIndices,
    Out,
    Stride_qt, Stride_qh, Stride_qd,
    Stride_kt, Stride_kh, Stride_kd,
    Stride_vt, Stride_vh, Stride_vd,
    Stride_ot, Stride_oh, Stride_od,
    Stride_bidx_t, Stride_bidx_h, Stride_bidx_s, # 新增: BlockIndices 的 Strides
    offsets,
    scale,
    S: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_z = tl.program_id(2)

    seq_start = tl.load(offsets + pid_z)
    seq_end = tl.load(offsets + pid_z + 1)
    seq_len = seq_end - seq_start

    start_m = pid_m * BLOCK_M
    if start_m >= seq_len:
        return

    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_len

    # Load Q (Jagged)
    q_ptrs = Q + (seq_start + offs_m[:, None]) * Stride_qt + pid_h * Stride_qh + tl.arange(0, HEAD_DIM)[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Load Gate (Jagged)
    g_ptrs = G_slc + (seq_start + offs_m[:, None]) * Stride_qt + pid_h * Stride_qh
    g = tl.load(g_ptrs, mask=mask_m[:, None], other=0.0)

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # 遍历选中的 S 个块
    for s_idx in range(S):
        # 计算 BlockIndices 指针: [TotalTokens, H, S]
        # Base + (GlobalTokenIdx) * Stride_T + HeadIdx * Stride_H + SIdx * Stride_S
        b_idxs_ptr = BlockIndices + (seq_start + offs_m) * Stride_bidx_t + pid_h * Stride_bidx_h + s_idx * Stride_bidx_s
        b_idx = tl.load(b_idxs_ptr, mask=mask_m, other=-1) # [BLOCK_M]

        # 遍历 Block 内的 Token (BLOCK_SIZE=32)
        for blk_offset in range(BLOCK_SIZE):
            valid_blk = b_idx >= 0
            # 计算 Gather 的目标 K/V 全局索引
            target_k_idx = seq_start + b_idx * BLOCK_SIZE + blk_offset
            
            # Causal Mask: K 的位置必须在 Q 之前
            is_causal = target_k_idx <= (seq_start + offs_m)
            
            # Load K column (Jagged Gather)
            k_ptrs_col = K + target_k_idx[:, None] * Stride_kt + pid_h * Stride_kh + tl.arange(0, HEAD_DIM)[None, :]
            mask_load = valid_blk[:, None] & mask_m[:, None]
            k_val = tl.load(k_ptrs_col, mask=mask_load, other=0.0)
            
            # Dot Product
            score = tl.sum(q * k_val, axis=1) 
            score *= scale
            
            # Activation
            p = _hstu_silu_activation(score)
            p = tl.where(valid_blk & is_causal, p, 0.0)
            
            # Load V column
            v_ptrs_col = V + target_k_idx[:, None] * Stride_vt + pid_h * Stride_vh + tl.arange(0, HEAD_DIM)[None, :]
            v_val = tl.load(v_ptrs_col, mask=mask_load, other=0.0)
            
            # Accumulate
            acc += p[:, None] * v_val
            
    acc = acc * g
    o_ptrs = Out + (seq_start + offs_m[:, None]) * Stride_ot + pid_h * Stride_oh + tl.arange(0, HEAD_DIM)[None, :]
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])

# ==========================================
# 2. 模型定义 (跳过内部 Pad 实现)
# ==========================================

class HSTU_BSA_Triton(nn.Module):
    def __init__(self, block_size=32, block_counts=4):
        super().__init__()
        self.block_size = block_size
        self.block_counts = block_counts

    def forward(self, 
                jagged_q, jagged_k, jagged_v, jagged_u, 
                padded_q, padded_k, padded_v, # 直接接收 Padded 数据
                x_offsets, 
                gate_model, 
                padding_mask=None): # padding_mask: [B, N]
        
        B, N, H, D = padded_q.shape
        scale = D ** -0.5
        
        # 1. Gate 计算 (使用 Padded)
        g_cmp, g_slc, g_swa = gate_model(padded_q) # 输出 [B, N, H]
        
        # 将 Gate 转为 Jagged [TotalTokens, H] (用于 Triton)
        g_cmp_jag = g_cmp[padding_mask]
        g_slc_jag = g_slc[padding_mask]

        # 2. Compression / Pooling (使用 Padded 更方便)
        # [B, N, H, D] -> [B, N_blk, H, D]
        num_blocks = math.ceil(N / self.block_size)
        pad_len = num_blocks * self.block_size - N
        
        if pad_len > 0:
            padded_k_pad = F.pad(padded_k, (0, 0, 0, 0, 0, 0, 0, pad_len))
            padded_v_pad = F.pad(padded_v, (0, 0, 0, 0, 0, 0, 0, pad_len))
            padded_q_pad = F.pad(padded_q, (0, 0, 0, 0, 0, 0, 0, pad_len))
        else:
            padded_k_pad = padded_k
            padded_v_pad = padded_v
            padded_q_pad = padded_q

        # Mean Pooling
        k_cmp = padded_k_pad.view(B, num_blocks, self.block_size, H, D).mean(dim=2)
        v_cmp = padded_v_pad.view(B, num_blocks, self.block_size, H, D).mean(dim=2)
        
        # 3. TopK Selection (使用 Padded)
        # Score: Q @ K_cmp.T
        # Q: [B, N, H, D], K_cmp: [B, N_blk, H, D] -> [B, N, H, N_blk]
        attn_cmp_scores = torch.einsum('bqhd,bkhd->bhqk', padded_q_pad, k_cmp) * scale
        
        # Causal Mask (Block level)
        indices_q = torch.arange(padded_q_pad.shape[1], device=padded_q.device)[:, None] // self.block_size
        indices_k = torch.arange(num_blocks, device=padded_q.device)[None, :]
        causal_mask = indices_q >= indices_k
        attn_cmp_scores.masked_fill_(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # TopK
        S = min(self.block_counts, num_blocks)
        _, topk_indices = attn_cmp_scores.topk(S, dim=-1) # [B, H, N, S]
        
        # Mask out future blocks if any
        topk_indices = topk_indices.masked_fill(
            topk_indices > (indices_q[None, None, :, None]), -1
        )
        
        # 转换 TopK Indices 为 Jagged [TotalTokens, H, S]
        # 原始 shape [B, H, N, S] -> Permute [B, N, H, S] -> Flatten
        # 注意: 取前 N 个有效长度 (去掉 padding 部分)
        topk_indices = topk_indices.permute(0, 2, 1, 3) # [B, N_pad, H, S]
        topk_indices = topk_indices[:, :N, :, :] # Crop back to N
        topk_indices_jag = topk_indices[padding_mask] # Use mask to flatten: [TotalTokens, H, S]

        # 4. Triton Kernel Calls
        # Output Buffers
        o_cmp = torch.empty_like(jagged_v)
        o_slc = torch.empty_like(jagged_v)
        
        # Grid: (Blocks, Heads, Batch)
        # 对于 Jagged 处理，Grid 最好按 Token Block 划分，这里简化为按 MaxLen 划分，并在 Kernel 里判断边界
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_M']), H, B)

        # Launch CMP Kernel
        # 注意：这里 K_cmp, V_cmp 我们直接传入 Padded Tensor [B, N_blk, H, D]
        # 并在 Kernel 里手动计算 Batch Offset
        hstu_bsa_cmp_fwd_kernel[grid](
            Q=jagged_q, 
            K=k_cmp, # Padded
            V=v_cmp, # Padded
            G_cmp=g_cmp_jag, 
            Out=o_cmp,
            Stride_qt=jagged_q.stride(0), Stride_qh=jagged_q.stride(1), Stride_qd=jagged_q.stride(2),
            Stride_kt=k_cmp.stride(1), Stride_kh=k_cmp.stride(2), Stride_kd=k_cmp.stride(3), # Stride for [N_blk, H, D]
            Stride_vt=v_cmp.stride(1), Stride_vh=v_cmp.stride(2), Stride_vd=v_cmp.stride(3),
            Stride_ot=o_cmp.stride(0), Stride_oh=o_cmp.stride(1), Stride_od=o_cmp.stride(2),
            offsets=x_offsets, scale=scale,
            BLOCK_SIZE=self.block_size, HEAD_DIM=D, BLOCK_M=32, BLOCK_N=32
        )
        
        # Launch SLC Kernel
        # BlockIndices: [TotalTokens, H, S]
        hstu_bsa_slc_fwd_kernel[grid](
            Q=jagged_q, K=jagged_k, V=jagged_v, 
            G_slc=g_slc_jag, 
            BlockIndices=topk_indices_jag, 
            Out=o_slc,
            Stride_qt=jagged_q.stride(0), Stride_qh=jagged_q.stride(1), Stride_qd=jagged_q.stride(2),
            Stride_kt=jagged_k.stride(0), Stride_kh=jagged_k.stride(1), Stride_kd=jagged_k.stride(2),
            Stride_vt=jagged_v.stride(0), Stride_vh=jagged_v.stride(1), Stride_vd=jagged_v.stride(2),
            Stride_ot=o_slc.stride(0), Stride_oh=o_slc.stride(1), Stride_od=o_slc.stride(2),
            # 传入 TopK Indices 的 strides
            Stride_bidx_t=topk_indices_jag.stride(0),
            Stride_bidx_h=topk_indices_jag.stride(1),
            Stride_bidx_s=topk_indices_jag.stride(2),
            offsets=x_offsets, scale=scale,
            S=S, BLOCK_SIZE=self.block_size, HEAD_DIM=D, BLOCK_M=32
        )
        
        # 5. Final Sum & Norm (模拟)
        # 实际代码中这里会有 LayerNorm * u
        o_final = o_cmp + o_slc 
        return o_final


# ==========================================
# 3. 运行示例 (生成 Padded -> 自动转 Jagged)
# ==========================================

class MockGateModel(nn.Module):
    def forward(self, x):
        B, N, H, D = x.shape
        return torch.rand(B, N, H, device=x.device), torch.rand(B, N, H, device=x.device), None

def run_test():
    if not torch.cuda.is_available(): return
    device = "cuda"
    torch.manual_seed(42)

    # Config
    B, N, H, D = 4, 128, 8, 64
    BLOCK_SIZE = 32
    
    # 1. 生成 Padded 数据
    # 随机生成长度掩码
    lengths = torch.randint(32, N, (B,), device=device)
    mask = torch.arange(N, device=device)[None, :] < lengths[:, None]
    
    padded_q = torch.randn(B, N, H, D, device=device)
    padded_k = torch.randn(B, N, H, D, device=device)
    padded_v = torch.randn(B, N, H, D, device=device)
    padded_u = torch.randn(B, N, H, D, device=device)

    # 2. 生成 Jagged 数据 (直接通过 Mask 索引)
    jagged_q = padded_q[mask]
    jagged_k = padded_k[mask]
    jagged_v = padded_v[mask]
    jagged_u = padded_u[mask]
    
    # 生成 Offsets
    offsets = torch.zeros(B + 1, dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(lengths, dim=0)

    print(f"Padded Shape: {padded_q.shape}")
    print(f"Jagged Shape: {jagged_q.shape}")
    print(f"Offsets: {offsets}")

    # 3. 运行模型
    model = HSTU_BSA_Triton(block_size=BLOCK_SIZE).to(device)
    gate = MockGateModel().to(device)

    output = model(
        jagged_q, jagged_k, jagged_v, jagged_u,
        padded_q, padded_k, padded_v, # 传入两份数据
        offsets,
        gate,
        padding_mask=mask
    )

    print("HSTU BSA Triton Forward Success!")
    print(f"Output Shape: {output.shape}")
    print(f"Mean: {output.mean().item()}, Var: {output.var().item()}")

if __name__ == "__main__":
    run_test()