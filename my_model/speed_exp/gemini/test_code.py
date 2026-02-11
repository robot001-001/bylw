import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

# ==========================================
# 1. FBGEMM 替代算子 (解决报错的核心)
# ==========================================
class FBGEMM_Ops:
    @staticmethod
    def jagged_to_padded_dense(values, offsets, max_lengths, padding_value=0.0):
        """
        将 Jagged Tensor (Flattened) [Total_Tokens, ...] 转换为 Padded [B, MaxLen, ...]
        """
        offset_tensor = offsets[0]
        max_len = max_lengths[0]
        B = offset_tensor.size(0) - 1
        # values shape: [TotalTokens, H, D] or [TotalTokens, D]
        rest_shape = values.shape[1:] 
        device = values.device
        
        # 计算每个样本的长度
        seq_lengths = offset_tensor[1:] - offset_tensor[:-1]
        
        # 构建 Mask: [B, MaxLen]
        mask = torch.arange(max_len, device=device)[None, :] < seq_lengths[:, None]
        
        # 初始化输出
        out_shape = (B, max_len) + rest_shape
        out = torch.full(out_shape, padding_value, dtype=values.dtype, device=device)
        
        # 赋值
        out[mask] = values
        return out

    @staticmethod
    def dense_to_jagged(padded_dense, offsets):
        """
        将 Padded [B, MaxLen, ...] 转换回 Jagged [Total_Tokens, ...]
        """
        offset_tensor = offsets[0]
        max_len = padded_dense.shape[1]
        seq_lengths = offset_tensor[1:] - offset_tensor[:-1]
        
        mask = torch.arange(max_len, device=padded_dense.device)[None, :] < seq_lengths[:, None]
        values = padded_dense[mask]
        return [values] # 返回列表以保持接口一致

# ==========================================
# 2. Triton Kernels (HSTU Logic: SiLU based)
# ==========================================

@triton.jit
def _hstu_silu(x):
    # SiLU (Swish): x * sigmoid(x)
    return x * tl.sigmoid(x)

@triton.jit
def hstu_bsa_cmp_kernel(
    Q, K_desc, V_desc, 
    G_cmp, Out, 
    Offsets, 
    stride_q_t, stride_q_h, stride_q_d,
    stride_k_b, stride_k_blk, stride_k_h, stride_k_d, # K is padded [B, N_blk, H, D]
    stride_v_b, stride_v_blk, stride_v_h, stride_v_d,
    stride_o_t, stride_o_h, stride_o_d,
    scale,
    BLOCK_SIZE: tl.constexpr, # Compression Block Size (e.g. 32)
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, # Iterate step for compressed blocks
):
    """
    粗粒度注意力 Kernel (Compression Attention)
    O = SiLU(Q @ K_cmp.T) @ V_cmp * Gate
    """
    # Grid: (num_seqs * num_heads) -> 这里的映射比较简单，每个 program 处理一个 seq 的一个 head
    # 为了更细粒度的并行，我们改为 (Total_Blocks_M, H, B) 这里的 M 指的是 Q 的 token 块
    
    pid_m = tl.program_id(0) # Q 的分块索引
    pid_h = tl.program_id(1) # Head
    pid_b = tl.program_id(2) # Batch

    # 获取当前 Sequence 的 Offset
    seq_start = tl.load(Offsets + pid_b)
    seq_end = tl.load(Offsets + pid_b + 1)
    seq_len = seq_end - seq_start
    
    # 压缩后的长度 (用于 K_cmp, V_cmp)
    # 注意：K_cmp, V_cmp 是 Padded 的 [B, Max_Blk, H, D]
    # 我们需要计算当前 sequence 有多少个 compressed block
    n_cmp_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    # 当前 Q 处理的范围
    start_m_idx = pid_m * BLOCK_M
    if start_m_idx >= seq_len:
        return

    offs_m = start_m_idx + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_len
    
    # 指针计算: Q [TotalTokens, H, D]
    # base + (seq_start + offs_m) * stride_t + h * stride_h
    q_ptrs = Q + (seq_start + offs_m[:, None]) * stride_q_t + pid_h * stride_q_h + tl.arange(0, HEAD_DIM)[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # 指针计算: Gate [TotalTokens, H] (假设 stride_d=1)
    g_ptr = G_cmp + (seq_start + offs_m) * stride_q_t + pid_h * stride_q_h 
    g = tl.load(g_ptr, mask=mask_m, other=0.0)

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 循环遍历 Compressed K/V
    # K_desc shape: [B, Max_Blk, H, D]
    for start_n in range(0, n_cmp_blocks, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_cmp_blocks

        # Load K_cmp
        k_ptrs = K_desc + pid_b * stride_k_b + offs_n[None, :] * stride_k_blk + pid_h * stride_k_h + tl.arange(0, HEAD_DIM)[:, None]
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)

        # Attn Score: Q @ K.T
        score = tl.dot(q, k)
        score *= scale

        # Causal Mask (Block Level)
        # Q所在的原始块索引 >= K所在的压缩块索引
        q_blk_idx = offs_m[:, None] // BLOCK_SIZE
        k_blk_idx = offs_n[None, :]
        is_causal = q_blk_idx >= k_blk_idx
        
        # HSTU Activation: SiLU
        # mask out non-causal or padding
        score = tl.where(is_causal & mask_m[:, None] & mask_n[None, :], score, -1e9)
        # 注意: HSTU 不做 softmax，而是直接 SiLU。对于 masked 掉的部分，SiLU(-inf) -> 0
        # 但为了数值稳定性，我们显式 mask 结果
        p = _hstu_silu(score)
        p = tl.where(is_causal & mask_m[:, None] & mask_n[None, :], p, 0.0)

        # Load V_cmp
        v_ptrs = V_desc + pid_b * stride_v_b + offs_n[:, None] * stride_v_blk + pid_h * stride_v_h + tl.arange(0, HEAD_DIM)[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Accumulate
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16))

    # Apply Gate
    acc = acc * g[:, None]

    # Store
    o_ptrs = Out + (seq_start + offs_m[:, None]) * stride_o_t + pid_h * stride_o_h + tl.arange(0, HEAD_DIM)[None, :]
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def hstu_bsa_slc_kernel(
    Q, K, V, 
    G_slc, BlockIdx, Out, 
    Offsets,
    stride_q_t, stride_q_h, stride_q_d, # Q, K, V are Jagged [Total, H, D]
    stride_idx_t, stride_idx_h, stride_idx_s, # BlockIdx [Total, H, S]
    scale,
    S: tl.constexpr, # Num Selected Blocks
    BLOCK_SIZE: tl.constexpr, # Raw block size (e.g. 32)
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr
):
    """
    细粒度注意力 Kernel (Selection Attention)
    根据 BlockIndices Gather 原始 K/V 进行计算
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    seq_start = tl.load(Offsets + pid_b)
    seq_end = tl.load(Offsets + pid_b + 1)
    seq_len = seq_end - seq_start

    start_m_idx = pid_m * BLOCK_M
    if start_m_idx >= seq_len:
        return
    
    offs_m = start_m_idx + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_len

    # Load Q
    q_ptrs = Q + (seq_start + offs_m[:, None]) * stride_q_t + pid_h * stride_q_h + tl.arange(0, HEAD_DIM)[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Load Gate
    g_ptr = G_slc + (seq_start + offs_m) * stride_q_t + pid_h * stride_q_h
    g = tl.load(g_ptr, mask=mask_m, other=0.0)

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 遍历每个 Query Token 选中的 S 个块
    for s_i in range(S):
        # 读取 Block Indices [BLOCK_M]
        # Layout: [TotalTokens, H, S]
        b_idx_ptr = BlockIdx + (seq_start + offs_m) * stride_idx_t + pid_h * stride_idx_h + s_i * stride_idx_s
        b_idx = tl.load(b_idx_ptr, mask=mask_m, other=-1) # [BLOCK_M]

        # 遍历选中块内的每一个 Token (BLOCK_SIZE)
        # 注意：这里会产生 non-contiguous memory access，因为每个 query 选的块不同
        for blk_offset in range(BLOCK_SIZE):
            # 计算目标 Token 在原始 Flattened K/V 中的绝对索引
            # target = seq_start + block_id * BLOCK_SIZE + offset
            target_k_idx = seq_start + b_idx * BLOCK_SIZE + blk_offset
            
            # 有效性检查:
            # 1. b_idx != -1
            # 2. Causal Mask: target <= current_query_idx
            is_valid_blk = b_idx >= 0
            is_causal = target_k_idx <= (seq_start + offs_m)
            
            # Load K vector (Gather)
            # stride_q_t/h/d same for K/V usually
            k_ptrs_col = K + target_k_idx[:, None] * stride_q_t + pid_h * stride_q_h + tl.arange(0, HEAD_DIM)[None, :]
            mask_load = is_valid_blk[:, None] & mask_m[:, None]
            k_val = tl.load(k_ptrs_col, mask=mask_load, other=0.0)

            # Dot Product (Row-wise)
            # Q[i] dot K[target_i]
            score = tl.sum(q * k_val, axis=1)
            score *= scale

            # Activation
            p = _hstu_silu(score)
            p = tl.where(is_valid_blk & is_causal, p, 0.0)

            # Load V vector
            v_ptrs_col = V + target_k_idx[:, None] * stride_q_t + pid_h * stride_q_h + tl.arange(0, HEAD_DIM)[None, :]
            v_val = tl.load(v_ptrs_col, mask=mask_load, other=0.0)

            # Accumulate
            acc += p[:, None] * v_val

    acc = acc * g[:, None]
    
    o_ptrs = Out + (seq_start + offs_m[:, None]) * stride_q_t + pid_h * stride_q_h + tl.arange(0, HEAD_DIM)[None, :]
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])


# ==========================================
# 3. Model Implementation
# ==========================================

class HSTU_BSA_Layer(nn.Module):
    def __init__(self, num_heads, head_dim, block_size=32, block_counts=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.block_counts = block_counts

    def forward(self, q, k, v, u, x_offsets, gate_model):
        """
        q, k, v, u: Jagged [TotalTokens, H, D]
        x_offsets: [B+1]
        gate_model: nn.Module
        """
        B = x_offsets.size(0) - 1
        total_tokens = q.shape[0]
        # 计算最大序列长度
        seq_lengths = x_offsets[1:] - x_offsets[:-1]
        max_seq_len = seq_lengths.max().item()
        
        scale = self.head_dim ** -0.5

        # 1. 转换 Jagged -> Padded
        padded_q = FBGEMM_Ops.jagged_to_padded_dense(q, [x_offsets], [max_seq_len]) # [B, N, H, D]
        padded_k = FBGEMM_Ops.jagged_to_padded_dense(k, [x_offsets], [max_seq_len])
        padded_v = FBGEMM_Ops.jagged_to_padded_dense(v, [x_offsets], [max_seq_len])

        # 2. 运行 Gate Model
        g_cmp, g_slc, g_swa = gate_model(padded_q) 

        # 将 Gate 转回 Jagged (用于 Triton Kernel)
        g_cmp_jag = FBGEMM_Ops.dense_to_jagged(g_cmp.unsqueeze(-1), [x_offsets])[0].squeeze(-1) 
        g_slc_jag = FBGEMM_Ops.dense_to_jagged(g_slc.unsqueeze(-1), [x_offsets])[0].squeeze(-1)

        # 3. Compression (Mean Pooling)
        # Pad sequence to multiple of block_size
        num_blocks = math.ceil(max_seq_len / self.block_size)
        pad_len = num_blocks * self.block_size - max_seq_len
        
        # --- 修正点开始 ---
        if pad_len > 0:
            # 修正 F.pad 参数：从后往前数，只需要填充 3 个维度 (D, H, N)
            # (D_left, D_right, H_left, H_right, N_left, N_right)
            pad_params = (0, 0, 0, 0, 0, pad_len)
            padded_k_p = F.pad(padded_k, pad_params)
            padded_v_p = F.pad(padded_v, pad_params)
        else:
            padded_k_p = padded_k
            padded_v_p = padded_v
        # --- 修正点结束 ---
        
        # 现在 View 操作应该可以正常运行了
        # [B, N_padded, H, D] -> [B, N_blk, BS, H, D] -> mean -> [B, N_blk, H, D]
        k_cmp = padded_k_p.view(B, num_blocks, self.block_size, self.num_heads, self.head_dim).mean(dim=2)
        v_cmp = padded_v_p.view(B, num_blocks, self.block_size, self.num_heads, self.head_dim).mean(dim=2)

        # 4. TopK Selection
        attn_scores = torch.einsum('bnhd,bmhd->bnhm', padded_q, k_cmp) * scale
        
        q_idx = torch.arange(max_seq_len, device=q.device)[:, None] // self.block_size
        k_idx = torch.arange(num_blocks, device=q.device)[None, :]
        causal_mask = q_idx >= k_idx 
        attn_scores.masked_fill_(~causal_mask.unsqueeze(0).unsqueeze(2), float('-inf'))

        S = min(self.block_counts, num_blocks)
        _, topk_indices = attn_scores.topk(S, dim=-1)
        
        topk_indices_jag = FBGEMM_Ops.dense_to_jagged(
            topk_indices.view(B, max_seq_len, -1), 
            [x_offsets]
        )[0].view(-1, self.num_heads, S).contiguous()

        # 5. Launch Triton Kernels
        o_cmp = torch.empty_like(q)
        o_slc = torch.empty_like(q)

        # 5.1 Compression Kernel
        grid_cmp = (triton.cdiv(max_seq_len, 32), self.num_heads, B)
        hstu_bsa_cmp_kernel[grid_cmp](
            Q=q, K_desc=k_cmp, V_desc=v_cmp,
            G_cmp=g_cmp_jag, Out=o_cmp, Offsets=x_offsets,
            stride_q_t=q.stride(0), stride_q_h=q.stride(1), stride_q_d=q.stride(2),
            stride_k_b=k_cmp.stride(0), stride_k_blk=k_cmp.stride(1), stride_k_h=k_cmp.stride(2), stride_k_d=k_cmp.stride(3),
            stride_v_b=v_cmp.stride(0), stride_v_blk=v_cmp.stride(1), stride_v_h=v_cmp.stride(2), stride_v_d=v_cmp.stride(3),
            stride_o_t=o_cmp.stride(0), stride_o_h=o_cmp.stride(1), stride_o_d=o_cmp.stride(2),
            scale=scale,
            BLOCK_SIZE=self.block_size, HEAD_DIM=self.head_dim, BLOCK_M=32, BLOCK_N=32
        )

        # 5.2 Selection Kernel
        grid_slc = (triton.cdiv(max_seq_len, 32), self.num_heads, B)
        hstu_bsa_slc_kernel[grid_slc](
            Q=q, K=k, V=v,
            G_slc=g_slc_jag, BlockIdx=topk_indices_jag, Out=o_slc,
            Offsets=x_offsets,
            stride_q_t=q.stride(0), stride_q_h=q.stride(1), stride_q_d=q.stride(2),
            stride_idx_t=topk_indices_jag.stride(0), stride_idx_h=topk_indices_jag.stride(1), stride_idx_s=topk_indices_jag.stride(2),
            scale=scale,
            S=S, BLOCK_SIZE=self.block_size, HEAD_DIM=self.head_dim, BLOCK_M=32
        )

        # 6. Epilogue
        # 修复点: 先 view 成 [TotalTokens, H*D] 再做 LayerNorm
        hidden_size = self.num_heads * self.head_dim
        
        # 处理 o_cmp
        o_cmp_flat = o_cmp.view(total_tokens, hidden_size)
        o_cmp = F.layer_norm(o_cmp_flat, [hidden_size])
        o_cmp = o_cmp.view(total_tokens, self.num_heads, self.head_dim) * u

        # 处理 o_slc
        o_slc_flat = o_slc.view(total_tokens, hidden_size)
        o_slc = F.layer_norm(o_slc_flat, [hidden_size])
        o_slc = o_slc.view(total_tokens, self.num_heads, self.head_dim) * u

        return o_cmp + o_slc

# ==========================================
# 4. Mock & Run
# ==========================================

class MockGate(nn.Module):
    def forward(self, x):
        B, N, H, D = x.shape
        return torch.rand(B, N, H, device=x.device), \
               torch.rand(B, N, H, device=x.device), \
               torch.rand(B, N, H, device=x.device)

def test_run():
    if not torch.cuda.is_available():
        print("Need CUDA for Triton")
        return

    device = "cuda"
    B, H, D = 4, 8, 64
    max_len = 128
    
    # 构造数据
    import random
    lengths = [random.randint(64, max_len) for _ in range(B)]
    offsets = [0]
    for l in lengths: offsets.append(offsets[-1] + l)
    x_offsets = torch.tensor(offsets, dtype=torch.long, device=device)
    total_tokens = offsets[-1]

    q = torch.randn(total_tokens, H, D, device=device)
    k = torch.randn(total_tokens, H, D, device=device)
    v = torch.randn(total_tokens, H, D, device=device)
    u = torch.randn(total_tokens, H, D, device=device)

    model = HSTU_BSA_Layer(H, D, block_size=32, block_counts=4).to(device)
    gate = MockGate().to(device)

    print(f"Start Run. Batch: {B}, Total Tokens: {total_tokens}, Heads: {H}, Dim: {D}")
    try:
        out = model(q, k, v, u, x_offsets, gate)
        print(f"Success! Output shape: {out.shape}")
        print(f"Mean: {out.mean().item()}, Var: {out.var().item()}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_run()