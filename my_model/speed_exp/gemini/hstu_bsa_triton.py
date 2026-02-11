import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def _hstu_silu_activation(x):
    return x * tl.sigmoid(x)

@triton.jit
def hstu_bsa_cmp_fwd_kernel(
    Q, K, V, 
    G_cmp, 
    Out,
    Stride_qt, Stride_qh, Stride_qd,
    Stride_kt, Stride_kh, Stride_kd,
    Stride_vt, Stride_vh, Stride_vd,
    Stride_ot, Stride_oh, Stride_od,
    offsets,      # [B+1] 原始序列的 offsets
    offsets_cmp,  # [B+1] 压缩序列的 offsets (新增)
    scale,
    BLOCK_SIZE: tl.constexpr, 
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,    
    BLOCK_N: tl.constexpr,    
):
    """
    支持 Jagged K_cmp/V_cmp 的压缩注意力 Kernel
    """
    pid_m = tl.program_id(0) 
    pid_h = tl.program_id(1) 
    pid_z = tl.program_id(2) 

    # --- 1. 定位原始序列 Q ---
    seq_start = tl.load(offsets + pid_z)
    seq_end = tl.load(offsets + pid_z + 1)
    seq_len = seq_end - seq_start
    
    # 当前 Q 处理的块
    start_m = pid_m * BLOCK_M
    if start_m >= seq_len:
        return

    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_len
    
    # Load Q: [BLOCK_M, HEAD_DIM]
    # Q 是 Jagged [TotalTokens, H, D]
    q_ptrs = Q + (seq_start + offs_m[:, None]) * Stride_qt + pid_h * Stride_qh + tl.arange(0, HEAD_DIM)[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # Load G_cmp: [BLOCK_M, 1]
    # G_cmp 也是 Jagged [TotalTokens, H]
    g_ptrs = G_cmp + (seq_start + offs_m[:, None]) * Stride_qt + pid_h * Stride_qh 
    g = tl.load(g_ptrs, mask=mask_m[:, None], other=0.0)

    # --- 2. 定位压缩序列 K/V ---
    # 使用 offsets_cmp 获取当前 batch 在压缩 buffer 中的起始位置
    cmp_start = tl.load(offsets_cmp + pid_z)
    cmp_end = tl.load(offsets_cmp + pid_z + 1)
    cmp_len = cmp_end - cmp_start # 该样本压缩后的长度

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # Loop over Compressed K/V
    for start_n in range(0, cmp_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < cmp_len
        
        # Load Jagged K_cmp: [BLOCK_N, HEAD_DIM]
        # 指针计算：Base + (cmp_start + offs_n) * stride
        k_ptrs = K + (cmp_start + offs_n[None, :]) * Stride_kt + pid_h * Stride_kh + tl.arange(0, HEAD_DIM)[:, None]
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        
        # Q @ K.T
        attn_score = tl.dot(q, k)
        attn_score *= scale
        
        # Causal Masking
        # 原始位置 // Block_Size >= 压缩位置
        mask_causal = (offs_m[:, None] // BLOCK_SIZE) >= offs_n[None, :]
        attn_score = tl.where(mask_causal & mask_m[:, None] & mask_n[None, :], attn_score, -1e9)
        
        # Activation
        p = _hstu_silu_activation(attn_score)
        p = tl.where(mask_causal & mask_n[None, :], p, 0.0)
        
        # Load Jagged V_cmp
        v_ptrs = V + (cmp_start + offs_n[:, None]) * Stride_vt + pid_h * Stride_vh + tl.arange(0, HEAD_DIM)[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Accumulate
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16))

    # Apply Gating
    acc = acc * g
    
    # Store Output
    o_ptrs = Out + (seq_start + offs_m[:, None]) * Stride_ot + pid_h * Stride_oh + tl.arange(0, HEAD_DIM)[None, :]
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])

# SLC Kernel 逻辑基本不变，只需确保传入正确的 Stride 即可兼容 Jagged G_slc
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

    # Load Q
    q_ptrs = Q + (seq_start + offs_m[:, None]) * Stride_qt + pid_h * Stride_qh + tl.arange(0, HEAD_DIM)[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Load G_slc (Jagged)
    g_ptrs = G_slc + (seq_start + offs_m[:, None]) * Stride_qt + pid_h * Stride_qh
    g = tl.load(g_ptrs, mask=mask_m[:, None], other=0.0)

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 遍历选中的 S 个块
    for s_idx in range(S):
        # Load BlockIndices [TotalTokens, H, S]
        # 这里假设 BlockIndices 是紧密的 jagged 布局
        b_idxs_ptr = BlockIndices + (seq_start + offs_m) * (S * tl.num_programs(1)) + pid_h * S + s_idx
        b_idx = tl.load(b_idxs_ptr, mask=mask_m, other=-1) 

        # Inner loop over block_size (gather K/V)
        for blk_offset in range(BLOCK_SIZE):
            valid_blk = b_idx >= 0
            # 计算绝对 token index: seq_start + block_id * BS + offset
            target_k_idx = seq_start + b_idx * BLOCK_SIZE + blk_offset
            
            is_causal = target_k_idx <= (seq_start + offs_m)
            
            # Load K column (Jagged Gather)
            k_ptrs_col = K + target_k_idx[:, None] * Stride_kt + pid_h * Stride_kh + tl.arange(0, HEAD_DIM)[None, :]
            mask_load = valid_blk[:, None] & mask_m[:, None]
            k_val = tl.load(k_ptrs_col, mask=mask_load, other=0.0)
            
            score = tl.sum(q * k_val, axis=1) 
            score *= scale
            
            p = _hstu_silu_activation(score)
            p = tl.where(valid_blk & is_causal, p, 0.0)
            
            # Load V column
            v_ptrs_col = V + target_k_idx[:, None] * Stride_vt + pid_h * Stride_vh + tl.arange(0, HEAD_DIM)[None, :]
            v_val = tl.load(v_ptrs_col, mask=mask_load, other=0.0)
            
            acc += p[:, None] * v_val
            
    acc = acc * g
    
    o_ptrs = Out + (seq_start + offs_m[:, None]) * Stride_ot + pid_h * Stride_oh + tl.arange(0, HEAD_DIM)[None, :]
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])



class HSTU_BSA_Triton(torch.nn.Module):
    def __init__(self, block_size=32, block_counts=4):
        super().__init__()
        self.block_size = block_size
        self.block_counts = block_counts

    def forward(self, 
                q, k, v, 
                g_cmp, g_slc, 
                x_offsets):
        """
        Args:
            q, k, v: Jagged [TotalTokens, H, D]
            g_cmp, g_slc: Jagged [TotalTokens, H, 1]
            x_offsets: [B+1]
        """
        # 维度检查与调整
        if g_cmp.dim() == 2: g_cmp = g_cmp.unsqueeze(-1)
        if g_slc.dim() == 2: g_slc = g_slc.unsqueeze(-1)
        
        # 获取基本信息
        B = x_offsets.size(0) - 1
        seq_lens = x_offsets[1:] - x_offsets[:-1]
        total_tokens = q.shape[0]
        device = q.device
        
        # --- 1. 准备 Jagged Pooling 索引 (纯 PyTorch 实现) ---
        with torch.no_grad():
            # 计算压缩后的长度信息
            cmp_seq_lens = (seq_lens + self.block_size - 1) // self.block_size
            offsets_cmp = torch.zeros_like(x_offsets)
            offsets_cmp[1:] = torch.cumsum(cmp_seq_lens, dim=0)
            total_cmp_tokens = offsets_cmp[-1].item()
            
            # 生成 Batch ID 和 Local Token ID
            # batch_ids: [0, 0, ..., 1, 1, ...] 标记每个 token 属于哪个 batch
            batch_ids = torch.repeat_interleave(torch.arange(B, device=device), seq_lens)
            
            # local_ids: [0, 1, 2, ..., 0, 1, ...] 标记每个 token 在其序列中的位置
            # 技巧: 全局索引 - 该 batch 的起始 offset
            local_ids = torch.arange(total_tokens, device=device) - x_offsets[:-1][batch_ids]
            
            # 计算对应的 Block ID
            local_block_ids = local_ids // self.block_size
            
            # 计算 Pooling 用的 segment_ids (将 token 映射到压缩后的全局索引)
            segment_ids = offsets_cmp[:-1][batch_ids] + local_block_ids

        # --- 2. Jagged Pooling (Mean) ---
        # K, V: [TotalTokens, H, D] -> [TotalCmpTokens, H, D]
        # 使用 index_add_ 实现求和
        k_cmp = torch.zeros((total_cmp_tokens, k.shape[1], k.shape[2]), 
                            dtype=k.dtype, device=device)
        v_cmp = torch.zeros((total_cmp_tokens, v.shape[1], v.shape[2]), 
                            dtype=v.dtype, device=device)
        
        k_cmp.index_add_(0, segment_ids, k)
        v_cmp.index_add_(0, segment_ids, v)
        
        # 计算每个 compressed block 包含多少个 token (处理 padding/边缘)
        ones = torch.ones(total_tokens, 1, 1, device=device, dtype=k.dtype)
        counts = torch.zeros((total_cmp_tokens, 1, 1), device=device, dtype=k.dtype)
        counts.index_add_(0, segment_ids, ones)
        
        # 求平均
        k_cmp = k_cmp / counts.clamp(min=1.0)
        v_cmp = v_cmp / counts.clamp(min=1.0)

        # --- 3. Coarse Attention & TopK ---
        # 为了计算 TopK 分数，我们需要 [B, N, NumBlocks] 的形式。
        # 这里临时转为 Padded View 是最高效的。
        
        max_n = seq_lens.max().item()
        max_blocks = cmp_seq_lens.max().item()
        num_heads = q.shape[1]
        dim = q.shape[2]
        
        # [Helper] Jagged -> Padded (替换 fbgemm.jagged_to_padded_dense)
        # 利用上面计算好的 batch_ids 和 local_ids 直接赋值
        padded_q = torch.zeros(B, max_n, num_heads, dim, device=device, dtype=q.dtype)
        padded_q[batch_ids, local_ids] = q
        
        padded_k_cmp = torch.zeros(B, max_blocks, num_heads, dim, device=device, dtype=k_cmp.dtype)
        # 为 k_cmp 生成索引
        batch_ids_cmp = torch.repeat_interleave(torch.arange(B, device=device), cmp_seq_lens)
        local_ids_cmp = torch.arange(total_cmp_tokens, device=device) - offsets_cmp[:-1][batch_ids_cmp]
        padded_k_cmp[batch_ids_cmp, local_ids_cmp] = k_cmp
        
        scale = dim ** -0.5
        # 计算分数: [B, H, N, S_blocks]
        attn_cmp_scores = torch.einsum('bqhd,bkhd->bhqk', padded_q, padded_k_cmp) * scale
        
        # Causal Masking
        indices_q = torch.arange(max_n, device=device)[:, None] // self.block_size
        indices_k = torch.arange(max_blocks, device=device)[None, :]
        causal_mask = indices_q >= indices_k
        attn_cmp_scores.masked_fill_(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # TopK Selection
        S = min(self.block_counts, max_blocks)
        _, topk_indices = attn_cmp_scores.topk(S, dim=-1) # [B, H, N, S]
        
        # [Helper] Padded -> Jagged (替换 fbgemm.dense_to_jagged)
        # 只有在有效 mask 内的索引才是需要的
        # 构造有效 mask [B, N]
        valid_mask = torch.arange(max_n, device=device)[None, :] < seq_lens[:, None]
        # 使用 boolean masking 展平: [B, H, N, S] -> select -> [TotalTokens, H, S]
        # permute 把 N 放到前面方便 mask
        topk_indices = topk_indices.permute(0, 2, 1, 3) # [B, N, H, S]
        topk_indices_jag = topk_indices[valid_mask].contiguous().view(-1, num_heads, S).int()

        # --- 4. Triton Kernel Launches ---
        
        # Output buffers
        o_cmp = torch.empty_like(v)
        o_slc = torch.empty_like(v)
        
        # Grid: (NumBlocks_Q, Heads, Batch)
        grid_triton = lambda meta: (triton.cdiv(max_n, meta['BLOCK_M']), num_heads, B)

        hstu_bsa_cmp_fwd_kernel[grid_triton](
            Q=q, K=k_cmp, V=v_cmp, 
            G_cmp=g_cmp.squeeze(-1), 
            Out=o_cmp,
            Stride_qt=q.stride(0), Stride_qh=q.stride(1), Stride_qd=q.stride(2),
            Stride_kt=k_cmp.stride(0), Stride_kh=k_cmp.stride(1), Stride_kd=k_cmp.stride(2),
            Stride_vt=v_cmp.stride(0), Stride_vh=v_cmp.stride(1), Stride_vd=v_cmp.stride(2),
            Stride_ot=o_cmp.stride(0), Stride_oh=o_cmp.stride(1), Stride_od=o_cmp.stride(2),
            offsets=x_offsets, 
            offsets_cmp=offsets_cmp, 
            scale=scale,
            BLOCK_SIZE=self.block_size, HEAD_DIM=dim,
            BLOCK_M=32, BLOCK_N=32
        )

        hstu_bsa_slc_fwd_kernel[grid_triton](
            Q=q, K=k, V=v, 
            G_slc=g_slc.squeeze(-1), 
            BlockIndices=topk_indices_jag, 
            Out=o_slc,
            Stride_qt=q.stride(0), Stride_qh=q.stride(1), Stride_qd=q.stride(2),
            Stride_kt=k.stride(0), Stride_kh=k.stride(1), Stride_kd=k.stride(2),
            Stride_vt=v.stride(0), Stride_vh=v.stride(1), Stride_vd=v.stride(2),
            Stride_ot=o_slc.stride(0), Stride_oh=o_slc.stride(1), Stride_od=o_slc.stride(2),
            offsets=x_offsets, 
            scale=scale,
            S=S, BLOCK_SIZE=self.block_size, HEAD_DIM=dim, BLOCK_M=32
        )
        
        return o_cmp, o_slc