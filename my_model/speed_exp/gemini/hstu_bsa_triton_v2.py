import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _hstu_silu_activation(x):
    return x * tl.sigmoid(x)

# -----------------------------------------------------------------------------
# [Fix] TopK Kernel (Pure Tensor Arithmetic - Most Robust)
# -----------------------------------------------------------------------------
@triton.jit
# def hstu_bsa_topk_kernel(
#     Q, K, 
#     Out_Indices, 
#     Stride_qt, Stride_qh, Stride_qd,
#     Stride_kt, Stride_kh, Stride_kd,
#     Stride_idx_t, Stride_idx_h, Stride_idx_s,
#     offsets,      
#     offsets_cmp,  
#     scale,
#     S: tl.constexpr,          # 逻辑 TopK 数量
#     BLOCK_N_PAD: tl.constexpr,# 物理计算维度 (>=16)
#     BLOCK_SIZE: tl.constexpr, 
#     HEAD_DIM: tl.constexpr,
#     BLOCK_M: tl.constexpr,    
# ):
#     pid_m = tl.program_id(0) 
#     pid_h = tl.program_id(1) 
#     pid_z = tl.program_id(2) 

#     seq_start = tl.load(offsets + pid_z)
#     seq_end = tl.load(offsets + pid_z + 1)
#     seq_len = seq_end - seq_start
    
#     start_m = pid_m * BLOCK_M
#     if start_m >= seq_len:
#         return

#     offs_m = start_m + tl.arange(0, BLOCK_M)
#     mask_m = offs_m < seq_len
    
#     q_ptrs = Q + (seq_start + offs_m[:, None]) * Stride_qt + pid_h * Stride_qh + tl.arange(0, HEAD_DIM)[None, :]
#     q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0) 

#     # -----------------------------------------------------------
#     # 初始化 Top S Tensor (Single Tensor, No List)
#     # -----------------------------------------------------------
#     top_vals = tl.full([BLOCK_M, S], float('-inf'), dtype=tl.float32)
#     top_idxs = tl.full([BLOCK_M, S], -1, dtype=tl.int32)

#     cmp_start = tl.load(offsets_cmp + pid_z)
#     cmp_end = tl.load(offsets_cmp + pid_z + 1)
#     cmp_len = cmp_end - cmp_start 

#     # -----------------------------------------------------------
#     # Stream Processing
#     # -----------------------------------------------------------
#     for start_n in range(0, cmp_len, S):
#         # 1. Load K (Pad to BLOCK_N_PAD >= 16)
#         offs_n_pad = start_n + tl.arange(0, BLOCK_N_PAD)
#         mask_n_valid = (offs_n_pad < cmp_len) & (offs_n_pad < (start_n + S))
        
#         k_ptrs = K + (cmp_start + offs_n_pad[None, :]) * Stride_kt + pid_h * Stride_kh + tl.arange(0, HEAD_DIM)[:, None]
        
#         # [Fix] 变量名不使用 k，改用 k_tensor
#         k_tensor = tl.load(k_ptrs, mask=mask_n_valid[None, :], other=0.0)
        
#         # 2. Compute Score
#         scores = tl.dot(q, k_tensor) * scale
        
#         # 3. Masking
#         mask_causal = (offs_m[:, None] // BLOCK_SIZE) >= offs_n_pad[None, :]
#         mask_final = mask_causal & mask_n_valid[None, :] & mask_m[:, None]
#         scores = tl.where(mask_final, scores, float('-inf'))

#         # -------------------------------------------------------
#         # Bubble Insertion (Pure Tensor Ops)
#         # -------------------------------------------------------
#         # 使用 constexpr loop 展开
#         for col in range(S):
#             # A. Extract Value/Index (Arithmetic Gather)
#             # col_mask: [BLOCK_N_PAD]
#             col_mask = (tl.arange(0, BLOCK_N_PAD) == col) 
            
#             # [BLOCK_M, BLOCK_N_PAD] * [BLOCK_N_PAD] -> Sum -> [BLOCK_M]
#             # 提取 scores 的第 col 列
#             val = tl.sum(scores * col_mask[None, :], axis=1)
            
#             current_idx_scalar = start_n + col
#             idx = tl.full([BLOCK_M], current_idx_scalar, dtype=tl.int32)
            
#             # B. Insert into Top Tensor
#             # [Fix] 循环变量不使用 k，改用 rank_idx
#             for rank_idx in range(S):
#                 # Arithmetic Gather: Extract column rank_idx from top_vals
#                 rank_mask = (tl.arange(0, S) == rank_idx) # [S]
                
#                 # top_vals: [BLOCK_M, S] * rank_mask[None, :] -> 此时只有第 rank_idx 列非零
#                 # sum(axis=1) -> 提取出第 rank_idx 列的值 [BLOCK_M]
#                 old_val = tl.sum(top_vals * rank_mask[None, :], axis=1)
#                 old_idx = tl.sum(top_idxs * rank_mask[None, :], axis=1)
                
#                 # Compare
#                 swap = val > old_val
                
#                 # Prepare values to write
#                 new_col_val = tl.where(swap, val, old_val)
#                 new_col_idx = tl.where(swap, idx, old_idx)
                
#                 # Arithmetic Scatter: Update column rank_idx in top_vals
#                 # 如果是这一列(rank_mask=True)，写入 new_col_val；否则保持 top_vals 原值
#                 # new_col_val[:, None] -> [BLOCK_M, 1]
#                 top_vals = tl.where(rank_mask[None, :], new_col_val[:, None], top_vals)
#                 top_idxs = tl.where(rank_mask[None, :], new_col_idx[:, None], top_idxs)
                
#                 # Update val/idx to carry over (Push out)
#                 val = tl.where(swap, old_val, val)
#                 idx = tl.where(swap, old_idx, idx)

#     # -----------------------------------------------------------
#     # Store Indices
#     # -----------------------------------------------------------
#     for rank_idx in range(S):
#         # Arithmetic Gather to store
#         rank_mask = (tl.arange(0, S) == rank_idx)
#         final_idx = tl.sum(top_idxs * rank_mask[None, :], axis=1)
        
#         idx_ptrs = Out_Indices + (seq_start + offs_m) * Stride_idx_t + pid_h * Stride_idx_h + rank_idx * Stride_idx_s
#         tl.store(idx_ptrs, final_idx, mask=mask_m)


@triton.jit
def hstu_bsa_topk_kernel(
    Q, K, 
    Out_Indices, 
    Stride_qt, Stride_qh, Stride_qd,
    Stride_kt, Stride_kh, Stride_kd,
    Stride_idx_t, Stride_idx_h, Stride_idx_s,
    offsets,      
    offsets_cmp,  
    scale,
    S: tl.constexpr,          
    BLOCK_N_PAD: tl.constexpr,
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
    
    q_ptrs = Q + (seq_start + offs_m[:, None]) * Stride_qt + pid_h * Stride_qh + tl.arange(0, HEAD_DIM)[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0) 

    # -----------------------------------------------------------
    # 初始化 Top S Tensor
    # -----------------------------------------------------------
    top_vals = tl.full([BLOCK_M, S], float('-inf'), dtype=tl.float32)
    top_idxs = tl.full([BLOCK_M, S], -1, dtype=tl.int32)

    cmp_start = tl.load(offsets_cmp + pid_z)
    cmp_end = tl.load(offsets_cmp + pid_z + 1)
    cmp_len = cmp_end - cmp_start 

    # -----------------------------------------------------------
    # Stream Processing
    # -----------------------------------------------------------
    for start_n in range(0, cmp_len, S):
        offs_n_pad = start_n + tl.arange(0, BLOCK_N_PAD)
        mask_n_valid = (offs_n_pad < cmp_len) & (offs_n_pad < (start_n + S))
        
        k_ptrs = K + (cmp_start + offs_n_pad[None, :]) * Stride_kt + pid_h * Stride_kh + tl.arange(0, HEAD_DIM)[:, None]
        k_tensor = tl.load(k_ptrs, mask=mask_n_valid[None, :], other=0.0)
        
        scores = tl.dot(q, k_tensor) * scale
        
        mask_causal = (offs_m[:, None] // BLOCK_SIZE) >= offs_n_pad[None, :]
        mask_final = mask_causal & mask_n_valid[None, :] & mask_m[:, None]
        scores = tl.where(mask_final, scores, float('-inf'))

        # -------------------------------------------------------
        # Bubble Insertion (NaN-Safe)
        # -------------------------------------------------------
        for col in range(S):
            # [Fix 1] 使用 where 避免 inf * 0 = NaN
            col_mask = (tl.arange(0, BLOCK_N_PAD) == col) 
            # 如果 mask 是 0，强制值为 0.0 (避免引入 NaN)
            # 如果 mask 是 1，保留原值 (可能是 -inf)
            safe_scores = tl.where(col_mask[None, :], scores, 0.0)
            val = tl.sum(safe_scores, axis=1)
            
            current_idx_scalar = start_n + col
            idx = tl.full([BLOCK_M], current_idx_scalar, dtype=tl.int32)
            
            for rank_idx in range(S):
                rank_mask = (tl.arange(0, S) == rank_idx) # [S]
                
                # [Fix 2] 同样对 top_vals 使用 where
                safe_top_vals = tl.where(rank_mask[None, :], top_vals, 0.0)
                old_val = tl.sum(safe_top_vals, axis=1)
                
                # int32 没有 NaN 问题，乘法是安全的，但为了统一也用 where
                safe_top_idxs = tl.where(rank_mask[None, :], top_idxs, 0)
                old_idx = tl.sum(safe_top_idxs, axis=1)
                
                # Compare & Swap
                swap = val > old_val
                
                new_col_val = tl.where(swap, val, old_val)
                new_col_idx = tl.where(swap, idx, old_idx)
                
                # Update State
                top_vals = tl.where(rank_mask[None, :], new_col_val[:, None], top_vals)
                top_idxs = tl.where(rank_mask[None, :], new_col_idx[:, None], top_idxs)
                
                # Push out
                val = tl.where(swap, old_val, val)
                idx = tl.where(swap, old_idx, idx)

    # -----------------------------------------------------------
    # Store Indices
    # -----------------------------------------------------------
    for rank_idx in range(S):
        rank_mask = (tl.arange(0, S) == rank_idx)
        # [Fix 3] Store Gather
        safe_top_idxs = tl.where(rank_mask[None, :], top_idxs, 0)
        final_idx = tl.sum(safe_top_idxs, axis=1)
        
        idx_ptrs = Out_Indices + (seq_start + offs_m) * Stride_idx_t + pid_h * Stride_idx_h + rank_idx * Stride_idx_s
        tl.store(idx_ptrs, final_idx, mask=mask_m)
# -----------------------------------------------------------------------------
# CMP / SLC Kernels (保持不变)
# -----------------------------------------------------------------------------
@triton.jit
def hstu_bsa_cmp_fwd_kernel(Q, K, V, G_cmp, Out, Stride_qt, Stride_qh, Stride_qd, Stride_kt, Stride_kh, Stride_kd, Stride_vt, Stride_vh, Stride_vd, Stride_ot, Stride_oh, Stride_od, Stride_gt, Stride_gh, offsets, offsets_cmp, scale, BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m, pid_h, pid_z = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq_start = tl.load(offsets + pid_z)
    seq_end = tl.load(offsets + pid_z + 1)
    if pid_m * BLOCK_M >= (seq_end - seq_start): return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < (seq_end - seq_start)
    q = tl.load(Q + (seq_start + offs_m[:, None])*Stride_qt + pid_h*Stride_qh + tl.arange(0, HEAD_DIM)[None, :], mask=mask_m[:, None], other=0.0)
    g = tl.load(G_cmp + (seq_start + offs_m[:, None])*Stride_gt + pid_h*Stride_gh, mask=mask_m[:, None], other=0.0)
    cmp_start = tl.load(offsets_cmp + pid_z)
    cmp_len = tl.load(offsets_cmp + pid_z + 1) - cmp_start
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, cmp_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < cmp_len
        k = tl.load(K + (cmp_start + offs_n[None, :])*Stride_kt + pid_h*Stride_kh + tl.arange(0, HEAD_DIM)[:, None], mask=mask_n[None, :], other=0.0)
        v = tl.load(V + (cmp_start + offs_n[:, None])*Stride_vt + pid_h*Stride_vh + tl.arange(0, HEAD_DIM)[None, :], mask=mask_n[:, None], other=0.0)
        score = tl.dot(q, k) * scale
        mask_causal = (offs_m[:, None] // BLOCK_SIZE) >= offs_n[None, :]
        score = tl.where(mask_causal & mask_n[None, :] & mask_m[:, None], score, -1e9)
        p = _hstu_silu_activation(score)
        p = tl.where(mask_causal & mask_n[None, :], p, 0.0)
        acc += tl.dot(p, v)
    tl.store(Out + (seq_start + offs_m[:, None])*Stride_ot + pid_h*Stride_oh + tl.arange(0, HEAD_DIM)[None, :], (acc * g).to(Out.dtype.element_ty), mask=mask_m[:, None])

@triton.jit
def hstu_bsa_slc_fwd_kernel(Q, K, V, G_slc, BlockIndices, Out, Stride_qt, Stride_qh, Stride_qd, Stride_kt, Stride_kh, Stride_kd, Stride_vt, Stride_vh, Stride_vd, Stride_ot, Stride_oh, Stride_od, Stride_gt, Stride_gh, offsets, scale, S: tl.constexpr, BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_m, pid_h, pid_z = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq_start = tl.load(offsets + pid_z)
    seq_end = tl.load(offsets + pid_z + 1)
    if pid_m * BLOCK_M >= (seq_end - seq_start): return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < (seq_end - seq_start)
    q = tl.load(Q + (seq_start + offs_m[:, None])*Stride_qt + pid_h*Stride_qh + tl.arange(0, HEAD_DIM)[None, :], mask=mask_m[:, None], other=0.0)
    g = tl.load(G_slc + (seq_start + offs_m[:, None])*Stride_gt + pid_h*Stride_gh, mask=mask_m[:, None], other=0.0)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for s_idx in range(S):
        b_idx = tl.load(BlockIndices + (seq_start + offs_m)*(S*tl.num_programs(1)) + pid_h*S + s_idx, mask=mask_m, other=-1)
        for blk_offset in range(BLOCK_SIZE):
            valid = (b_idx >= 0)
            target_idx = seq_start + b_idx * BLOCK_SIZE + blk_offset
            is_bounds = target_idx < seq_end
            is_causal = target_idx <= (seq_start + offs_m)
            mask_load = valid[:, None] & mask_m[:, None] & is_bounds[:, None]
            k = tl.load(K + target_idx[:, None]*Stride_kt + pid_h*Stride_kh + tl.arange(0, HEAD_DIM)[None, :], mask=mask_load, other=0.0)
            v = tl.load(V + target_idx[:, None]*Stride_vt + pid_h*Stride_vh + tl.arange(0, HEAD_DIM)[None, :], mask=mask_load, other=0.0)
            score = tl.sum(q * k, axis=1) * scale
            p = _hstu_silu_activation(score)
            p = tl.where(valid & is_causal & is_bounds, p, 0.0)
            acc += p[:, None] * v
    tl.store(Out + (seq_start + offs_m[:, None])*Stride_ot + pid_h*Stride_oh + tl.arange(0, HEAD_DIM)[None, :], (acc * g).to(Out.dtype.element_ty), mask=mask_m[:, None])

# -----------------------------------------------------------------------------
# Python Wrapper
# -----------------------------------------------------------------------------
class HSTU_BSA_Triton(torch.nn.Module):
    def __init__(self, block_size=32, block_counts=4):
        super().__init__()
        self.block_size = block_size
        self.block_counts = block_counts
        
        # S: 逻辑 TopK 数量
        self.s_logic = block_counts
        # Pad 计算大小 (>=16)
        self.s_pad = max(self.s_logic, 16)

    def forward(self, q, k, v, g_cmp, g_slc, x_offsets):
        if g_cmp.dim() == 2: g_cmp = g_cmp.unsqueeze(-1)
        if g_slc.dim() == 2: g_slc = g_slc.unsqueeze(-1)
        
        B = x_offsets.size(0) - 1
        seq_lens = x_offsets[1:] - x_offsets[:-1]
        total_tokens = q.shape[0]
        device = q.device
        num_heads = q.shape[1]
        dim = q.shape[2]
        
        # 1. Jagged Pooling
        with torch.no_grad():
            cmp_seq_lens = (seq_lens + self.block_size - 1) // self.block_size
            offsets_cmp = torch.zeros_like(x_offsets)
            offsets_cmp[1:] = torch.cumsum(cmp_seq_lens, dim=0)
            total_cmp_tokens = offsets_cmp[-1].item()
            
            batch_ids = torch.repeat_interleave(torch.arange(B, device=device), seq_lens)
            local_ids = torch.arange(total_tokens, device=device) - x_offsets[:-1][batch_ids]
            local_block_ids = local_ids // self.block_size
            segment_ids = offsets_cmp[:-1][batch_ids] + local_block_ids

        k_cmp = torch.zeros((total_cmp_tokens, num_heads, dim), dtype=k.dtype, device=device)
        v_cmp = torch.zeros((total_cmp_tokens, num_heads, v.shape[-1]), dtype=v.dtype, device=device)
        k_cmp.index_add_(0, segment_ids, k)
        v_cmp.index_add_(0, segment_ids, v)
        k_cmp = k_cmp / self.block_size
        v_cmp = v_cmp / self.block_size

        # 2. Kernel Setup
        max_n = seq_lens.max().item()
        scale = dim ** -0.5
        o_cmp = torch.empty_like(v)
        o_slc = torch.empty_like(v)
        grid_triton = lambda meta: (triton.cdiv(max_n, meta['BLOCK_M']), num_heads, B)

        # 3. Launch TopK Kernel
        topk_indices = torch.full((total_tokens, num_heads, self.s_logic), -1, dtype=torch.int32, device=device)
        
        hstu_bsa_topk_kernel[grid_triton](
            Q=q, K=k_cmp, 
            Out_Indices=topk_indices,
            Stride_qt=q.stride(0), Stride_qh=q.stride(1), Stride_qd=q.stride(2),
            Stride_kt=k_cmp.stride(0), Stride_kh=k_cmp.stride(1), Stride_kd=k_cmp.stride(2),
            Stride_idx_t=topk_indices.stride(0), Stride_idx_h=topk_indices.stride(1), Stride_idx_s=topk_indices.stride(2),
            offsets=x_offsets, 
            offsets_cmp=offsets_cmp, 
            scale=scale,
            S=self.s_logic,           
            BLOCK_N_PAD=self.s_pad,   
            BLOCK_SIZE=self.block_size, HEAD_DIM=dim,
            BLOCK_M=32
        )
        
        # 4. Launch CMP & SLC
        hstu_bsa_cmp_fwd_kernel[grid_triton](
            Q=q, K=k_cmp, V=v_cmp, 
            G_cmp=g_cmp.squeeze(-1), Out=o_cmp,
            Stride_qt=q.stride(0), Stride_qh=q.stride(1), Stride_qd=q.stride(2),
            Stride_kt=k_cmp.stride(0), Stride_kh=k_cmp.stride(1), Stride_kd=k_cmp.stride(2),
            Stride_vt=v_cmp.stride(0), Stride_vh=v_cmp.stride(1), Stride_vd=v_cmp.stride(2),
            Stride_ot=o_cmp.stride(0), Stride_oh=o_cmp.stride(1), Stride_od=o_cmp.stride(2),
            Stride_gt=g_cmp.stride(0), Stride_gh=g_cmp.stride(1),
            offsets=x_offsets, offsets_cmp=offsets_cmp, scale=scale,
            BLOCK_SIZE=self.block_size, HEAD_DIM=dim, BLOCK_M=32, BLOCK_N=32
        )

        hstu_bsa_slc_fwd_kernel[grid_triton](
            Q=q, K=k, V=v, 
            G_slc=g_slc.squeeze(-1), 
            BlockIndices=topk_indices, Out=o_slc,
            Stride_qt=q.stride(0), Stride_qh=q.stride(1), Stride_qd=q.stride(2),
            Stride_kt=k.stride(0), Stride_kh=k.stride(1), Stride_kd=k.stride(2),
            Stride_vt=v.stride(0), Stride_vh=v.stride(1), Stride_vd=v.stride(2),
            Stride_ot=o_slc.stride(0), Stride_oh=o_slc.stride(1), Stride_od=o_slc.stride(2),
            Stride_gt=g_slc.stride(0), Stride_gh=g_slc.stride(1),
            offsets=x_offsets, scale=scale,
            S=self.block_counts, 
            BLOCK_SIZE=self.block_size, HEAD_DIM=dim, BLOCK_M=32
        )
        
        return o_cmp, o_slc