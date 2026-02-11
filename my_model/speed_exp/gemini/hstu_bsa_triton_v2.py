import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Triton Helper: Bitonic Sort (保持不变)
# -----------------------------------------------------------------------------
@triton.jit
def _bitonic_sort_descending(v, i):
    S: tl.constexpr = v.shape[0]
    for k in tl.static_range(1, tl.cdiv(S.bit_length() - 1, 1) + 1):
        step = 1 << k
        for j in tl.static_range(k):
            mask_step = 1 << (k - 1 - j)
            idx = tl.arange(0, S)
            partner_idx = idx ^ mask_step
            val = v
            partner_val = tl.view(v, [S])[partner_idx]
            idx_val = i
            partner_idx_val = tl.view(i, [S])[partner_idx]
            descending_group = ((idx // step) % 2) == 0
            is_smaller = val < partner_val
            swap = (descending_group & is_smaller) | ((~descending_group) & (~is_smaller))
            v = tl.where(swap, partner_val, val)
            i = tl.where(swap, partner_idx_val, idx_val)
            
    for k in tl.static_range(tl.cdiv(S.bit_length() - 1, 1), -1, -1):
         mask_step = 1 << k
         idx = tl.arange(0, S)
         partner_idx = idx ^ mask_step
         val = v
         partner_val = tl.view(v, [S])[partner_idx]
         idx_val = i
         partner_idx_val = tl.view(i, [S])[partner_idx]
         larger = tl.maximum(val, partner_val)
         smaller = tl.minimum(val, partner_val)
         larger_idx = tl.where(val > partner_val, idx_val, partner_idx_val)
         smaller_idx = tl.where(val > partner_val, partner_idx_val, idx_val)
         v = tl.where(idx < partner_idx, larger, smaller)
         i = tl.where(idx < partner_idx, larger_idx, smaller_idx)
    return v, i

@triton.jit
def _hstu_silu_activation(x):
    return x * tl.sigmoid(x)

# -----------------------------------------------------------------------------
# [Fix] TopK Kernel (处理维度 < 16 的问题)
# -----------------------------------------------------------------------------
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
    S: tl.constexpr,          # 实际需要的 TopK 数量
    BLOCK_N_PAD: tl.constexpr,# [Fix] Pad 到至少 16 的计算维度
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

    # 寄存器 TopK 列表 (大小为 S)
    # 注意：为了让 Bitonic Sort 工作，S 最好是 2 的幂次。如果 S < 16，我们这里 top_vals 依然维持 S 大小
    # 只要 Bitonic Sort 函数能处理 S 即可 (前面实现的版本可以)
    # 但为了方便合并，我们通常让 S >= 16 并且是 2 的幂次。
    # 这里我们假设 Python 端已经处理好了 S (比如 padding 到了 16/32)
    top_vals = tl.full([BLOCK_M, S], float('-inf'), dtype=tl.float32)
    top_idxs = tl.full([BLOCK_M, S], -1, dtype=tl.int32)

    cmp_start = tl.load(offsets_cmp + pid_z)
    cmp_end = tl.load(offsets_cmp + pid_z + 1)
    cmp_len = cmp_end - cmp_start 

    # 循环步长使用 S，但加载/计算使用 BLOCK_N_PAD (>=16)
    for start_n in range(0, cmp_len, S):
        # [Fix] 使用 Pad 后的维度生成 offset
        offs_n_pad = start_n + tl.arange(0, BLOCK_N_PAD)
        # Mask: 1. 不能超过总长 cmp_len; 2. 逻辑上只取前 S 个 (当前批次)
        # 如果 start_n + k >= start_n + S，说明是 Pad 出来的无效计算位
        mask_n_valid = (offs_n_pad < cmp_len) & (offs_n_pad < (start_n + S))
        
        # Load K (Pad)
        k_ptrs = K + (cmp_start + offs_n_pad[None, :]) * Stride_kt + pid_h * Stride_kh + tl.arange(0, HEAD_DIM)[:, None]
        k = tl.load(k_ptrs, mask=mask_n_valid[None, :], other=0.0)
        
        # Compute Score: [BLOCK_M, D] @ [BLOCK_N_PAD, D].T -> [BLOCK_M, BLOCK_N_PAD]
        # 现在维度 >= 16，tl.dot 安全了
        scores = tl.dot(q, k) 
        scores *= scale
        
        # Causal Masking & Valid Masking
        # Pad 出的部分 (-inf) 自动会被淘汰
        mask_causal = (offs_m[:, None] // BLOCK_SIZE) >= offs_n_pad[None, :]
        mask_final = mask_causal & mask_n_valid[None, :] & mask_m[:, None]
        scores = tl.where(mask_final, scores, float('-inf'))
        
        # [Fix] Slice: 我们只关心前 S 个结果用于 Merge
        # 因为 BLOCK_N_PAD >= S，我们只取 [:S]
        # 这一步将数据切回 S 大小，以便与 top_vals (Size S) 进行 Bitonic Merge
        # Triton 支持 slice 操作吗？不支持标准的 slice语法。
        # 我们可以重新构造一个 range(0, S) 来 gather
        slice_idx = tl.arange(0, S)
        new_vals = tl.view(scores, [BLOCK_M, BLOCK_N_PAD])[:, slice_idx] # Gather cols 0..S-1
        new_idxs = (start_n + slice_idx)[None, :].to(tl.int32)
        new_idxs = tl.broadcast_to(new_idxs, [BLOCK_M, S])
        
        # -----------------------------------------------------------
        # Stream-TopK Merge (In-Register) - 保持之前的逻辑
        # -----------------------------------------------------------
        
        # 1. Sort New Candidates
        new_vals, new_idxs = _bitonic_sort_descending(new_vals, new_idxs)
        
        # 2. Merge with Current Top
        # Reverse new to form bitonic sequence
        rev_idx = S - 1 - tl.arange(0, S)
        new_vals_rev = tl.view(new_vals, [BLOCK_M, S])[:, rev_idx]
        new_idxs_rev = tl.view(new_idxs, [BLOCK_M, S])[:, rev_idx]
        
        # Compare and Swap
        cmp1 = new_vals_rev > top_vals
        temp_v = top_vals
        temp_i = top_idxs
        top_vals = tl.where(cmp1, new_vals_rev, temp_v)
        top_idxs = tl.where(cmp1, new_idxs_rev, temp_i)
        
        # Re-sort
        top_vals, top_idxs = _bitonic_sort_descending(top_vals, top_idxs)

    # Store Indices
    idx_ptrs = Out_Indices + (seq_start + offs_m[:, None]) * Stride_idx_t + pid_h * Stride_idx_h + tl.arange(0, S)[None, :] * Stride_idx_s
    tl.store(idx_ptrs, top_idxs, mask=mask_m[:, None])

# -----------------------------------------------------------------------------
# CMP / SLC Kernels (保持不变，省略以节省篇幅，请保留之前的版本)
# ... hstu_bsa_cmp_fwd_kernel ...
# ... hstu_bsa_slc_fwd_kernel ...
# -----------------------------------------------------------------------------
# 为了代码完整性，这里我还是把这两个 Kernel 声明一下，确保可以直接运行
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
# Python Wrapper Fix
# -----------------------------------------------------------------------------

class HSTU_BSA_Triton(torch.nn.Module):
    def __init__(self, block_size=32, block_counts=4):
        super().__init__()
        self.block_size = block_size
        self.block_counts = block_counts
        
        # [Fix] 1. 确保 S 是 2 的幂次 (Bitonic Sort 要求)
        # [Fix] 2. 确保 S 至少是 16 (tl.dot 要求)
        # S_pow2: 用于 Bitonic Sort 的逻辑大小
        self.s_pow2 = 1
        while self.s_pow2 < self.block_counts:
            self.s_pow2 *= 2
            
        # S_pad: 用于 tl.dot 的物理计算大小，必须 >= 16
        # 如果 s_pow2 < 16 (例如 2, 4, 8)，我们 Pad 到 16
        # 如果 s_pow2 >= 16 (例如 16, 32)，则直接使用 s_pow2
        self.s_pad = max(self.s_pow2, 16)

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
        # Output: [TotalTokens, H, S] (Jagged)
        # 这里的 Buffer 大小使用 s_pad 还是 s_pow2 ? 
        # 为了安全，我们申请 s_pad 大小，或者只申请 s_pow2，但 Kernel 里可能会越界写？
        # Kernel 里 store 使用的是 S (即 s_pad)，所以 buffer 必须够大。
        # 修正：Kernel 里的 S 应该是 s_pow2 (逻辑 TopK)。pad 仅用于 dot。
        # 让我们调整 Kernel 传参：S=s_pow2, BLOCK_N_PAD=s_pad。
        # Buffer 只需要存 s_pow2 即可 (Bitonic Sort 后只保留 s_pow2 个)。
        
        # [Critical] 重新审视 Kernel store 部分：
        # Kernel 最后的 tl.store 使用 range(0, S)。如果 S=2, store 2个。
        # 所以 Buffer 大小只要 s_pow2 即可。
        
        topk_indices = torch.full((total_tokens, num_heads, self.s_pow2), -1, dtype=torch.int32, device=device)
        
        hstu_bsa_topk_kernel[grid_triton](
            Q=q, K=k_cmp, 
            Out_Indices=topk_indices,
            Stride_qt=q.stride(0), Stride_qh=q.stride(1), Stride_qd=q.stride(2),
            Stride_kt=k_cmp.stride(0), Stride_kh=k_cmp.stride(1), Stride_kd=k_cmp.stride(2),
            Stride_idx_t=topk_indices.stride(0), Stride_idx_h=topk_indices.stride(1), Stride_idx_s=topk_indices.stride(2),
            offsets=x_offsets, 
            offsets_cmp=offsets_cmp, 
            scale=scale,
            # S 是逻辑 TopK (如 2, 4, 16), BLOCK_N_PAD 是物理计算维 (如 16, 16, 16)
            S=self.s_pow2, 
            BLOCK_N_PAD=self.s_pad,
            BLOCK_SIZE=self.block_size, HEAD_DIM=dim,
            BLOCK_M=32
        )
        
        S_real = self.block_counts

        # 4. Launch CMP Kernel
        hstu_bsa_cmp_fwd_kernel[grid_triton](
            Q=q, K=k_cmp, V=v_cmp, 
            G_cmp=g_cmp.squeeze(-1), 
            Out=o_cmp,
            Stride_qt=q.stride(0), Stride_qh=q.stride(1), Stride_qd=q.stride(2),
            Stride_kt=k_cmp.stride(0), Stride_kh=k_cmp.stride(1), Stride_kd=k_cmp.stride(2),
            Stride_vt=v_cmp.stride(0), Stride_vh=v_cmp.stride(1), Stride_vd=v_cmp.stride(2),
            Stride_ot=o_cmp.stride(0), Stride_oh=o_cmp.stride(1), Stride_od=o_cmp.stride(2),
            Stride_gt=g_cmp.stride(0), Stride_gh=g_cmp.stride(1),
            offsets=x_offsets, 
            offsets_cmp=offsets_cmp, 
            scale=scale,
            BLOCK_SIZE=self.block_size, HEAD_DIM=dim,
            BLOCK_M=32, BLOCK_N=32
        )

        # 5. Launch SLC Kernel
        hstu_bsa_slc_fwd_kernel[grid_triton](
            Q=q, K=k, V=v, 
            G_slc=g_slc.squeeze(-1), 
            BlockIndices=topk_indices, 
            Out=o_slc,
            Stride_qt=q.stride(0), Stride_qh=q.stride(1), Stride_qd=q.stride(2),
            Stride_kt=k.stride(0), Stride_kh=k.stride(1), Stride_kd=k.stride(2),
            Stride_vt=v.stride(0), Stride_vh=v.stride(1), Stride_vd=v.stride(2),
            Stride_ot=o_slc.stride(0), Stride_oh=o_slc.stride(1), Stride_od=o_slc.stride(2),
            Stride_gt=g_slc.stride(0), Stride_gh=g_slc.stride(1),
            offsets=x_offsets, 
            scale=scale,
            S=S_real, 
            BLOCK_SIZE=self.block_size, HEAD_DIM=dim, BLOCK_M=32
        )
        
        return o_cmp, o_slc