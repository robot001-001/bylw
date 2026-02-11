import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _hstu_silu_activation(x):
    return x * tl.sigmoid(x)

# -----------------------------------------------------------------------------
# [Fix] TopK Kernel (List-based Bubble Insertion)
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
    S: tl.constexpr,          # 逻辑 TopK 数量
    BLOCK_N_PAD: tl.constexpr,# 物理计算维度 (>=16)
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
    # 初始化 Top S 列表 (Python List of Registers)
    # -----------------------------------------------------------
    # 我们使用 Python 列表来模拟数组，避免 Tensor 索引问题
    top_vals_list = []
    top_idxs_list = []
    for _ in range(S):
        top_vals_list.append(tl.full([BLOCK_M], float('-inf'), dtype=tl.float32))
        top_idxs_list.append(tl.full([BLOCK_M], -1, dtype=tl.int32))

    cmp_start = tl.load(offsets_cmp + pid_z)
    cmp_end = tl.load(offsets_cmp + pid_z + 1)
    cmp_len = cmp_end - cmp_start 

    # -----------------------------------------------------------
    # Stream Processing
    # -----------------------------------------------------------
    # 每次处理 BLOCK_N_PAD (16) 个候选者
    for start_n in range(0, cmp_len, BLOCK_N_PAD):
        # 1. Load K
        offs_n_pad = start_n + tl.arange(0, BLOCK_N_PAD)
        mask_n_valid = offs_n_pad < cmp_len
        
        k_ptrs = K + (cmp_start + offs_n_pad[None, :]) * Stride_kt + pid_h * Stride_kh + tl.arange(0, HEAD_DIM)[:, None]
        k = tl.load(k_ptrs, mask=mask_n_valid[None, :], other=0.0)
        
        # 2. Compute Score
        scores = tl.dot(q, k) 
        scores *= scale
        
        # 3. Masking
        mask_causal = (offs_m[:, None] // BLOCK_SIZE) >= offs_n_pad[None, :]
        mask_final = mask_causal & mask_n_valid[None, :] & mask_m[:, None]
        scores = tl.where(mask_final, scores, float('-inf'))

        # 生成对应的全局索引
        curr_idxs = (start_n + tl.arange(0, BLOCK_N_PAD))[None, :]
        curr_idxs = tl.broadcast_to(curr_idxs.to(tl.int32), [BLOCK_M, BLOCK_N_PAD])
        
        # -------------------------------------------------------
        # Bubble Insertion Strategy (Unrolled)
        # -------------------------------------------------------
        # 我们有 BLOCK_N_PAD 列新数据。我们将每一列依次插入到 Top S 列表中。
        # 虽然是双重循环，但都是 constexpr，会被完全展开为寄存器操作。
        
        for col in range(BLOCK_N_PAD):
            # 提取当前列 (View slice) - 这是合法的，因为 col 是编译期常量
            # 注意：Triton 中 tensor[:, constant] 是合法的切片
            # 但为了保险，我们使用 range slice
            r_slice = tl.arange(0, BLOCK_M)
            c_slice = tl.arange(col, col+1) 
            # 这种切片方式在某些版本可能还是有问题，最稳健的是 reshape 后索引
            # 但既然 scores 是 [BLOCK_M, BLOCK_N_PAD]，我们可以直接用 split
            # 或者更简单的：我们不需要提取，直接在计算时 broadcast
            # 让我们尝试最简单的 broadcast mask 提取，虽然有点浪费计算，但绝对兼容
            
            # 使用 split 可能是最干净的，但 Triton split 语法不直观。
            # 让我们信任 view 切片对于 constant index 的支持
            # 如果 scores[:, col] 失败，我们用 reshape
            
            # 尝试方案：Standard indexing
            # val = tl.load? No, register.
            # val = scores[:, col]
            
            # 为了绕过任何潜在的 Slice 错误，我们使用全手动掩码提取 (Gather by Arithmetic)
            # 虽然笨，但 100% work。
            col_mask = (tl.arange(0, BLOCK_N_PAD) == col) # [BLOCK_N_PAD]
            # Sum reduce to extract column: val = sum(scores * mask, 1)
            val = tl.sum(scores * col_mask[None, :], axis=1)
            idx = tl.sum(curr_idxs * col_mask[None, :], axis=1)
            
            # 将 val, idx 插入到 top_vals_list 中
            for k in range(S):
                old_val = top_vals_list[k]
                old_idx = top_idxs_list[k]
                
                # 比较：如果新值更大，则交换
                swap = val > old_val
                
                top_vals_list[k] = tl.where(swap, val, old_val)
                top_idxs_list[k] = tl.where(swap, idx, old_idx)
                
                # 被挤出来的值变成新的 val，继续去比较下一个位置
                val = tl.where(swap, old_val, val)
                idx = tl.where(swap, old_idx, idx)

    # -----------------------------------------------------------
    # Store Indices
    # -----------------------------------------------------------
    # 将列表写回内存
    for k in range(S):
        idx_ptrs = Out_Indices + (seq_start + offs_m) * Stride_idx_t + pid_h * Stride_idx_h + k * Stride_idx_s
        tl.store(idx_ptrs, top_idxs_list[k], mask=mask_m)

# -----------------------------------------------------------------------------
# CMP / SLC Kernels (Keep unchanged)
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
        
        # S 逻辑大小
        self.s_pow2 = 1
        while self.s_pow2 < self.block_counts:
            self.s_pow2 *= 2
        # Pad 计算大小 (>=16)
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
            S=self.s_pow2, 
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