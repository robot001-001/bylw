import torch
import torch.nn.functional as F
import triton
import triton.language as tl

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
    Stride_gt, Stride_gh, # [Fix] 新增 Gate 的 Stride
    offsets,      
    offsets_cmp,  
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
    
    start_m = pid_m * BLOCK_M
    if start_m >= seq_len:
        return

    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_len
    
    # Load Q
    q_ptrs = Q + (seq_start + offs_m[:, None]) * Stride_qt + pid_h * Stride_qh + tl.arange(0, HEAD_DIM)[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # Load G_cmp [Fix] 使用 Stride_gt
    g_ptrs = G_cmp + (seq_start + offs_m[:, None]) * Stride_gt + pid_h * Stride_gh 
    g = tl.load(g_ptrs, mask=mask_m[:, None], other=0.0)

    cmp_start = tl.load(offsets_cmp + pid_z)
    cmp_end = tl.load(offsets_cmp + pid_z + 1)
    cmp_len = cmp_end - cmp_start 

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    for start_n in range(0, cmp_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < cmp_len
        
        k_ptrs = K + (cmp_start + offs_n[None, :]) * Stride_kt + pid_h * Stride_kh + tl.arange(0, HEAD_DIM)[:, None]
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        
        attn_score = tl.dot(q, k)
        attn_score *= scale
        
        mask_causal = (offs_m[:, None] // BLOCK_SIZE) >= offs_n[None, :]
        attn_score = tl.where(mask_causal & mask_m[:, None] & mask_n[None, :], attn_score, -1e9)
        
        p = _hstu_silu_activation(attn_score)
        p = tl.where(mask_causal & mask_n[None, :], p, 0.0)
        
        v_ptrs = V + (cmp_start + offs_n[:, None]) * Stride_vt + pid_h * Stride_vh + tl.arange(0, HEAD_DIM)[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        acc += tl.dot(p, v)

    acc = acc * g
    
    o_ptrs = Out + (seq_start + offs_m[:, None]) * Stride_ot + pid_h * Stride_oh + tl.arange(0, HEAD_DIM)[None, :]
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])


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
    Stride_gt, Stride_gh, # [Fix] 新增 Gate Stride
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

    q_ptrs = Q + (seq_start + offs_m[:, None]) * Stride_qt + pid_h * Stride_qh + tl.arange(0, HEAD_DIM)[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Load G_slc [Fix] 使用 Stride_gt
    g_ptrs = G_slc + (seq_start + offs_m[:, None]) * Stride_gt + pid_h * Stride_gh
    g = tl.load(g_ptrs, mask=mask_m[:, None], other=0.0)

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for s_idx in range(S):
        b_idxs_ptr = BlockIndices + (seq_start + offs_m) * (S * tl.num_programs(1)) + pid_h * S + s_idx
        b_idx = tl.load(b_idxs_ptr, mask=mask_m, other=-1) 

        for blk_offset in range(BLOCK_SIZE):
            valid_blk = b_idx >= 0
            target_k_idx = seq_start + b_idx * BLOCK_SIZE + blk_offset
            
            # Boundary Checks
            is_within_bounds = target_k_idx < seq_end
            is_causal = target_k_idx <= (seq_start + offs_m)
            
            mask_load = valid_blk[:, None] & mask_m[:, None] & is_within_bounds[:, None]
            
            k_ptrs_col = K + target_k_idx[:, None] * Stride_kt + pid_h * Stride_kh + tl.arange(0, HEAD_DIM)[None, :]
            k_val = tl.load(k_ptrs_col, mask=mask_load, other=0.0)
            
            score = tl.sum(q * k_val, axis=1) 
            score *= scale
            
            p = _hstu_silu_activation(score)
            p = tl.where(valid_blk & is_causal & is_within_bounds, p, 0.0)
            
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

    def forward(self, q, k, v, g_cmp, g_slc, x_offsets):
        # 确保 Gate 是 3D 的 [Tokens, H, 1]
        if g_cmp.dim() == 2: g_cmp = g_cmp.unsqueeze(-1)
        if g_slc.dim() == 2: g_slc = g_slc.unsqueeze(-1)
        
        B = x_offsets.size(0) - 1
        seq_lens = x_offsets[1:] - x_offsets[:-1]
        total_tokens = q.shape[0]
        device = q.device
        
        # 1. Jagged Pooling Indices
        with torch.no_grad():
            cmp_seq_lens = (seq_lens + self.block_size - 1) // self.block_size
            offsets_cmp = torch.zeros_like(x_offsets)
            offsets_cmp[1:] = torch.cumsum(cmp_seq_lens, dim=0)
            total_cmp_tokens = offsets_cmp[-1].item()
            
            batch_ids = torch.repeat_interleave(torch.arange(B, device=device), seq_lens)
            local_ids = torch.arange(total_tokens, device=device) - x_offsets[:-1][batch_ids]
            local_block_ids = local_ids // self.block_size
            segment_ids = offsets_cmp[:-1][batch_ids] + local_block_ids

        # 2. Pooling (Sum / 32)
        k_cmp = torch.zeros((total_cmp_tokens, k.shape[1], k.shape[2]), dtype=k.dtype, device=device)
        v_cmp = torch.zeros((total_cmp_tokens, v.shape[1], v.shape[2]), dtype=v.dtype, device=device)
        k_cmp.index_add_(0, segment_ids, k)
        v_cmp.index_add_(0, segment_ids, v)
        k_cmp = k_cmp / self.block_size
        v_cmp = v_cmp / self.block_size

        # 3. TopK Selection
        max_n = seq_lens.max().item()
        max_blocks = cmp_seq_lens.max().item()
        num_heads = q.shape[1]
        dim = q.shape[2]
        
        padded_q = torch.zeros(B, max_n, num_heads, dim, device=device, dtype=q.dtype)
        padded_q[batch_ids, local_ids] = q
        
        padded_k_cmp = torch.zeros(B, max_blocks, num_heads, dim, device=device, dtype=k_cmp.dtype)
        batch_ids_cmp = torch.repeat_interleave(torch.arange(B, device=device), cmp_seq_lens)
        local_ids_cmp = torch.arange(total_cmp_tokens, device=device) - offsets_cmp[:-1][batch_ids_cmp]
        padded_k_cmp[batch_ids_cmp, local_ids_cmp] = k_cmp
        
        scale = dim ** -0.5
        attn_cmp_scores = torch.einsum('bqhd,bkhd->bhqk', padded_q, padded_k_cmp) * scale
        
        indices_q = torch.arange(max_n, device=device)[:, None] // self.block_size
        indices_k = torch.arange(max_blocks, device=device)[None, :]
        causal_mask = indices_q >= indices_k
        attn_cmp_scores.masked_fill_(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        S = min(self.block_counts, max_blocks)
        _, topk_indices = attn_cmp_scores.topk(S, dim=-1)
        
        valid_mask = torch.arange(max_n, device=device)[None, :] < seq_lens[:, None]
        topk_indices = topk_indices.permute(0, 2, 1, 3)
        topk_indices_jag = topk_indices[valid_mask].contiguous().view(-1, num_heads, S).int()

        # 4. Launch
        o_cmp = torch.empty_like(v)
        o_slc = torch.empty_like(v)
        
        grid_triton = lambda meta: (triton.cdiv(max_n, meta['BLOCK_M']), num_heads, B)

        hstu_bsa_cmp_fwd_kernel[grid_triton](
            Q=q, K=k_cmp, V=v_cmp, 
            G_cmp=g_cmp.squeeze(-1), 
            Out=o_cmp,
            Stride_qt=q.stride(0), Stride_qh=q.stride(1), Stride_qd=q.stride(2),
            Stride_kt=k_cmp.stride(0), Stride_kh=k_cmp.stride(1), Stride_kd=k_cmp.stride(2),
            Stride_vt=v_cmp.stride(0), Stride_vh=v_cmp.stride(1), Stride_vd=v_cmp.stride(2),
            Stride_ot=o_cmp.stride(0), Stride_oh=o_cmp.stride(1), Stride_od=o_cmp.stride(2),
            Stride_gt=g_cmp.stride(0), Stride_gh=g_cmp.stride(1), # [Fix] Pass Gate Strides
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
            Stride_gt=g_slc.stride(0), Stride_gh=g_slc.stride(1), # [Fix] Pass Gate Strides
            offsets=x_offsets, 
            scale=scale,
            S=S, BLOCK_SIZE=self.block_size, HEAD_DIM=dim, BLOCK_M=32
        )
        
        return o_cmp, o_slc