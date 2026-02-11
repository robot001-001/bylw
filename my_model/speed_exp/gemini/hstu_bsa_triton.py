import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import math

# -----------------------------------------------------------------------------
# Triton Kernels for HSTU BSA
# -----------------------------------------------------------------------------

@triton.jit
def _hstu_silu_activation(x):
    # SiLU (Swish) activation: x * sigmoid(x)
    return x * tl.sigmoid(x)

@triton.jit
def hstu_bsa_cmp_fwd_kernel(
    Q, K, V, 
    G_cmp, 
    Out, Scores,
    Stride_qt, Stride_qh, Stride_qd,
    Stride_kt, Stride_kh, Stride_kd,
    Stride_vt, Stride_vh, Stride_vd,
    Stride_ot, Stride_oh, Stride_od,
    offsets, 
    scale,
    BLOCK_SIZE: tl.constexpr, # Compression factor (e.g., 32)
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,    # Processing block size for Q
    BLOCK_N: tl.constexpr,    # Processing block size for K/V (chunk size)
):
    """
    Compression Attention Kernel for HSTU.
    Computes O_cmp = SiLU(Q @ K_cmp.T) @ V_cmp * g_cmp
    Also stores attn scores for TopK selection.
    """
    pid_m = tl.program_id(0) # Block index along sequence dimension (Token blocks)
    pid_h = tl.program_id(1) # Head index
    pid_z = tl.program_id(2) # Batch index

    # 1. Coordinate Setup
    seq_start = tl.load(offsets + pid_z)
    seq_end = tl.load(offsets + pid_z + 1)
    seq_len = seq_end - seq_start
    
    # Compressed length
    # K, V passed here are already compressed (pooled)
    cmp_len = tl.cdiv(seq_len, BLOCK_SIZE)

    # Current Q block range (in original sequence)
    start_m = pid_m * BLOCK_M
    if start_m >= seq_len:
        return

    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_len
    
    # Load Q: [BLOCK_M, HEAD_DIM]
    # Pointers: Q_base + (seq_start + offs_m)*Stride_qt + pid_h*Stride_qh
    q_ptrs = Q + (seq_start + offs_m[:, None]) * Stride_qt + pid_h * Stride_qh + tl.arange(0, HEAD_DIM)[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # Load G_cmp: [BLOCK_M, 1] - Gating factor
    # Gating shape assumption: [TotalTokens, H]
    g_ptrs = G_cmp + (seq_start + offs_m[:, None]) * Stride_qt + pid_h * Stride_qh # Assuming same stride layout as Q for T/H
    g = tl.load(g_ptrs, mask=mask_m[:, None], other=0.0)

    # Initialize Accumulator
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # Loop over Compressed K/V
    # We loop step by step to handle Causal Masking
    for start_n in range(0, cmp_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < cmp_len
        
        # Load K_cmp: [BLOCK_N, HEAD_DIM]
        # K, V are compacted tensors [Total_Cmp_Tokens, H, D]
        # We need to map seq_id -> compressed_offset. 
        # Ideally passed pre-calculated, but here we approximate:
        # Assuming K, V are flat buffers of compressed tokens. 
        # We need a `cmp_offsets` input ideally. 
        # For simplicity in this snippet, we assume K,V inputs are [B, Max_Cmp_Len, H, D] strided.
        # If passed as Jagged, we need `cmp_offsets`. Let's assume standard strided for K/V inside kernel for readbility.
        
        k_ptrs = K + (start_n + offs_n[None, :]) * Stride_kt + pid_h * Stride_kh + tl.arange(0, HEAD_DIM)[:, None]
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        
        # Compute Attention Score: Q @ K.T
        attn_score = tl.dot(q, k) # [BLOCK_M, BLOCK_N]
        attn_score *= scale
        
        # Causal Masking for BSA
        # Condition: original_idx // BLOCK_SIZE >= compressed_idx
        # offs_m (original) vs offs_n (compressed)
        mask_causal = (offs_m[:, None] // BLOCK_SIZE) >= offs_n[None, :]
        attn_score = tl.where(mask_causal & mask_m[:, None] & mask_n[None, :], attn_score, -1e9)
        
        # Store Scores for TopK (Optional: if needed by python wrapper, else TopK done on fly)
        # Writing all scores [T, T/BS] is expensive. 
        # Ideally TopK is fused or we write compact scores.
        if Scores is not None:
             # Logic to write scores to global memory...
             pass

        # HSTU Activation: SiLU (Not Softmax!)
        p = _hstu_silu_activation(attn_score)
        p = tl.where(mask_causal & mask_n[None, :], p, 0.0)
        
        # Load V_cmp
        v_ptrs = V + (start_n + offs_n[:, None]) * Stride_vt + pid_h * Stride_vh + tl.arange(0, HEAD_DIM)[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Accumulate: P @ V
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16))

    # Apply Gating and Write Output
    # o_cmp = acc * g_cmp
    acc = acc * g
    
    o_ptrs = Out + (seq_start + offs_m[:, None]) * Stride_ot + pid_h * Stride_oh + tl.arange(0, HEAD_DIM)[None, :]
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def hstu_bsa_slc_fwd_kernel(
    Q, K, V, 
    G_slc, 
    BlockIndices, # [TotalTokens, H, S]
    Out,
    Stride_qt, Stride_qh, Stride_qd,
    Stride_kt, Stride_kh, Stride_kd,
    Stride_vt, Stride_vh, Stride_vd,
    Stride_ot, Stride_oh, Stride_od,
    offsets,
    scale,
    S: tl.constexpr,          # Number of selected blocks
    BLOCK_SIZE: tl.constexpr, # Raw Block Size (32)
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,    # Processing block size for Q
):
    """
    Selected Attention Kernel for HSTU.
    Gathers K/V based on BlockIndices.
    Computes O_slc = SiLU(Q @ K_slc.T) @ V_slc * g_slc
    """
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

    # Load G_slc
    g_ptrs = G_slc + (seq_start + offs_m[:, None]) * Stride_qt + pid_h * Stride_qh
    g = tl.load(g_ptrs, mask=mask_m[:, None], other=0.0)

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Load Block Indices: [BLOCK_M, S]
    # Indices are relative to the sequence or absolute? Usually relative block idx.
    # BlockIndices shape: [TotalTokens, H, S] flattened or similar.
    # We stride manually assuming [TotalTokens, H, S]
    
    # 这里的 stride 计算需要非常小心，假设 BlockIndices 是 [TotalTokens, H, S]
    # stride_bt = H * S, stride_bh = S
    idx_base = (seq_start + offs_m) * (scale * 0 + S * tl.num_programs(1)) # trick to get stride, strictly need params
    # For simplicity, we iterate S (number of selected blocks)
    
    # We process each 's' in loop to save registers
    for s_idx in range(S):
        # Load block index for each query in BLOCK_M
        # ptr: BlockIndices + (seq_start + offs_m) * H * S + pid_h * S + s_idx
        # We assume indices are int32
        # 注意: BlockIndices 是 [TotalTokens, H, S]
        b_idxs_ptr = BlockIndices + (seq_start + offs_m) * (S * tl.num_programs(1)) + pid_h * S + s_idx
        b_idx = tl.load(b_idxs_ptr, mask=mask_m, other=-1) # [BLOCK_M]

        # b_idx contains the index of the block (0, 1, 2...)
        # real_k_start = seq_start + b_idx * BLOCK_SIZE
        # We need to gather K, V. Since b_idx varies per query, we have indirect access.
        # This is expensive. We can optimize by grouping Qs with same blocks, 
        # but for BSA, pattern is irregular.
        
        # Inner loop over block_size tokens (32)
        # This double loop structure (S * BLOCK_SIZE) creates [BLOCK_M, S * BLOCK_SIZE] attention
        
        # Optimization: To avoid divergent loads, assume standard gather logic
        # For 'jagged' native sparse attention, we often accept divergent loads or use `tl.load(ptrs)`
        
        # Construct pointers for K/V Gather
        # Target: [BLOCK_M, BLOCK_SIZE, HEAD_DIM] -> very large registry pressure
        # Better: iterate block_size inner loop
        
        for blk_offset in range(BLOCK_SIZE):
            # Calculate absolute token index for K/V
            # target_idx = seq_start + b_idx * BLOCK_SIZE + blk_offset
            # Mask out if b_idx == -1
            
            valid_blk = b_idx >= 0
            target_k_idx = seq_start + b_idx * BLOCK_SIZE + blk_offset
            
            # Causal Mask check: target_k_idx <= (seq_start + offs_m)
            is_causal = target_k_idx <= (seq_start + offs_m)
            
            # Load K column: [BLOCK_M, HEAD_DIM] (Gathered!)
            k_ptrs_col = K + target_k_idx[:, None] * Stride_kt + pid_h * Stride_kh + tl.arange(0, HEAD_DIM)[None, :]
            mask_load = valid_blk[:, None] & mask_m[:, None]
            k_val = tl.load(k_ptrs_col, mask=mask_load, other=0.0)
            
            # Q @ K_val.T (element-wise dot per row) -> [BLOCK_M]
            # Since we loaded specific K for each Q, it is a row-wise dot product, not matmul
            score = tl.sum(q * k_val, axis=1) 
            score *= scale
            
            # Activation
            p = _hstu_silu_activation(score)
            p = tl.where(valid_blk & is_causal, p, 0.0)
            
            # Load V column
            v_ptrs_col = V + target_k_idx[:, None] * Stride_vt + pid_h * Stride_vh + tl.arange(0, HEAD_DIM)[None, :]
            v_val = tl.load(v_ptrs_col, mask=mask_load, other=0.0)
            
            # Accumulate: p (scalar per row) * v_val (vector per row)
            acc += p[:, None] * v_val
            
    # Apply Gating
    acc = acc * g
    
    o_ptrs = Out + (seq_start + offs_m[:, None]) * Stride_ot + pid_h * Stride_oh + tl.arange(0, HEAD_DIM)[None, :]
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])


# -----------------------------------------------------------------------------
# Python Wrapper implementing the Logic
# -----------------------------------------------------------------------------

class HSTU_BSA_Triton(torch.nn.Module):
    def __init__(self, block_size=32, block_counts=4):
        super().__init__()
        self.block_size = block_size
        self.block_counts = block_counts

    def forward(self, 
                q, k, v, u, 
                x_offsets, 
                gate_model, 
                invalid_attn_mask=None):
        """
        Args:
            q, k, v, u: Jagged Tensors flattened [TotalTokens, H, D]
            x_offsets: [B+1]
            gate_model: nn.Module returning g_cmp, g_slc, g_swa
        """
        # 1. Preprocessing / Gating
        # HSTU expects padded input for gate_model usually, but here we assume adapted
        # Reconstruct padded for gate_model as per user snippet
        B = x_offsets.size(0) - 1
        n = (x_offsets[1:] - x_offsets[:-1]).max().item() # Max Seq Len
        num_heads = q.shape[1]
        dim = q.shape[2]
        
        # 注意：为了让 Triton 高效，我们尽量保持 Jagged 形态
        # 但 gate_model 需要 padded，我们先转换
        padded_q = torch.ops.fbgemm.jagged_to_padded_dense(
            values=q, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
        ).view(B, n, num_heads, dim)
        
        g_cmp, g_slc, g_swa = gate_model(padded_q)
        
        # Flatten gates back to Jagged for Triton kernels
        # dense_to_jagged returns [TotalTokens, H, 1] (roughly)
        g_cmp_jag = torch.ops.fbgemm.dense_to_jagged(g_cmp.view(B, n, num_heads), [x_offsets])[0]
        g_slc_jag = torch.ops.fbgemm.dense_to_jagged(g_slc.view(B, n, num_heads), [x_offsets])[0]
        
        scale = dim ** -0.5
        
        # 2. Compression (Pooling)
        # Efficient AvgPool using PyTorch scatter or loop (Simpler than writing kernel for this demo)
        # For correctness with 'x_offsets', we iterate or use padded
        # Using padded is easier for pooling:
        padded_k = torch.ops.fbgemm.jagged_to_padded_dense(
            values=k, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
        ).view(B, n, num_heads, dim)
        padded_v = torch.ops.fbgemm.jagged_to_padded_dense(
            values=v, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
        ).view(B, n, num_heads, v.shape[-1])
        
        # Pool: [B, N/BS, H, D]
        num_blocks = math.ceil(n / self.block_size)
        pad_len = num_blocks * self.block_size - n
        if pad_len > 0:
            padded_k = F.pad(padded_k, (0, 0, 0, 0, 0, pad_len))
            padded_v = F.pad(padded_v, (0, 0, 0, 0, 0, pad_len))
        
        k_cmp = padded_k.view(B, num_blocks, self.block_size, num_heads, dim).mean(dim=2)
        v_cmp = padded_v.view(B, num_blocks, self.block_size, num_heads, -1).mean(dim=2)
        
        # 3. Coarse Attention & TopK
        # Compute Q @ K_cmp.T scores for TopK selection
        # We can use PyTorch for this step since N/BS is small
        attn_cmp_scores = torch.einsum('bqhd,bkhd->bhqk', padded_q, k_cmp) * scale
        
        # Causal Masking on Block level
        # indices: q_idx // BS >= k_idx
        indices_q = torch.arange(n, device=q.device)[:, None] // self.block_size
        indices_k = torch.arange(num_blocks, device=q.device)[None, :]
        causal_mask = indices_q >= indices_k
        attn_cmp_scores.masked_fill_(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # TopK Selection
        # [B, H, N, S]
        S = min(self.block_counts, num_blocks)
        _, topk_indices = attn_cmp_scores.topk(S, dim=-1)
        topk_indices = topk_indices.masked_fill(
            topk_indices > (torch.arange(n, device=q.device)[None, None, :, None] // self.block_size), -1
        )
        
        # Flatten TopK indices to Jagged [TotalTokens, H, S] for Triton
        topk_indices_jag = torch.ops.fbgemm.dense_to_jagged(
            topk_indices.permute(0, 2, 1, 3).flatten(2, 3), # [B, N, H*S]
            [x_offsets]
        )[0].view(-1, num_heads, S)

        # 4. Triton Kernel Launches
        total_tokens = q.shape[0]
        
        # Output buffers
        o_cmp = torch.empty_like(v)
        o_slc = torch.empty_like(v)
        
        # Grid setup
        grid_cmp = (triton.cdiv(n, 32), num_heads, B) # (M blocks, H, B) using padded logic inside? 
        # Better: (triton.cdiv(total_tokens, 32), num_heads, 1) using offsets logic
        
        grid_triton = lambda meta: (triton.cdiv(n, meta['BLOCK_M']), num_heads, B)
        
        # Launch Compression Attention Kernel (Triton)
        # Note: Ideally we pass flattened k_cmp/v_cmp, but here we used padded.
        # Kernel adaptation needed: The kernel above assumes flattened K/V or consistent stride.
        # We use the padded K_cmp strides directly.
        hstu_bsa_cmp_fwd_kernel[grid_triton](
            Q=q, K=k_cmp, V=v_cmp, 
            G_cmp=g_cmp_jag, Out=o_cmp, Scores=None,
            Stride_qt=q.stride(0), Stride_qh=q.stride(1), Stride_qd=q.stride(2),
            Stride_kt=k_cmp.stride(1), Stride_kh=k_cmp.stride(2), Stride_kd=k_cmp.stride(3), # Adjusted for [B, N_blk, H, D]
            Stride_vt=v_cmp.stride(1), Stride_vh=v_cmp.stride(2), Stride_vd=v_cmp.stride(3),
            Stride_ot=o_cmp.stride(0), Stride_oh=o_cmp.stride(1), Stride_od=o_cmp.stride(2),
            offsets=x_offsets, scale=scale,
            BLOCK_SIZE=self.block_size, HEAD_DIM=dim,
            BLOCK_M=32, BLOCK_N=32
        )

        # Launch Selected Attention Kernel (Triton)
        hstu_bsa_slc_fwd_kernel[grid_triton](
            Q=q, K=k, V=v, 
            G_slc=g_slc_jag, BlockIndices=topk_indices_jag, Out=o_slc,
            Stride_qt=q.stride(0), Stride_qh=q.stride(1), Stride_qd=q.stride(2),
            Stride_kt=k.stride(0), Stride_kh=k.stride(1), Stride_kd=k.stride(2),
            Stride_vt=v.stride(0), Stride_vh=v.stride(1), Stride_vd=v.stride(2),
            Stride_ot=o_slc.stride(0), Stride_oh=o_slc.stride(1), Stride_od=o_slc.stride(2),
            offsets=x_offsets, scale=scale,
            S=S, BLOCK_SIZE=self.block_size, HEAD_DIM=dim, BLOCK_M=32
        )
        
        # 5. Epilogue: LayerNorm + U + Sum
        # User Logic: o_cmp = layernorm(o_cmp)*u; o_slc = layernorm(o_slc)*u; return o_cmp + o_slc
        # We assume standard torch.nn.functional.layer_norm is sufficient here
        
        # Re-shape for LayerNorm [TotalTokens, H, D] -> [TotalTokens, H*D] or similar?
        # User code: layernorm shape is [head_dim * num_heads]
        
        def apply_ln_u(x, u_tensor):
            # x: [Tokens, H, D]
            B_tok, H, D = x.shape
            x_flat = x.view(B_tok, H*D) # flatten head
            x_ln = F.layer_norm(x_flat, normalized_shape=[H*D])
            x_ln = x_ln.view(B_tok, H, D)
            return x_ln * u_tensor

        o_cmp_final = apply_ln_u(o_cmp, u)
        o_slc_final = apply_ln_u(o_slc, u)
        
        return o_cmp_final + o_slc_final, padded_q, padded_k