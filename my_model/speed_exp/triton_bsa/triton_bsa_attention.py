# -*- coding: utf-8 -*-
"""
Triton implementation of Block Sparse Attention (BSA) for HSTU.
Based on HSTU Triton implementation and Native Sparse Attention implementation.
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

from triton.language.extra.libdevice import fast_dividef, fast_expf


@triton.jit
def _silu(x):
    """SiLU activation function: x * sigmoid(x)"""
    return x * fast_dividef(1.0, 1.0 + fast_expf(-x))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    ],
    key=["B", "T", "H", "ATTN_DIM", "LINEAR_DIM"],
)
@triton.jit
def _bsa_compression_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    u_ptr,
    k_cmp_ptr,
    v_cmp_ptr,
    o_cmp_ptr,
    g_cmp_ptr,
    attn_scores_ptr,
    block_indices_ptr,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    ATTN_DIM: tl.constexpr,
    LINEAR_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    SCALE: tl.constexpr,
):
    """
    BSA Compression kernel: Compute compressed attention and select top-k blocks.
    
    Args:
        q_ptr: Query tensor [B, T, H, ATTN_DIM]
        k_ptr: Key tensor [B, T, H, ATTN_DIM]
        v_ptr: Value tensor [B, T, H, LINEAR_DIM]
        u_ptr: U tensor [B, T, H, LINEAR_DIM]
        k_cmp_ptr: Compressed key output [B, C, H, ATTN_DIM] where C = T // BLOCK_SIZE
        v_cmp_ptr: Compressed value output [B, C, H, LINEAR_DIM]
        o_cmp_ptr: Compressed attention output [B, T, H, LINEAR_DIM]
        g_cmp_ptr: Gate scores for compression [B, T, H]
        attn_scores_ptr: Attention scores [B, H, T, C]
        block_indices_ptr: Selected block indices [B, H, T, S]
    """
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    # Compute compressed k and v (mean pooling over blocks)
    C = tl.cdiv(T, BLOCK_SIZE)
    
    # Load compressed k and v (computed in separate kernel or pre-computed)
    # For now, we'll compute them here
    if pid_t < C:
        # Compute mean pooling for k_cmp and v_cmp
        k_acc = tl.zeros([BLOCK_D_Q], dtype=tl.float32)
        v_acc = tl.zeros([BLOCK_D_V], dtype=tl.float32)
        count = 0
        
        start_idx = pid_t * BLOCK_SIZE
        end_idx = tl.minimum(start_idx + BLOCK_SIZE, T)
        
        for t_idx in range(start_idx, end_idx):
            k_offsets = pid_b * T * H * ATTN_DIM + t_idx * H * ATTN_DIM + pid_h * ATTN_DIM
            v_offsets = pid_b * T * H * LINEAR_DIM + t_idx * H * LINEAR_DIM + pid_h * LINEAR_DIM
            
            k_vals = tl.load(k_ptr + k_offsets + tl.arange(0, BLOCK_D_Q))
            v_vals = tl.load(v_ptr + v_offsets + tl.arange(0, BLOCK_D_V))
            
            k_acc += k_vals
            v_acc += v_vals
            count += 1
        
        if count > 0:
            k_acc = k_acc / count
            v_acc = v_acc / count
            
            k_cmp_offsets = pid_b * C * H * ATTN_DIM + pid_t * H * ATTN_DIM + pid_h * ATTN_DIM
            v_cmp_offsets = pid_b * C * H * LINEAR_DIM + pid_t * H * LINEAR_DIM + pid_h * LINEAR_DIM
            
            tl.store(k_cmp_ptr + k_cmp_offsets + tl.arange(0, BLOCK_D_Q), k_acc.to(k_cmp_ptr.dtype.element_ty))
            tl.store(v_cmp_ptr + v_cmp_offsets + tl.arange(0, BLOCK_D_V), v_acc.to(v_cmp_ptr.dtype.element_ty))
    
    # Compute compressed attention scores
    if pid_t < T:
        q_offsets = pid_b * T * H * ATTN_DIM + pid_t * H * ATTN_DIM + pid_h * ATTN_DIM
        q_vals = tl.load(q_ptr + q_offsets + tl.arange(0, BLOCK_D_Q))
        
        # Compute attention scores with compressed keys
        attn_acc = tl.zeros([C], dtype=tl.float32)
        
        for c_idx in range(C):
            # Causal mask: only attend to blocks before current position
            block_start = c_idx * BLOCK_SIZE
            if block_start <= pid_t:
                k_cmp_offsets = pid_b * C * H * ATTN_DIM + c_idx * H * ATTN_DIM + pid_h * ATTN_DIM
                k_cmp_vals = tl.load(k_cmp_ptr + k_cmp_offsets + tl.arange(0, BLOCK_D_Q))
                
                # Compute attention score: q @ k_cmp
                attn_score = tl.sum(q_vals * k_cmp_vals) * SCALE
                attn_acc = attn_acc + tl.where(c_idx <= pid_t // BLOCK_SIZE, attn_score, 0.0)
        
        # Apply SiLU activation
        attn_silu = _silu(attn_acc) / SCALE
        
        # Store attention scores
        attn_offsets = pid_b * H * T * C + pid_h * T * C + pid_t * C
        tl.store(attn_scores_ptr + attn_offsets + tl.arange(0, C), attn_silu.to(attn_scores_ptr.dtype.element_ty))
        
        # Compute compressed output: attn_silu @ v_cmp
        o_cmp_acc = tl.zeros([BLOCK_D_V], dtype=tl.float32)
        
        for c_idx in range(C):
            if c_idx <= pid_t // BLOCK_SIZE:
                v_cmp_offsets = pid_b * C * H * LINEAR_DIM + c_idx * H * LINEAR_DIM + pid_h * LINEAR_DIM
                v_cmp_vals = tl.load(v_cmp_ptr + v_cmp_offsets + tl.arange(0, BLOCK_D_V))
                
                attn_val = tl.load(attn_scores_ptr + attn_offsets + c_idx)
                o_cmp_acc += attn_val * v_cmp_vals
        
        # Apply gate and u
        g_cmp_val = tl.load(g_cmp_ptr + pid_b * T * H + pid_t * H + pid_h)
        u_offsets = pid_b * T * H * LINEAR_DIM + pid_t * H * LINEAR_DIM + pid_h * LINEAR_DIM
        u_vals = tl.load(u_ptr + u_offsets + tl.arange(0, BLOCK_D_V))
        
        o_cmp = o_cmp_acc * g_cmp_val * u_vals
        
        o_cmp_offsets = pid_b * T * H * LINEAR_DIM + pid_t * H * LINEAR_DIM + pid_h * LINEAR_DIM
        tl.store(o_cmp_ptr + o_cmp_offsets + tl.arange(0, BLOCK_D_V), o_cmp.to(o_cmp_ptr.dtype.element_ty))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    ],
    key=["B", "T", "H", "ATTN_DIM", "LINEAR_DIM", "S"],
)
@triton.jit
def _bsa_selection_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    u_ptr,
    o_slc_ptr,
    g_slc_ptr,
    block_indices_ptr,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    ATTN_DIM: tl.constexpr,
    LINEAR_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    SCALE: tl.constexpr,
):
    """
    BSA Selection kernel: Compute attention with selected blocks.
    
    Args:
        q_ptr: Query tensor [B, T, H, ATTN_DIM]
        k_ptr: Key tensor [B, T, H, ATTN_DIM]
        v_ptr: Value tensor [B, T, H, LINEAR_DIM]
        u_ptr: U tensor [B, T, H, LINEAR_DIM]
        o_slc_ptr: Selected attention output [B, T, H, LINEAR_DIM]
        g_slc_ptr: Gate scores for selection [B, T, H]
        block_indices_ptr: Selected block indices [B, H, T, S]
    """
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    if pid_t >= T:
        return
    
    # Load query
    q_offsets = pid_b * T * H * ATTN_DIM + pid_t * H * ATTN_DIM + pid_h * ATTN_DIM
    q_vals = tl.load(q_ptr + q_offsets + tl.arange(0, BLOCK_D_Q))
    
    # Load block indices for this token
    block_idx_offsets = pid_b * H * T * S + pid_h * T * S + pid_t * S
    block_indices = tl.load(block_indices_ptr + block_idx_offsets + tl.arange(0, S))
    
    # Gather k and v from selected blocks
    o_slc_acc = tl.zeros([BLOCK_D_V], dtype=tl.float32)
    attn_sum = 0.0
    
    for s_idx in range(S):
        block_idx = tl.load(block_indices_ptr + block_idx_offsets + s_idx)
        
        # Skip invalid blocks
        if block_idx < 0:
            continue
        
        block_start = block_idx * BLOCK_SIZE
        
        # Process each token in the selected block
        for block_offset in range(BLOCK_SIZE):
            t_idx = block_start + block_offset
            
            # Causal mask: only attend to tokens before current position
            if t_idx > pid_t or t_idx < 0:
                continue
            
            # Load k and v
            k_offsets = pid_b * T * H * ATTN_DIM + t_idx * H * ATTN_DIM + pid_h * ATTN_DIM
            v_offsets = pid_b * T * H * LINEAR_DIM + t_idx * H * LINEAR_DIM + pid_h * LINEAR_DIM
            
            k_vals = tl.load(k_ptr + k_offsets + tl.arange(0, BLOCK_D_Q))
            v_vals = tl.load(v_ptr + v_offsets + tl.arange(0, BLOCK_D_V))
            
            # Compute attention score
            attn_score = tl.sum(q_vals * k_vals) * SCALE
            attn_silu = _silu(attn_score) / SCALE
            
            # Accumulate output
            o_slc_acc += attn_silu * v_vals
            attn_sum += attn_silu
    
    # Normalize (layer norm approximation)
    if attn_sum > 0:
        o_slc_acc = o_slc_acc / attn_sum
    
    # Apply gate and u
    g_slc_val = tl.load(g_slc_ptr + pid_b * T * H + pid_t * H + pid_h)
    u_offsets = pid_b * T * H * LINEAR_DIM + pid_t * H * LINEAR_DIM + pid_h * LINEAR_DIM
    u_vals = tl.load(u_ptr + u_offsets + tl.arange(0, BLOCK_D_V))
    
    o_slc = o_slc_acc * g_slc_val * u_vals
    
    o_slc_offsets = pid_b * T * H * LINEAR_DIM + pid_t * H * LINEAR_DIM + pid_h * LINEAR_DIM
    tl.store(o_slc_ptr + o_slc_offsets + tl.arange(0, BLOCK_D_V), o_slc.to(o_slc_ptr.dtype.element_ty))


def triton_bsa_compression_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    u: torch.Tensor,
    g_cmp: torch.Tensor,
    block_counts: int,
    block_size: int,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for BSA compression.
    
    Args:
        q: Query tensor [B, T, H, ATTN_DIM]
        k: Key tensor [B, T, H, ATTN_DIM]
        v: Value tensor [B, T, H, LINEAR_DIM]
        u: U tensor [B, T, H, LINEAR_DIM]
        g_cmp: Gate scores for compression [B, T, H]
        block_counts: Number of blocks to select
        block_size: Size of each block
        scale: Attention scale factor (default: sqrt(ATTN_DIM))
    
    Returns:
        o_cmp: Compressed attention output [B, T, H, LINEAR_DIM]
        block_indices: Selected block indices [B, H, T, S]
        attn_scores: Attention scores [B, H, T, C]
    """
    B, T, H, ATTN_DIM = q.shape
    _, _, _, LINEAR_DIM = v.shape
    C = (T + block_size - 1) // block_size
    
    if scale is None:
        scale = ATTN_DIM ** 0.5
    
    # Allocate outputs
    k_cmp = torch.zeros(B, C, H, ATTN_DIM, dtype=k.dtype, device=k.device)
    v_cmp = torch.zeros(B, C, H, LINEAR_DIM, dtype=v.dtype, device=v.device)
    o_cmp = torch.zeros_like(v)
    attn_scores = torch.zeros(B, H, T, C, dtype=torch.float32, device=q.device)
    block_indices = torch.zeros(B, H, T, block_counts, dtype=torch.int32, device=q.device)
    
    # Launch kernel
    grid = (B, T, H)
    
    _bsa_compression_kernel[grid](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        u_ptr=u,
        k_cmp_ptr=k_cmp,
        v_cmp_ptr=v_cmp,
        o_cmp_ptr=o_cmp,
        g_cmp_ptr=g_cmp,
        attn_scores_ptr=attn_scores,
        block_indices_ptr=block_indices,
        B=B,
        T=T,
        H=H,
        ATTN_DIM=ATTN_DIM,
        LINEAR_DIM=LINEAR_DIM,
        BLOCK_SIZE=block_size,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_D_Q=ATTN_DIM,
        BLOCK_D_V=LINEAR_DIM,
        SCALE=scale,
    )
    
    # Select top-k blocks (simplified version, full implementation would use bitonic sort)
    # For now, we'll do this on CPU/GPU with PyTorch
    attn_scores_reshaped = attn_scores.permute(0, 2, 1, 3)  # [B, T, H, C]
    
    # Set local mask (current block gets score 1.0)
    local_mask = torch.arange(T, device=q.device)[:, None] // block_size == torch.arange(C, device=q.device)[None, :]
    attn_scores_reshaped = attn_scores_reshaped.masked_fill(local_mask.unsqueeze(0).unsqueeze(2), 1.0)
    
    # Select top-k blocks
    _, topk_indices = torch.topk(attn_scores_reshaped, block_counts, dim=-1)
    block_indices = topk_indices.permute(0, 2, 1, 3)  # [B, H, T, S]
    
    # Apply interleave pattern (alternate blocks share indices)
    block_indices[:, :, 1::2] = block_indices[:, :, 0::2]
    
    return o_cmp, block_indices, attn_scores


def triton_bsa_selection_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    u: torch.Tensor,
    g_slc: torch.Tensor,
    block_indices: torch.Tensor,
    block_size: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Forward pass for BSA selection.
    
    Args:
        q: Query tensor [B, T, H, ATTN_DIM]
        k: Key tensor [B, T, H, ATTN_DIM]
        v: Value tensor [B, T, H, LINEAR_DIM]
        u: U tensor [B, T, H, LINEAR_DIM]
        g_slc: Gate scores for selection [B, T, H]
        block_indices: Selected block indices [B, H, T, S]
        block_size: Size of each block
        scale: Attention scale factor
    
    Returns:
        o_slc: Selected attention output [B, T, H, LINEAR_DIM]
    """
    B, T, H, ATTN_DIM = q.shape
    _, _, _, LINEAR_DIM = v.shape
    S = block_indices.shape[-1]
    
    if scale is None:
        scale = ATTN_DIM ** 0.5
    
    o_slc = torch.zeros_like(v)
    
    grid = (B, T, H)
    
    _bsa_selection_kernel[grid](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        u_ptr=u,
        o_slc_ptr=o_slc,
        g_slc_ptr=g_slc,
        block_indices_ptr=block_indices,
        B=B,
        T=T,
        H=H,
        ATTN_DIM=ATTN_DIM,
        LINEAR_DIM=LINEAR_DIM,
        BLOCK_SIZE=block_size,
        S=S,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_D_Q=ATTN_DIM,
        BLOCK_D_V=LINEAR_DIM,
        SCALE=scale,
    )
    
    return o_slc


def triton_block_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    u: torch.Tensor,
    g_cmp: torch.Tensor,
    g_slc: torch.Tensor,
    g_swa: Optional[torch.Tensor],
    block_counts: int,
    block_size: int,
    window_size: int = 0,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Complete Block Sparse Attention forward pass.
    
    Args:
        q: Query tensor [B, T, H, ATTN_DIM]
        k: Key tensor [B, T, H, ATTN_DIM]
        v: Value tensor [B, T, H, LINEAR_DIM]
        u: U tensor [B, T, H, LINEAR_DIM]
        g_cmp: Gate scores for compression [B, T, H]
        g_slc: Gate scores for selection [B, T, H]
        g_swa: Gate scores for sliding window (optional)
        block_counts: Number of blocks to select
        block_size: Size of each block
        window_size: Sliding window size (not implemented yet)
        scale: Attention scale factor
    
    Returns:
        o: Output tensor [B, T, H, LINEAR_DIM]
        block_indices: Selected block indices [B, H, T, S]
    """
    # Compression phase
    o_cmp, block_indices, _ = triton_bsa_compression_fwd(
        q=q,
        k=k,
        v=v,
        u=u,
        g_cmp=g_cmp,
        block_counts=block_counts,
        block_size=block_size,
        scale=scale,
    )
    
    # Selection phase
    o_slc = triton_bsa_selection_fwd(
        q=q,
        k=k,
        v=v,
        u=u,
        g_slc=g_slc,
        block_indices=block_indices,
        block_size=block_size,
        scale=scale,
    )
    
    # Combine outputs
    o = o_cmp + o_slc
    
    # TODO: Add sliding window attention if window_size > 0
    
    return o, block_indices

