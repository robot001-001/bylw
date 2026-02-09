# -*- coding: utf-8 -*-
"""
Triton implementation of HSTU with Block Sparse Attention (BSA).
Combines HSTU architecture with Triton-accelerated BSA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple

from triton.language.extra.libdevice import fast_dividef, fast_expf

# Import BSA kernels
try:
    from .triton_bsa_attention import triton_block_sparse_attn
except ImportError:
    # Fallback for direct import
    from triton_bsa_attention import triton_block_sparse_attn


@triton.jit
def _silu(x):
    """SiLU activation function: x * sigmoid(x)"""
    return x * fast_dividef(1.0, 1.0 + fast_expf(-x))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    ],
    # key=["M", "K", "N"],
    key=["M", "N"],
)
@triton.jit
def _layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Layer normalization kernel."""
    pid = tl.program_id(0)
    
    if pid >= M:
        return
    
    # Load input
    x_offsets = pid * N + tl.arange(0, BLOCK_N)
    mask = tl.arange(0, BLOCK_N) < N
    x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x_vals) / N
    
    # Compute variance
    x_centered = x_vals - mean
    var = tl.sum(x_centered * x_centered) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize
    x_norm = x_centered * rstd
    
    # Load weight and bias
    weight_vals = tl.load(weight_ptr + tl.arange(0, BLOCK_N), mask=mask, other=0.0)
    bias_vals = tl.load(bias_ptr + tl.arange(0, BLOCK_N), mask=mask, other=0.0)
    
    # Apply affine transformation
    out_vals = x_norm * weight_vals + bias_vals
    
    # Store outputs
    tl.store(out_ptr + x_offsets, out_vals, mask=mask)
    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    ],
    key=["M", "K", "N"],
)
@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Matrix multiplication kernel with optional bias."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Compute C = A @ B
    for k in range(0, K, BLOCK_K):
        # Load A
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a_vals = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :], mask=a_mask, other=0.0)
        
        # Load B
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b_vals = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :], mask=b_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(a_vals, b_vals)
    
    # Add bias if present
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias_vals[None, :]
    
    # Store result
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=c_mask)


def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6):
    """Triton-accelerated layer normalization."""
    M, N = x.shape
    out = torch.empty_like(x)
    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    
    grid = (M,)
    _layer_norm_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        mean_ptr=mean,
        rstd_ptr=rstd,
        M=M,
        N=N,
        eps=eps,
        BLOCK_M=64,
        BLOCK_N=64,
    )
    
    return out, mean, rstd


def triton_matmul(a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor] = None):
    """Triton-accelerated matrix multiplication."""
    M, K = a.shape
    _, N = b.shape
    
    c = torch.empty(M, N, dtype=a.dtype, device=a.device)
    
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _matmul_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        bias_ptr=bias if bias is not None else torch.empty(0, device=a.device),
        M=M,
        K=K,
        N=N,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        HAS_BIAS=bias is not None,
    )
    
    return c


class TritonHSTUBSAAttention(nn.Module):
    """
    Triton-accelerated HSTU with Block Sparse Attention.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        num_heads: int,
        block_counts: int = 4,
        block_size: int = 32,
        window_size: int = 0,
        dropout_ratio: float = 0.0,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear_hidden_dim = linear_hidden_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.block_counts = block_counts
        self.block_size = block_size
        self.window_size = window_size
        self.dropout_ratio = dropout_ratio
        self.epsilon = epsilon
        
        # UVQK projection
        self.uvqk = nn.Parameter(
            torch.empty(
                embedding_dim,
                linear_hidden_dim * 2 * num_heads + attention_dim * num_heads * 2,
            ).normal_(mean=0, std=0.02)
        )
        
        # Output projection
        self.o = nn.Linear(
            linear_hidden_dim * num_heads,
            embedding_dim,
        )
        nn.init.xavier_uniform_(self.o.weight)
        
        # Layer norm weights
        self.norm_weight = nn.Parameter(torch.ones(embedding_dim))
        self.norm_bias = nn.Parameter(torch.zeros(embedding_dim))
    
    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        invalid_attn_mask: torch.Tensor,
        gate_model: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [sum(N_i), D]
            x_offsets: Sequence offsets [B+1]
            invalid_attn_mask: Attention mask [B, N, N]
            gate_model: Gate model for generating g_cmp, g_slc, g_swa
        
        Returns:
            output: Output tensor [sum(N_i), D]
            padded_q: Padded query [B, N, H, ATTN_DIM]
            padded_k: Padded key [B, N, H, ATTN_DIM]
        """
        B = x_offsets.size(0) - 1
        n = invalid_attn_mask.size(-1)
        
        # Convert jagged to padded
        padded_x = torch.ops.fbgemm.jagged_to_padded_dense(
            values=x, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
        )  # [B, n, D]
        
        # Layer normalization
        padded_x_flat = padded_x.view(B * n, self.embedding_dim)
        normed_x_flat, _, _ = triton_layer_norm(
            padded_x_flat,
            self.norm_weight,
            self.norm_bias,
            eps=self.epsilon,
        )
        normed_x = normed_x_flat.view(B, n, self.embedding_dim)
        
        # UVQK projection
        normed_x_flat = normed_x.view(B * n, self.embedding_dim)
        uvqk_flat = triton_matmul(normed_x_flat, self.uvqk)
        uvqk = uvqk_flat.view(B, n, -1)
        
        # Split into u, v, q, k
        u, v, q, k = torch.split(
            uvqk,
            [
                self.linear_hidden_dim * self.num_heads,
                self.linear_hidden_dim * self.num_heads,
                self.attention_dim * self.num_heads,
                self.attention_dim * self.num_heads,
            ],
            dim=-1,
        )
        
        # Reshape for multi-head attention
        q = q.view(B, n, self.num_heads, self.attention_dim)
        k = k.view(B, n, self.num_heads, self.attention_dim)
        v = v.view(B, n, self.num_heads, self.linear_hidden_dim)
        u = u.view(B, n, self.num_heads, self.linear_hidden_dim)
        
        # Generate gates
        g_cmp, g_slc, g_swa = gate_model(q)
        
        # Block Sparse Attention
        attn_output, block_indices = triton_block_sparse_attn(
            q=q,
            k=k,
            v=v,
            u=u,
            g_cmp=g_cmp,
            g_slc=g_slc,
            g_swa=g_swa,
            block_counts=self.block_counts,
            block_size=self.block_size,
            window_size=self.window_size,
        )
        
        # Reshape and convert back to jagged
        attn_output = attn_output.reshape(B, n, self.num_heads * self.linear_hidden_dim)
        attn_output_jagged = torch.ops.fbgemm.dense_to_jagged(
            attn_output,
            [x_offsets],
        )[0]
        
        # Output projection
        output = self.o(
            F.dropout(attn_output_jagged, p=self.dropout_ratio, training=self.training)
        )
        
        return output, q, k


def triton_hstu_attention_with_bsa(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    u: torch.Tensor,
    x_offsets: torch.Tensor,
    invalid_attn_mask: torch.Tensor,
    gate_model: nn.Module,
    block_counts: int = 4,
    block_size: int = 32,
    window_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Triton-accelerated HSTU attention with BSA.
    
    Args:
        num_heads: Number of attention heads
        attention_dim: Attention dimension
        linear_dim: Linear hidden dimension
        q: Query tensor [sum(N_i), num_heads * attention_dim]
        k: Key tensor [sum(N_i), num_heads * attention_dim]
        v: Value tensor [sum(N_i), num_heads * linear_dim]
        u: U tensor [sum(N_i), num_heads * linear_dim]
        x_offsets: Sequence offsets [B+1]
        invalid_attn_mask: Attention mask [B, N, N]
        gate_model: Gate model
        block_counts: Number of blocks to select
        block_size: Size of each block
        window_size: Sliding window size
    
    Returns:
        attn_output: Attention output [sum(N_i), num_heads * linear_dim]
        padded_q: Padded query [B, N, H, ATTN_DIM]
        padded_k: Padded key [B, N, H, ATTN_DIM]
    """
    B = x_offsets.size(0) - 1
    n = invalid_attn_mask.size(-1)
    
    # Convert jagged to padded
    padded_q = torch.ops.fbgemm.jagged_to_padded_dense(
        values=q, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
    ).view(B, n, num_heads, attention_dim)
    
    padded_k = torch.ops.fbgemm.jagged_to_padded_dense(
        values=k, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
    ).view(B, n, num_heads, attention_dim)
    
    padded_v = torch.ops.fbgemm.jagged_to_padded_dense(
        values=v, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
    ).view(B, n, num_heads, linear_dim)
    
    padded_u = torch.ops.fbgemm.jagged_to_padded_dense(
        values=u, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
    ).view(B, n, num_heads, linear_dim)
    
    # Generate gates
    g_cmp, g_slc, g_swa = gate_model(padded_q)
    
    # Block Sparse Attention
    attn_output, block_indices = triton_block_sparse_attn(
        q=padded_q,
        k=padded_k,
        v=padded_v,
        u=padded_u,
        g_cmp=g_cmp,
        g_slc=g_slc,
        g_swa=g_swa,
        block_counts=block_counts,
        block_size=block_size,
        window_size=window_size,
    )
    
    # Convert back to jagged
    attn_output = attn_output.reshape(B, n, num_heads * linear_dim)
    attn_output = torch.ops.fbgemm.dense_to_jagged(
        attn_output,
        [x_offsets],
    )[0]
    
    return attn_output, padded_q, padded_k

