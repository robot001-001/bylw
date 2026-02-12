import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Tuple

def layernorm(x, eps=1e-6):
    bsize, seq_len, num_heads, head_dim = x.shape
    return F.layer_norm(
        x.reshape(bsize, seq_len, -1), normalized_shape=[head_dim * num_heads], eps=eps
    ).reshape(bsize, seq_len, num_heads, head_dim)


# @torch.compile
def compression(
    k: torch.Tensor, # [2, 1024, 8, 32]
    v: torch.Tensor, # [2, 1024, 8, 64]
    block_size: int # 64
) -> torch.Tensor:
    B, T, H = k.shape[:3]
    num_block = math.ceil(T / block_size)
    if k.shape[1] % block_size != 0:
        k = F.pad(k, (0, 0, 0, 0, 0, num_block * block_size - T))
        v = F.pad(v, (0, 0, 0, 0, 0, num_block * block_size - T))
    k_cmp = k.view(B, num_block, block_size, H, -1).mean(dim=2) # [2, 8, 8, 32]
    v_cmp = v.view(B, num_block, block_size, H, -1).mean(dim=2) # [2, 8, 8, 64]
    return k_cmp, v_cmp


def bsa_compression(
    q, k, v, u,
    g_cmp,
    block_counts, block_size,
    scale
):
    bsize, seq_len, num_heads, attn_dim = q.shape # [2, 1024, 8, 32]
    _, _, _, linear_dim = v.shape # [2, 1024, 8, 64]
    BS = block_size
    block_counts = torch.full(
        (bsize, seq_len, num_heads), # [2, 1024, 8] 
        block_counts, dtype=torch.long, device=q.device
    )
    q, k, v, u = map(lambda x: x.float(), (q, k, v, u))
    k_cmp, v_cmp = compression(k, v, BS) # [bsize, seq_len/block_size, num_heads, head_dim]
    C = k_cmp.shape[1] # seq_len/block_size
    S = min(block_counts.max().item(), C)

    casual_mask = ((torch.arange(seq_len) - BS + 1)[:, None] // BS < torch.arange(C)[None, :]).to(q.device)
    empty_mask = casual_mask.all(-1, True)
    local_mask = (torch.arange(seq_len)[:, None] // BS == torch.arange(C)[None, :]).to(q.device)

    attn_cmp = torch.einsum('bqhd,bkhd->bhqk', q*scale, k_cmp)
    attn_cmp = attn_cmp.masked_fill(casual_mask & empty_mask.logical_not(), 0)
    attn_cmp = F.silu(attn_cmp) / scale
    o_cmp = torch.einsum('bhqk, bkhd -> bqhd', attn_cmp, v_cmp) * g_cmp.unsqueeze(-1)
    # [bsize, seq_len, num_heads, linear_dim]

    o_cmp = layernorm(o_cmp)*u
    attn_select = attn_cmp.masked_fill(local_mask, float(1.0))
    block_indices = attn_select.topk(S, -1)[1]
    block_indices = block_indices.masked_fill(block_indices > (block_indices.new_tensor(range(seq_len))[:, None] // BS), -1)
    # [bsize, num_heads, seq_len, block_counts]
    return block_indices, o_cmp.to(q.dtype)


def bsa_cal(
    q, k, v, u,
    g_slc,
    block_indices,
    block_counts, block_size,
    scale
):
    bsize, seq_len, num_heads, head_dim = q.shape
    S = block_indices.shape[-1] # 每个 token 选多少个 block
    BS = block_size
    q, k, v, u = map(lambda x: x.float(), (q, k, v, u))
    offsets = torch.arange(BS, device=q.device).view(1, 1, 1, 1, BS)
    start_indices = block_indices.unsqueeze(-1) * BS
    gather_ids = start_indices + offsets
    gather_ids = gather_ids.view(bsize, seq_len, num_heads, S * BS)
    valid_mask = (gather_ids >= 0) & (gather_ids < seq_len)
    safe_gather_ids = gather_ids.clamp(0, seq_len - 1)
    b_idx = torch.arange(bsize, device=q.device).view(bsize, 1, 1, 1)
    h_idx = torch.arange(num_heads, device=q.device).view(1, 1, num_heads, 1)
    k_slc = k[b_idx, safe_gather_ids, h_idx, :]
    v_slc = v[b_idx, safe_gather_ids, h_idx, :]

    q_unsq = q.unsqueeze(3)
    attn_logits = torch.matmul(q_unsq, k_slc.transpose(-1, -2)).squeeze(3)
    mask = ~valid_mask
    current_t = torch.arange(seq_len, device=q.device).view(1, seq_len, 1, 1)
    mask = mask | (gather_ids > current_t)
    attn_logits = attn_logits.masked_fill(mask, 0)
    attn_weights = F.silu(attn_logits) / scale
    o_slc = torch.matmul(attn_weights.unsqueeze(3), v_slc).squeeze(3)
    o_slc = layernorm(o_slc)*u
    return o_slc


def bsa_score(o_cmp, o_slc, o_swa):
    return o_cmp + o_slc


def block_sparse_attn(
    q, k, v, u,
    g_cmp, g_slc, g_swa,
    block_counts, block_size,
    window_size
):
    # qk: [bsize, seq_len, num_heads, attn_dim]
    # vu: [bsize, seq_len, num_heads, linear_dim]
    scale = k.shape[-1]
    block_indices, o_cmp = bsa_compression(
        q, k, v, u,
        g_cmp,
        block_counts, block_size, scale
    )
    block_indices[:, 1::2] = block_indices[:, 0::2]
    o_slc = bsa_cal(
        q, k, v, u,
        g_slc, 
        block_indices, 
        block_counts, block_size, scale
    )
    o = bsa_score(o_cmp, o_slc, None)
    return o, block_indices




def _hstu_attention_with_bsa(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    u: torch.Tensor,
    g_cmp: int,
    g_slc: int,
    g_swa: int,
    x_offsets: torch.Tensor,
    invalid_attn_mask: torch.Tensor,
    gate_model: nn.Module
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B: int = x_offsets.size(0) - 1
    n: int = invalid_attn_mask.size(-1)
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

    attn_output, block_indices = block_sparse_attn(
        q=padded_q,
        k=padded_k,
        v=padded_v,
        u=padded_u,
        g_cmp=g_cmp,
        g_slc=g_slc,
        g_swa=g_swa,
        block_counts=4,
        block_size=32,
        window_size=0
    )
    attn_output = attn_output.reshape(B, n, num_heads * linear_dim)
    attn_output = torch.ops.fbgemm.dense_to_jagged(
        attn_output,
        [x_offsets],
    )[0]
    
    return attn_output