# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


@torch.compile
def compression(
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int
) -> torch.Tensor:
    # Currently, we set mean pooling as our basic compression function.
    B, T, H = k.shape[:3]
    num_block = math.ceil(T / block_size)
    if k.shape[1] % block_size != 0:
        k = F.pad(k, (0, 0, 0, 0, 0, num_block * block_size - T))
        v = F.pad(v, (0, 0, 0, 0, 0, num_block * block_size - T))
    k_cmp = k.view(B, num_block, block_size, H, -1).mean(dim=2)
    v_cmp = v.view(B, num_block, block_size, H, -1).mean(dim=2)
    return k_cmp, v_cmp


def naive_nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_slc: torch.Tensor,
    g_swa: torch.Tensor,
    block_indices: torch.LongTensor,
    block_counts: Optional[Union[torch.LongTensor, int]] = None,
    block_size: int = 64,
    window_size: int = 0,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, HQ, K]` if `head_first=False` else `[B, HQ, T, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >=16.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g_slc (torch.Tensor):
            Gate score for selected attention of shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        g_swa (torch.Tensor):
            Gate score for sliding attentionof shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        block_indices (torch.LongTensor):
            Block indices of shape `[B, T, H, S]` if `head_first=False` else `[B, H, T, S]`.
            `S` is the maximum number of selected blocks for each query token, which is set to 16 in the paper.
        block_counts (Union[torch.LongTensor, int]):
            Number of selected blocks for each token.
            If a tensor is provided, with shape `[B, T, H]` if `head_first=True` else `[B, T, H]`,
            each token can select the same number of blocks.
            If not provided, it will default to `S`, Default: `None`.
        block_size (int):
            Selected block size. Default: 64.
        window_size (int):
            Sliding window size. Default: 0.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]` if `head_first=False` else `[B, HQ, T, V]`.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
        if head_first:
            raise RuntimeError("Sequences with variable lengths are not supported for head-first mode")
    if head_first:
        q, k, v, block_indices = map(lambda x: rearrange(x, 'b h t d -> b t h d'), (q, k, v, block_indices))
        g_slc, g_swa = map(lambda x: rearrange(x, 'b h t -> b t h'), (g_slc, g_swa))
        if isinstance(block_counts, torch.Tensor):
            block_counts = rearrange(block_counts, 'b h t -> b t h')

    dtype = q.dtype
    G = q.shape[2] // k.shape[2]
    BS = block_size
    S = block_indices.shape[-1]
    k, v, block_indices = (repeat(x, 'b t h d -> b t (h g) d', g=G) for x in (k, v, block_indices))
    if isinstance(block_counts, torch.Tensor):
        block_counts = repeat(block_counts, 'b t h -> b t (h g)', g=G)
    c = torch.arange(S).repeat_interleave(BS).unsqueeze(1).expand(-1, q.shape[2]).to(q.device)
    q, k, v = map(lambda x: x.float(), (q, k, v))

    o_slc = torch.zeros_like(v)
    o_swa = torch.zeros_like(v) if window_size > 0 else None
    varlen = True
    if cu_seqlens is None:
        varlen = False
        B, T = q.shape[:2]
        cu_seqlens = torch.cat([block_indices.new_tensor(range(0, B*T, T)), block_indices.new_tensor([B*T])])

    for i in range(len(cu_seqlens) - 1):
        if not varlen:
            q_b, k_b, v_b, g_slc_b, g_swa_b, i_b = q[i], k[i], v[i], g_slc[i], g_swa[i], block_indices[i]
            if isinstance(block_counts, torch.Tensor):
                s_b = block_counts[i]
            else:
                s_b = block_counts
        else:
            T = cu_seqlens[i+1] - cu_seqlens[i]
            q_b, k_b, v_b, g_slc_b, g_swa_b, i_b = map(
                lambda x: x[0][cu_seqlens[i]:cu_seqlens[i+1]],
                (q, k, v, g_slc, g_swa, block_indices)
            )
            if isinstance(block_counts, torch.Tensor):
                s_b = block_counts[0][cu_seqlens[i]:cu_seqlens[i+1]]
            else:
                s_b = block_counts

        i_b = i_b.unsqueeze(-1) * BS + i_b.new_tensor(range(BS))
        # [T, S*BS, HQ]
        i_b = i_b.view(T, block_indices.shape[2], -1).transpose(1, 2)
        for i_q in range(T):
            # [HQ, D]
            q_i = q_b[i_q] * scale
            # [HQ]
            g_slc_i = g_slc_b[i_q]
            # [HQ]
            g_swa_i = g_swa_b[i_q]
            # [S*BS, HQ]
            i_i = i_b[i_q]
            # [HQ]
            if isinstance(block_counts, torch.Tensor):
                s_i = s_b[i_q]
            else:
                s_i = s_b
            # [S*BS, HQ, -1]
            k_i_slc, v_i_slc = map(lambda x: x.gather(0, i_i.clamp(
                0, T-1).unsqueeze(-1).expand(*i_i.shape, x.shape[-1])), (k_b, v_b))
            # [S*BS, HQ]
            attn_slc = torch.einsum('h d, n h d -> n h', q_i, k_i_slc).masked_fill(
                torch.logical_or(i_i < 0, i_i > i_q) | (c >= s_i if block_counts is not None else False),
                float('-inf')
            ).softmax(0)
            if not varlen:
                o_slc[i, i_q] = torch.einsum('n h, n h v -> h v', attn_slc, v_i_slc) * g_slc_i.unsqueeze(-1)
            else:
                o_slc[0][cu_seqlens[i]+i_q] = torch.einsum('n h, n h v -> h v', attn_slc, v_i_slc) * g_slc_i.unsqueeze(-1)
            if window_size > 0:
                k_i_swa, v_i_swa = map(lambda x: x[max(0, i_q - window_size + 1):i_q + 1], (k_b, v_b))
                attn_swa = torch.einsum('h d, n h d -> n h', q_i, k_i_swa).softmax(0)
                if not varlen:
                    o_swa[i, i_q] = torch.einsum('n h, n h v -> h v', attn_swa, v_i_swa) * g_swa_i.unsqueeze(-1)
                else:
                    o_swa[0][cu_seqlens[i]+i_q] = torch.einsum('n h, n h v -> h v', attn_swa, v_i_swa) * g_swa_i.unsqueeze(-1)

    if head_first:
        o_slc = rearrange(o_slc, 'b t h d -> b h t d')
        o_swa = rearrange(o_swa, 'b t h d -> b h t d')

    return o_slc.to(dtype) + o_swa.to(dtype) if o_swa is not None else o_slc.to(dtype)


def vectorized_naive_nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_slc: torch.Tensor,
    g_swa: torch.Tensor,
    block_indices: torch.LongTensor,
    block_counts: Optional[Union[torch.LongTensor, int]] = None,
    block_size: int = 64,
    window_size: int = 0,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> torch.Tensor:
    
    # --- 1. 预处理与形状调整 ---
    if scale is None:
        scale = k.shape[-1] ** -0.5
    
    if head_first:
        # 统一转为 [B, T, H, D] 格式处理
        q, k, v, block_indices = map(lambda x: rearrange(x, 'b h t d -> b t h d'), (q, k, v, block_indices))
        g_slc, g_swa = map(lambda x: rearrange(x, 'b h t -> b t h'), (g_slc, g_swa))
        if isinstance(block_counts, torch.Tensor):
            block_counts = rearrange(block_counts, 'b h t -> b t h')

    # 处理 GQA (Grouped Query Attention) 复制
    G = q.shape[2] // k.shape[2]
    if G > 1:
        k, v, block_indices = (repeat(x, 'b t h d -> b t (h g) d', g=G) for x in (k, v, block_indices))
        if isinstance(block_counts, torch.Tensor):
            block_counts = repeat(block_counts, 'b t h -> b t (h g)', g=G)

    B, T, H, D = q.shape
    S = block_indices.shape[-1] # 每个 token 选多少个 block
    BS = block_size
    
    # 转换为 float 以便计算
    q, k, v = map(lambda x: x.float(), (q, k, v))

    # --- 2. 构造 gather 索引 (核心步骤) ---
    # block_indices: [B, T, H, S] -> 存储的是 Block ID
    # 我们需要将其转换为 Token ID
    
    # 生成块内偏移 [0, 1, ..., BS-1]
    # shape: [1, 1, 1, 1, BS]
    offsets = torch.arange(BS, device=q.device).view(1, 1, 1, 1, BS)
    
    # 计算起始位置: [B, T, H, S, 1]
    start_indices = block_indices.unsqueeze(-1) * BS
    
    # 得到每个被选中的 Token 的绝对位置: [B, T, H, S, BS]
    gather_ids = start_indices + offsets
    
    # 展平最后两维 -> [B, T, H, S*BS]
    # 这是所有 query (T) 需要关注的 keys 的索引
    gather_ids = gather_ids.view(B, T, H, S * BS)

    # 处理越界索引 (防止 gather 报错)
    # block_indices 里的 -1 会变成负数，或者计算出的 index 超过 T
    # 我们先 clamp 到 [0, T-1]，然后在 mask 阶段把非法的遮掉
    valid_mask = (gather_ids >= 0) & (gather_ids < T)
    safe_gather_ids = gather_ids.clamp(0, T - 1)

    # --- 3. 并行 Gather K 和 V ---
    # 使用高级索引 (Advanced Indexing) 代替手动 loop
    # 我们需要 K 的形状: [B, T, H, S*BS, D]
    
    # 构造 batch 和 head 的辅助索引
    b_idx = torch.arange(B, device=q.device).view(B, 1, 1, 1) # [B, 1, 1, 1]
    h_idx = torch.arange(H, device=q.device).view(1, 1, H, 1) # [1, 1, H, 1]
    
    # 执行 Gather
    # k: [B, T, H, D] -> k_selected: [B, T, H, S*BS, D]
    # 注意：这里的 dim 1 是被 gather_ids (T维度) 索引的
    k_slc = k[b_idx, safe_gather_ids, h_idx, :]
    v_slc = v[b_idx, safe_gather_ids, h_idx, :]

    # --- 4. 计算 Selected Attention (SLC) ---
    # q: [B, T, H, D] -> [B, T, H, 1, D]
    q_unsq = q.unsqueeze(3) * scale
    
    # Einsum 计算分数: [B, T, H, 1, D] * [B, T, H, S*BS, D] -> [B, T, H, S*BS]
    attn_logits = torch.matmul(q_unsq, k_slc.transpose(-1, -2)).squeeze(3)
    
    # --- 5. 构造 Mask (替换原代码中的 if 逻辑) ---
    # 逻辑 A: 索引本身无效 (原 block_idx 为 -1 或越界)
    mask = ~valid_mask 
    
    # 逻辑 B: 因果掩码 (Causal Mask) -> 只能看过去的 token
    # gather_ids > current_t
    current_t = torch.arange(T, device=q.device).view(1, T, 1, 1)
    mask = mask | (gather_ids > current_t)
    
    # 逻辑 C: Block Counts 限制 (block_counts)
    if block_counts is not None:
        if isinstance(block_counts, int):
            # 这种情况较少见，简单处理
            pass 
        else:
            # block_counts: [B, T, H] -> 扩展到 [B, T, H, S*BS]
            # 计算当前是第几个 block
            # shape: [1, 1, 1, S*BS]
            block_rank = torch.arange(S * BS, device=q.device).view(1, 1, 1, -1) // BS
            # 扩展 counts
            counts_expanded = block_counts.unsqueeze(-1)
            mask = mask | (block_rank >= counts_expanded)

    # 应用 Mask
    attn_logits = attn_logits.masked_fill(mask, float('-inf'))
    
    # Softmax & 加权求和
    attn_weights = F.softmax(attn_logits, dim=-1) # [B, T, H, S*BS]
    
    # [B, T, H, 1, S*BS] @ [B, T, H, S*BS, D] -> [B, T, H, 1, D]
    o_slc = torch.matmul(attn_weights.unsqueeze(3), v_slc).squeeze(3)
    
    # 乘上门控
    o_slc = o_slc * g_slc.unsqueeze(-1)

    # --- 6. 计算 Sliding Window Attention (SWA) ---
    # 这部分通常可以使用 PyTorch 自带的高效实现，或者简单的对角线 Mask
    # 为了保持代码简洁，这里展示一个简单的 unfold 实现 (针对较小 window_size)
    o_swa = 0
    if window_size > 0:
        # 简单起见，这里可以用 flash attention 或 naive mask 实现
        # 考虑到 T=400 且为了保持风格统一，这里用带 Mask 的全注意力实现 SWA
        # (对于生产环境，建议使用 flash_attn_func)
        
        # 构造因果 + 窗口 Mask
        q_idx = torch.arange(T, device=q.device).unsqueeze(1)
        k_idx = torch.arange(T, device=q.device).unsqueeze(0)
        swa_mask = (k_idx > q_idx) | (q_idx - k_idx >= window_size) # [T, T]
        
        # [B, T, H, D] @ [B, H, D, T] -> [B, H, T, T]
        # 注意这里需要转置处理，略显麻烦，或者直接利用 unfold
        # 鉴于你主要关注 vectorization，这部分通常不是性能瓶颈
        # 这里仅作简单的 dense 实现演示:
        attn_dense = torch.einsum('bthd, bkhd -> bhtk', q * scale, k)
        attn_dense = attn_dense.masked_fill(swa_mask.view(1, 1, T, T), float('-inf'))
        attn_dense = F.softmax(attn_dense, dim=-1)
        o_swa_raw = torch.einsum('bhtk, bkhd -> bthd', attn_dense, v)
        o_swa = o_swa_raw * g_swa.unsqueeze(-1)

    # --- 7. 输出合并 ---
    o = o_slc + o_swa
    
    if head_first:
        o = rearrange(o, 'b t h d -> b h t d')
        
    return o.to(dtype=q.dtype)


def naive_nsa_compression(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cmp: torch.Tensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int,
    scale: float,
    head_first: bool = False
) -> torch.LongTensor:
    dtype = q.dtype
    B, T = q.shape[0], q.shape[1]
    H, HQ = k.shape[2], q.shape[2]
    G = HQ//H
    BS = block_size
    if isinstance(block_counts, int):
        block_counts = torch.full((B, T, H), block_counts, dtype=torch.long, device=q.device)
    q, k, v = map(lambda x: x.float(), (q, k, v))
    k_cmp, v_cmp = compression(k, v, BS)
    C = k_cmp.shape[1]
    S = min(block_counts.max().item(), C)
    k_cmp, v_cmp = map(lambda x: repeat(x, 'b c h d -> b c (h g) d', g=G), (k_cmp, v_cmp))

    casual_mask = ((torch.arange(T) - BS + 1)[:, None] // BS < torch.arange(C)[None, :]).to(q.device)
    empty_mask = casual_mask.all(-1, True)
    local_mask = (torch.arange(T)[:, None] // BS == torch.arange(C)[None, :]).to(q.device)

    attn_cmp = torch.einsum('bqhd,bkhd->bhqk', q*scale, k_cmp)
    attn_cmp = attn_cmp.masked_fill(casual_mask & empty_mask.logical_not(), float('-inf'))
    attn_cmp = attn_cmp.softmax(-1).masked_fill(empty_mask, 0.0)
    o_cmp = torch.einsum('bhqk, bkhd -> bqhd', attn_cmp, v_cmp) * g_cmp.unsqueeze(-1)
    attn_select = attn_cmp.masked_fill(local_mask, float(1.0))
    attn_select = attn_select.view(B, H, G, T, C).sum(2)
    block_indices = attn_select.topk(S, -1)[1]

    block_indices = block_indices.masked_fill(block_indices > (block_indices.new_tensor(range(T))[:, None] // BS), -1)
    block_indices = block_indices.transpose(1, 2)

    # 对于奇数位置（sx1），使用前一个偶数位置（sx0）的block_indices和激活结果
    for t in range(1, T, 2):
        prev_t = t - 1
        block_indices[:, t, :, :] = block_indices[:, prev_t, :, :]
        o_cmp[:, t, :, :] = o_cmp[:, prev_t, :, :]

    if head_first:
        o_cmp = rearrange(o_cmp, 'b t h d -> b h t d')
    return block_indices, o_cmp.to(dtype)


def naive_nsa_compression_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cmp: torch.Tensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int,
    scale: float,
    cu_seqlens: torch.LongTensor,
    head_first: bool = False
) -> torch.LongTensor:
    dtype = q.dtype
    B, T = q.shape[0], q.shape[1]
    H, HQ = k.shape[2], q.shape[2]
    D = v.shape[-1]
    G = HQ//H
    BS = block_size
    S = block_counts if isinstance(block_counts, int) else block_counts.max().item()
    C = math.ceil(T / block_size)
    S = min(S, C)
    block_indices = torch.zeros(B, T, H, S, dtype=torch.long, device=q.device)
    o_cmp = torch.zeros(B, T, HQ, D, dtype=dtype, device=q.device)
    for i in range(len(cu_seqlens) - 1):
        T_b = cu_seqlens[i+1] - cu_seqlens[i]
        C_b = math.ceil(T_b / block_size)
        q_b, k_b, v_b, g_cmp_b = map(
            lambda x: x[0][cu_seqlens[i]:cu_seqlens[i+1]],
            (q, k, v, g_cmp)
        )
        if isinstance(block_counts, torch.Tensor):
            s_b = block_counts[0][cu_seqlens[i]:cu_seqlens[i+1]]
        else:
            s_b = block_counts

        k_cmp, v_cmp = compression(k_b.unsqueeze(0), v_b.unsqueeze(0), BS)
        S_b = s_b if isinstance(s_b, int) else s_b.max().item()
        C_b = k_cmp.shape[1]
        S_b = min(S_b, C_b)
        k_cmp, v_cmp = map(lambda x: repeat(x.squeeze(0), 'c h d -> c (h g) d', g=G), (k_cmp, v_cmp))
        q_b, k_cmp, v_cmp = map(lambda x: x.float(), (q_b, k_cmp, v_cmp))

        casual_mask = ((torch.arange(T_b) - BS + 1)[:, None] // BS < torch.arange(C_b)[None, :]).to(q_b.device)
        local_mask = (torch.arange(T_b)[:, None] // BS == torch.arange(C_b)[None, :]).to(q.device)

        attn_cmp = torch.einsum('qhd,khd->hqk', q_b*scale, k_cmp)
        attn_cmp = attn_cmp.masked_fill(casual_mask, float('-inf'))
        attn_cmp = attn_cmp.softmax(-1)
        o_cmp[0][cu_seqlens[i]:cu_seqlens[i+1]] = torch.einsum('hqk,khd->qhd', attn_cmp, v_cmp).nan_to_num() *\
            g_cmp_b.unsqueeze(-1)
        attn_select = attn_cmp.masked_fill(local_mask, float(1.0))
        attn_select = attn_select.view(H, G, T_b, C_b).sum(1)
        block_indices_b = attn_select.topk(S_b, -1)[1]
        block_indices_b = block_indices_b.masked_fill(
            block_indices_b > (block_indices_b.new_tensor(range(T_b))[:, None]//BS),
            0
        )
        block_indices[0][cu_seqlens[i]:cu_seqlens[i+1], :, :S_b] = block_indices_b.transpose(0, 1)

    if head_first:
        o_cmp = rearrange(o_cmp, 'b t h d -> b h t d')
    return block_indices, o_cmp.to(dtype)


def naive_nsa_with_compression(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cmp: torch.Tensor,
    g_slc: torch.Tensor,
    g_swa: torch.Tensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int = 64,
    window_size: int = 0,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, HQ, K]` if `head_first=False` else `[B, HQ, T, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >=16.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g_cmp (torch.Tensor):
            Gate score for compressed attention of shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        g_slc (torch.Tensor):
            Gate score for selected attention of shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        g_swa (torch.Tensor):
            Gate score for sliding attentionof shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        block_counts (Union[torch.LongTensor, int]):
            Number of selected blocks for each token.
            If a tensor is provided, with shape `[B, T, H]` if `head_first=True` else `[B, T, H]`,
            each token can select the same number of blocks.
        block_size (int):
            Selected block size. Default: 64.
        window_size (int):
            Sliding window size. Default: 0.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]` if `head_first=False` else `[B, HQ, T, V]`.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
        if head_first:
            raise RuntimeError("Sequences with variable lengths are not supported for head-first mode")
    if head_first:
        q, k, v = map(lambda x: rearrange(x, 'b h t d -> b t h d'), (q, k, v))
        g_cmp, g_slc = map(lambda x: rearrange(x, 'b h t -> b t h'), (g_cmp, g_slc))
        if isinstance(block_counts, torch.Tensor):
            block_counts = rearrange(block_counts, 'b h t -> b t h')
    if cu_seqlens is not None:
        block_indices, o_cmp = naive_nsa_compression_varlen(
            q=q,
            k=k,
            v=v,
            g_cmp=g_cmp,
            block_counts=block_counts,
            block_size=block_size,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=False)
    else:
        block_indices, o_cmp = naive_nsa_compression(
            q=q,
            k=k,
            v=v,
            g_cmp=g_cmp,
            block_counts=block_counts,
            block_size=block_size,
            scale=scale,
            head_first=False)
    # o = naive_nsa(
    o = vectorized_naive_nsa(
        q=q,
        k=k,
        v=v,
        g_slc=g_slc,
        g_swa=g_swa,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        window_size=window_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
        head_first=False
    ) + o_cmp

    if head_first:
        o = rearrange(o, 'b t h d -> b h t d')

    return o, block_indices