# run local
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(
    precision=4,        # 显示的小数位数
    threshold=float('inf'),     # 触发折叠（显示省略号）的元素总数阈值
    edgeitems=3,        # 折叠时，开头和结尾显示的元素个数
    linewidth=80,       # 每行的字符宽度
    profile=None,       # 使用预设配置 ('default', 'short', 'full')
    sci_mode=True       # 是否使用科学计数法 (True/False)
)

from typing import Union, Optional
import math
import os
import sys
import logging



logging_dir = os.path.join(os.path.dirname(__file__), '../../../log')

def init_log():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []
    formatter = logging.Formatter(
        '%(message)s'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    if logging_dir:
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir, exist_ok=True)
        log_file_path = os.path.join(logging_dir, "train_nsa.log")
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logging.info(f"Log directory setup success: {log_file_path}")
    else:
        logging.warning("FLAGS.logging_dir is None. Logging to console only.")

# @torch.compile
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


def naive_nsa_compression(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cmp: torch.Tensor,
    block_counts: int,
    block_size: int,
    scale: float,
):
    logging.info(f'*************inside naive_nsa_compression*************')
    dtype = q.dtype
    B, T, H, D = q.shape
    BS = block_size

    block_counts = torch.full((B, T, H), block_counts, dtype=torch.long, device=q.device) # 2, 128, 4
    q, k, v = map(lambda x: x.float(), (q, k, v))
    k_cmp, v_cmp = compression(k, v, BS)
    logging.info(f'k_cmp, v_cmp: {k_cmp.shape, v_cmp.shape}')
    C = k_cmp.shape[1] # total_blocks: 8
    S = min(block_counts.max().item(), C)

    casual_mask = ((torch.arange(T) - BS + 1)[:, None] // BS < torch.arange(C)[None, :]).to(q.device) # [128, 8]↗三角
    logging.info(f'casual_mask: {casual_mask.shape}')
    empty_mask = casual_mask.all(-1, True) # [128, 1] 第一块是true，其余false
    logging.info(f'empty_mask: {empty_mask.shape}')
    local_mask = (torch.arange(T)[:, None] // BS == torch.arange(C)[None, :]).to(q.device)
    logging.info(f'local_mask: {local_mask.shape}')
    attn_cmp = torch.einsum('bqhd,bkhd->bhqk', q*scale, k_cmp)
    attn_cmp = attn_cmp.masked_fill(casual_mask & empty_mask.logical_not(), float('-inf')) # 防报错，无意义
    attn_cmp = F.softmax(attn_cmp, dim=-1)
    o_cmp = torch.einsum('bhqk, bkhd -> bqhd', attn_cmp*(empty_mask.logical_not()), v_cmp) * g_cmp.unsqueeze(-1)
    # o_cmp = torch.einsum('bhqk, bkhd -> bqhd', attn_cmp, v_cmp) * g_cmp.unsqueeze(-1)
    logging.info(f'o_cmp: {o_cmp.shape}')
    attn_select = attn_cmp.masked_fill(local_mask, float(1.0))
    logging.info(f'attn_select: {attn_select.shape}')
    block_indices = attn_select.topk(S, -1)[1]

    block_indices = block_indices.masked_fill(block_indices > (block_indices.new_tensor(range(T))[:, None] // BS), -1)
    block_indices = block_indices.transpose(1, 2)
    logging.info(f'block_indices: {block_indices.shape}')

    logging.info(f'*************end of naive_nsa_compression*************')
    return block_indices, o_cmp.to(dtype)


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
):
    logging.info(f'-------------inside naive_nsa-------------')
    B, T, H, D = q.shape
    S = block_indices.shape[-1]
    BS = block_size

    q, k, v = map(lambda x: x.float(), (q, k, v))
    offsets = torch.arange(BS, device=q.device).view(1, 1, 1, 1, BS)
    logging.info(f'offsets: {offsets}')
    start_indices = block_indices.unsqueeze(-1) * BS
    # logging.info(f'block_indices: {block_indices}')
    logging.info(f'start_indices: {start_indices.shape}')
    gather_ids = start_indices + offsets
    # logging.info(f'gather_ids: {gather_ids.shape, gather_ids}')
    gather_ids = gather_ids.view(B, T, H, S * BS)
    logging.info(f'gather_ids: {gather_ids.shape}')
    valid_mask = (gather_ids >= 0) & (gather_ids < T)
    logging.info(f'valid_mask: {valid_mask.shape}')
    safe_gather_ids = gather_ids.clamp(0, T - 1)
    logging.info(f'safe_gather_ids: {safe_gather_ids.shape}')
    b_idx = torch.arange(B, device=q.device).view(B, 1, 1, 1)
    h_idx = torch.arange(H, device=q.device).view(1, 1, H, 1)
    k_slc = k[b_idx, safe_gather_ids, h_idx, :]
    v_slc = v[b_idx, safe_gather_ids, h_idx, :]
    logging.info(f'k.shape, k_slc.shape: {k.shape, k_slc.shape}')
    logging.info(f'v.shape, v_slc.shape: {v.shape, v_slc.shape}')
    q_unsq = q.unsqueeze(3) * scale
    logging.info(f'q.shape, q_unsq.shape: {q.shape, q_unsq.shape}')
    attn_logits = torch.matmul(q_unsq, k_slc.transpose(-1, -2)).squeeze(3)
    mask = ~valid_mask 
    current_t = torch.arange(T, device=q.device).view(1, T, 1, 1)
    mask = mask | (gather_ids > current_t)
    attn_logits = attn_logits.masked_fill(mask, float('-inf'))
    attn_weights = F.softmax(attn_logits, dim=-1)
    o_slc = torch.matmul(attn_weights.unsqueeze(3), v_slc).squeeze(3)
    o_slc = o_slc * g_slc.unsqueeze(-1)
    o_swa = 0
    if window_size > 0:
        q_idx = torch.arange(T, device=q.device).unsqueeze(1)
        k_idx = torch.arange(T, device=q.device).unsqueeze(0)
        swa_mask = (k_idx > q_idx) | (q_idx - k_idx >= window_size)
        attn_dense = torch.einsum('bthd, bkhd -> bhtk', q * scale, k)
        attn_dense = attn_dense.masked_fill(swa_mask.view(1, 1, T, T), float('-inf'))
        attn_dense = F.softmax(attn_dense, dim=-1)
        o_swa_raw = torch.einsum('bhtk, bkhd -> bthd', attn_dense, v)
        o_swa = o_swa_raw * g_swa.unsqueeze(-1)
    o = o_slc + o_swa
    logging.info(f'-------------end of naive_nsa-------------')
    return o.to(dtype=q.dtype)


def naive_nsa_with_compression(
    q: torch.Tensor, # [B, T, HQ, K]
    k: torch.Tensor, # [B, T, H, K]
    v: torch.Tensor,
    g_cmp: torch.Tensor,
    g_slc: torch.Tensor,
    g_swa: torch.Tensor,
    block_counts: int,
    block_size: int = 64,
    window_size: int = 0,
    scale: float = None
):
    logging.info(f'=============inside naive_nsa_with_compression=============')
    if scale is None:
        scale = k.shape[-1] ** -0.5
    logging.info(f'scale: {scale}')
    
    block_indices, o_cmp = naive_nsa_compression(
        q=q,
        k=k,
        v=v,
        g_cmp=g_cmp,
        block_counts=block_counts,
        block_size=block_size,
        scale=scale,)
    
    o = naive_nsa(
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
    ) + o_cmp

    logging.info(f'=============end of naive_nsa_with_compression=============')
    return o, block_indices



if __name__ == "__main__":
    init_log()
    Bsize, seq_len, num_heads, head_dim = 2, 128, 4, 8
    q = torch.randn(Bsize, seq_len, num_heads, head_dim)
    k = torch.randn(Bsize, seq_len, num_heads, head_dim)
    v = torch.randn(Bsize, seq_len, num_heads, head_dim)
    g_cmp = torch.tensor(0.1, dtype=torch.float32)
    g_slc = torch.tensor(0.2, dtype=torch.float32)
    g_swa = torch.tensor(0.3, dtype=torch.float32)
    block_counts = 4
    block_size = 16
    window_size = 0
    scale = None
    o, block_indices = naive_nsa_with_compression(
        q=q, k=k, v=v,
        g_cmp=g_cmp, g_slc=g_slc, g_swa=g_swa,
        block_counts=block_counts, block_size=block_size,
        window_size=window_size,
        scale=scale
    )
    logging.info(f'block_indices.shape: {block_indices.shape}')
    logging.info(f'block_indices[:, -1, :, :]: {block_indices[:, -1, :, :]}')