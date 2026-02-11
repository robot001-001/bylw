import torch
import torch.nn.functional as F
import math
import triton

# 假设你已经定义了 HSTU_BSA_Triton 类和 Kernel
from hstu_bsa_triton_v2 import HSTU_BSA_Triton, hstu_bsa_cmp_fwd_kernel, hstu_bsa_slc_fwd_kernel
# 如果都在一个文件里，就不用 import

# -----------------------------------------------------------------------------
# 0. Helper Functions (修正版：支持任意维度)
# -----------------------------------------------------------------------------
def jagged_to_padded(jagged, offsets, max_len):
    """
    Jagged [TotalTokens, ...] -> Padded [B, MaxLen, ...]
    自动适配 Head 维度
    """
    B = len(offsets) - 1
    # 获取除 token 维以外的其余维度形状
    other_shape = list(jagged.shape[1:]) 
    # 构造目标形状 [B, MaxLen, H, D] 或 [B, MaxLen, H, 1]
    target_shape = [B, max_len] + other_shape
    
    padded = torch.zeros(target_shape, dtype=jagged.dtype, device=jagged.device)
    for i in range(B):
        start, end = offsets[i], offsets[i+1]
        length = end - start
        padded[i, :length] = jagged[start:end]
    return padded

def padded_to_jagged(padded, offsets):
    """
    Padded [B, MaxLen, ...] -> Jagged [TotalTokens, ...]
    """
    jagged_list = []
    for i in range(len(offsets) - 1):
        length = offsets[i+1] - offsets[i]
        jagged_list.append(padded[i, :length])
    return torch.cat(jagged_list, dim=0)

# -----------------------------------------------------------------------------
# 1. Golden Reference Implementation (修正版：正确处理 Head 维度)
# -----------------------------------------------------------------------------
def hstu_bsa_reference(q, k, v, g_cmp, g_slc, offsets, block_size, topk_n):
    """
    纯 PyTorch 实现的参照版本，支持 Multi-Head
    """
    B = len(offsets) - 1
    max_len = (offsets[1:] - offsets[:-1]).max().item()
    D = q.shape[-1]
    H = q.shape[1]
    scale = D ** -0.5

    # 1. 转为 Padded: [B, N, H, D]
    q_pad = jagged_to_padded(q, offsets, max_len) 
    k_pad = jagged_to_padded(k, offsets, max_len)
    v_pad = jagged_to_padded(v, offsets, max_len)
    g_cmp_pad = jagged_to_padded(g_cmp, offsets, max_len) # [B, N, H, 1]
    g_slc_pad = jagged_to_padded(g_slc, offsets, max_len)

    # Padding Mask: [B, N]
    seq_idx = torch.arange(max_len, device=q.device)[None, :]
    lens = (offsets[1:] - offsets[:-1]).unsqueeze(1)
    padding_mask = seq_idx < lens 

    # 2. Compression (Pooling)
    num_blocks = math.ceil(max_len / block_size)
    pad_len = num_blocks * block_size - max_len
    
    # Pad K/V on dim=1 (Time)
    # F.pad format: (last_dim_left, last_dim_right, ..., dim1_left, dim1_right)
    # K shape [B, N, H, D], we pad N dimension.
    # pad order: D_l, D_r, H_l, H_r, N_l, N_r
    k_padded_extend = F.pad(k_pad.permute(0, 2, 3, 1), (0, pad_len)).permute(0, 3, 1, 2) # [B, N_pad, H, D]
    v_padded_extend = F.pad(v_pad.permute(0, 2, 3, 1), (0, pad_len)).permute(0, 3, 1, 2)
    
    # Reshape & Mean: [B, NumBlocks, BS, H, D] -> [B, NumBlocks, H, D]
    k_cmp = k_padded_extend.view(B, num_blocks, block_size, H, D).mean(dim=2)
    v_cmp = v_padded_extend.view(B, num_blocks, block_size, H, D).mean(dim=2)

    # 3. Coarse Attention (CMP)
    # Q: [B, N, H, D], K_cmp: [B, K, H, D] -> Score: [B, H, N, K]
    attn_score_cmp = torch.einsum('bqhd,bkhd->bhqk', q_pad, k_cmp) * scale
    
    # Causal Mask (Block Level)
    # Q_idx // BS >= K_idx
    q_blk_idx = torch.arange(max_len, device=q.device)[:, None] // block_size
    k_blk_idx = torch.arange(num_blocks, device=q.device)[None, :]
    causal_mask_blk = q_blk_idx >= k_blk_idx # [N, K]
    
    # Apply Mask: [1, 1, N, K]
    attn_score_cmp_masked = attn_score_cmp.masked_fill(~causal_mask_blk.unsqueeze(0).unsqueeze(0), -1e9)
    
    # Activation
    probs_cmp = attn_score_cmp_masked * torch.sigmoid(attn_score_cmp_masked) # SiLU
    probs_cmp = torch.where(causal_mask_blk.unsqueeze(0).unsqueeze(0), probs_cmp, torch.tensor(0.0, device=q.device))
    
    # O_cmp = Probs [B, H, N, K] @ V_cmp [B, K, H, D]
    # Einsum: bhqk, bkhd -> bqhd
    o_cmp = torch.einsum('bhqk,bkhd->bqhd', probs_cmp, v_cmp)
    
    # Apply Gating and Padding Mask
    o_cmp = o_cmp * g_cmp_pad * padding_mask.unsqueeze(-1).unsqueeze(-1)

    # 4. Selected Attention (SLC)
    # TopK Selection from attn_score_cmp
    # Mask future blocks heavily
    attn_score_for_topk = attn_score_cmp.masked_fill(~causal_mask_blk.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    S = min(topk_n, num_blocks)
    _, topk_indices = attn_score_for_topk.topk(S, dim=-1) # [B, H, N, S]
    
    # 构建 Selected Mask [B, H, N, NumBlocks]
    selected_mask_blk = torch.zeros_like(attn_score_cmp, dtype=torch.bool)
    selected_mask_blk.scatter_(3, topk_indices, True)
    
    # Expand to Token Level [B, H, N, N_padded]
    selected_mask_token = selected_mask_blk.repeat_interleave(block_size, dim=3)
    selected_mask_token = selected_mask_token[:, :, :, :max_len]

    # Full Attention Score: Q @ K.T -> [B, H, N, N]
    attn_score_full = torch.einsum('bqhd,bkhd->bhqk', q_pad, k_pad) * scale
    
    # Causal Mask (Token Level)
    q_idx = torch.arange(max_len, device=q.device)[:, None]
    k_idx = torch.arange(max_len, device=q.device)[None, :]
    causal_mask_token = q_idx >= k_idx # [N, N]
    
    # Final Mask = Causal & Padding & Selected
    # Causal: [1, 1, N, N]
    # Padding: [B, 1, 1, N] (Key padding) -> 其实上面 k_pad 已经是 0 了，但为了 mask 准确性
    # Selected: [B, H, N, N]
    
    final_mask = causal_mask_token.unsqueeze(0).unsqueeze(0) & \
                 padding_mask.unsqueeze(1).unsqueeze(1) & \
                 selected_mask_token
    
    probs_slc = attn_score_full * torch.sigmoid(attn_score_full)
    probs_slc = torch.where(final_mask, probs_slc, torch.tensor(0.0, device=q.device))
    
    # O_slc = Probs [B, H, N, N] @ V [B, N, H, D] -> [B, N, H, D]
    o_slc = torch.einsum('bhqk,bkhd->bqhd', probs_slc, v_pad)
    o_slc = o_slc * g_slc_pad * padding_mask.unsqueeze(-1).unsqueeze(-1)

    return padded_to_jagged(o_cmp, offsets), padded_to_jagged(o_slc, offsets)


# -----------------------------------------------------------------------------
# 2. Test Driver
# -----------------------------------------------------------------------------
def test_correctness():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparams
    B = 4
    H = 4
    D = 64
    block_size = 32
    topk_n = 2
    
    min_len = 40
    max_len = 100
    lengths = torch.randint(min_len, max_len, (B,), device=device)
    offsets = torch.cat([torch.tensor([0], device=device), torch.cumsum(lengths, 0)])
    total_tokens = offsets[-1].item()
    
    print(f"Testing with Batch={B}, Heads={H}, Dim={D}")
    print(f"Lengths: {lengths.tolist()}")
    
    # Inputs
    q = torch.randn((total_tokens, H, D), device=device)
    k = torch.randn((total_tokens, H, D), device=device)
    v = torch.randn((total_tokens, H, D), device=device)
    g_cmp = torch.rand((total_tokens, H, 1), device=device)
    g_slc = torch.rand((total_tokens, H, 1), device=device)

    # Triton Run
    # 确保 HSTU_BSA_Triton 类使用上一条回答中的 "Pure PyTorch" 修复版
    model = HSTU_BSA_Triton(block_size=block_size, block_counts=topk_n).to(device)
    triton_o_cmp, triton_o_slc = model(q, k, v, g_cmp, g_slc, offsets)

    # Reference Run
    ref_o_cmp, ref_o_slc = hstu_bsa_reference(
        q, k, v, g_cmp, g_slc, offsets, block_size, topk_n
    )

    # Compare
    # 适当放宽误差限，因为 float32 累加顺序不同
    tol = 5e-4 
    diff_cmp = (triton_o_cmp - ref_o_cmp).abs().max()
    diff_slc = (triton_o_slc - ref_o_slc).abs().max()
    
    print(f"\nDiff CMP: {diff_cmp:.6f}")
    print(f"Diff SLC: {diff_slc:.6f}")
    
    if diff_cmp < tol and diff_slc < tol:
        print("✅ TEST PASSED")
    else:
        print("❌ TEST FAILED")
        # Print debug info if failed
        print("Triton Cmp Sample:", triton_o_cmp[0,0,:3])
        print("Ref Cmp Sample:   ", ref_o_cmp[0,0,:3])

if __name__ == "__main__":
    # 需要先定义 HSTU_BSA_Triton 类 (参考上一条回答的修复版)
    test_correctness()