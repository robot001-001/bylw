import torch
import torch.nn.functional as F
import math

# 假设你已经定义了上面的类和Kernel
from hstu_bsa_triton import HSTU_BSA_Triton 

def jagged_to_padded(jagged, offsets, max_len):
    """辅助函数：Jagged转Padded"""
    B = len(offsets) - 1
    D = jagged.shape[-1]
    padded = torch.zeros((B, max_len, D), dtype=jagged.dtype, device=jagged.device)
    for i in range(B):
        start, end = offsets[i], offsets[i+1]
        length = end - start
        padded[i, :length] = jagged[start:end]
    return padded

def padded_to_jagged(padded, offsets):
    """辅助函数：Padded转Jagged (用于对比结果)"""
    jagged_list = []
    for i in range(len(offsets) - 1):
        length = offsets[i+1] - offsets[i]
        jagged_list.append(padded[i, :length])
    return torch.cat(jagged_list, dim=0)

# -----------------------------------------------------------------------------
# 1. Golden Reference Implementation (Pure PyTorch)
# -----------------------------------------------------------------------------
def hstu_bsa_reference(q, k, v, g_cmp, g_slc, offsets, block_size, topk_n):
    """
    纯 PyTorch 实现的参照版本（使用 Padding 逻辑），用于验证 Triton Kernel 的正确性。
    """
    B = len(offsets) - 1
    max_len = (offsets[1:] - offsets[:-1]).max().item()
    D = q.shape[-1]
    scale = D ** -0.5

    # 1. 转为 Padded
    q_pad = jagged_to_padded(q, offsets, max_len) # [B, N, D]
    k_pad = jagged_to_padded(k, offsets, max_len)
    v_pad = jagged_to_padded(v, offsets, max_len)
    g_cmp_pad = jagged_to_padded(g_cmp, offsets, max_len) # [B, N, 1]
    g_slc_pad = jagged_to_padded(g_slc, offsets, max_len)

    # Padding Mask
    seq_idx = torch.arange(max_len, device=q.device)[None, :]
    lens = (offsets[1:] - offsets[:-1]).unsqueeze(1)
    padding_mask = seq_idx < lens # [B, N] True is valid

    # 2. Compression (Pooling)
    # 计算需要补齐到 block_size 的长度
    num_blocks = math.ceil(max_len / block_size)
    pad_len = num_blocks * block_size - max_len
    
    k_padded_extend = F.pad(k_pad, (0, 0, 0, pad_len))
    v_padded_extend = F.pad(v_pad, (0, 0, 0, pad_len))
    
    # [B, NumBlocks, BS, D] -> Mean -> [B, NumBlocks, D]
    k_cmp = k_padded_extend.view(B, num_blocks, block_size, D).mean(dim=2)
    v_cmp = v_padded_extend.view(B, num_blocks, block_size, D).mean(dim=2)

    # 3. Coarse Attention (用于计算 Output_Cmp 和 TopK)
    # Score: Q @ K_cmp.T
    # [B, N, D] @ [B, NumBlocks, D].T -> [B, N, NumBlocks]
    attn_score_cmp = torch.matmul(q_pad, k_cmp.transpose(1, 2)) * scale
    
    # Causal Mask (Block Level)
    # Q_idx // BS >= K_idx
    q_blk_idx = torch.arange(max_len, device=q.device)[:, None] // block_size
    k_blk_idx = torch.arange(num_blocks, device=q.device)[None, :]
    causal_mask_blk = q_blk_idx >= k_blk_idx # [N, NumBlocks]
    
    attn_score_cmp_masked = attn_score_cmp.masked_fill(~causal_mask_blk.unsqueeze(0), -1e9)
    
    # 3.1 Output Cmp
    # Activation: SiLU
    probs_cmp = attn_score_cmp_masked * torch.sigmoid(attn_score_cmp_masked)
    # Mask Causal for value accumulation (Zero out invalid)
    probs_cmp = torch.where(causal_mask_blk.unsqueeze(0), probs_cmp, torch.tensor(0.0, device=q.device))
    
    # O_cmp = Probs @ V_cmp
    o_cmp = torch.matmul(probs_cmp, v_cmp)
    # Apply Gating and Padding Mask
    o_cmp = o_cmp * g_cmp_pad * padding_mask.unsqueeze(-1)

    # 4. Selected Attention (SLC)
    # TopK Selection
    # Mask out future blocks strongly for TopK selection
    attn_score_for_topk = attn_score_cmp.masked_fill(~causal_mask_blk.unsqueeze(0), float('-inf'))
    
    S = min(topk_n, num_blocks)
    _, topk_indices = attn_score_for_topk.topk(S, dim=-1) # [B, N, S]
    
    # 构建 Selected Mask [B, N, NumBlocks]
    # 如果 block_idx 在 topk_indices 中，则为 True
    selected_mask_blk = torch.zeros_like(attn_score_cmp, dtype=torch.bool)
    selected_mask_blk.scatter_(2, topk_indices, True)
    
    # 将 Block Mask 扩展回 Token Level [B, N, N_padded]
    # [B, N, NumBlocks] -> repeat -> [B, N, NumBlocks * BS]
    selected_mask_token = selected_mask_blk.repeat_interleave(block_size, dim=2)
    # Crop 掉多余的 padding
    selected_mask_token = selected_mask_token[:, :, :max_len]

    # Full Attention Score: Q @ K.T
    attn_score_full = torch.matmul(q_pad, k_pad.transpose(1, 2)) * scale
    
    # Causal Mask (Token Level)
    q_idx = torch.arange(max_len, device=q.device)[:, None]
    k_idx = torch.arange(max_len, device=q.device)[None, :]
    causal_mask_token = q_idx >= k_idx
    
    # Apply Causal & Selected Mask
    # 我们只保留: 1. Causal valid 2. Padding valid 3. Selected valid
    final_mask = causal_mask_token.unsqueeze(0) & padding_mask.unsqueeze(1) & selected_mask_token
    
    # SiLU Activation
    probs_slc = attn_score_full * torch.sigmoid(attn_score_full)
    probs_slc = torch.where(final_mask, probs_slc, torch.tensor(0.0, device=q.device))
    
    # O_slc = Probs @ V
    o_slc = torch.matmul(probs_slc, v_pad)
    # Apply Gating
    o_slc = o_slc * g_slc_pad * padding_mask.unsqueeze(-1)

    # Return Jagged results for comparison
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
    
    # 随机生成不等长的序列长度
    min_len = 40
    max_len = 128
    lengths = torch.randint(min_len, max_len, (B,), device=device)
    offsets = torch.cat([torch.tensor([0], device=device), torch.cumsum(lengths, 0)])
    total_tokens = offsets[-1].item()
    
    print(f"Testing with Batch={B}, Heads={H}, Dim={D}, BlockSize={block_size}")
    print(f"Sequence Lengths: {lengths.tolist()}")
    print(f"Total Tokens: {total_tokens}")

    # Initialize Inputs
    q = torch.randn((total_tokens, H, D), device=device, dtype=torch.float32)
    k = torch.randn((total_tokens, H, D), device=device, dtype=torch.float32)
    v = torch.randn((total_tokens, H, D), device=device, dtype=torch.float32)
    
    # Gates (假设已经经过 sigmoid 或者网络输出，这里随机生成)
    g_cmp = torch.rand((total_tokens, H, 1), device=device, dtype=torch.float32)
    g_slc = torch.rand((total_tokens, H, 1), device=device, dtype=torch.float32)

    # --- Run Triton Model ---
    print("\nRunning Triton Kernel...")
    model = HSTU_BSA_Triton(block_size=block_size, block_counts=topk_n).to(device)
    
    # 预热
    # model(q, k, v, g_cmp, g_slc, offsets) 
    
    triton_o_cmp, triton_o_slc = model(q, k, v, g_cmp, g_slc, offsets)

    # --- Run Reference Model ---
    print("Running PyTorch Reference...")
    ref_o_cmp, ref_o_slc = hstu_bsa_reference(
        q, k, v, g_cmp, g_slc, offsets, block_size, topk_n
    )

    # --- Verification ---
    # 1. Check Output CMP (Compressed Attention)
    # 允许一定的误差 (Triton kernel 内部累加顺序和 float32 精度差异)
    tol = 1e-4
    diff_cmp = (triton_o_cmp - ref_o_cmp).abs().max()
    is_close_cmp = torch.allclose(triton_o_cmp, ref_o_cmp, atol=tol, rtol=1e-3)
    
    print(f"\n[Comparison] Output CMP:")
    print(f"  Max Diff: {diff_cmp.item():.6f}")
    print(f"  Pass: {is_close_cmp}")

    if not is_close_cmp:
        print("  -> Debug: Sample Diff")
        print("  Triton:", triton_o_cmp[0,0,:5])
        print("  Ref:   ", ref_o_cmp[0,0,:5])

    # 2. Check Output SLC (Selected Attention)
    diff_slc = (triton_o_slc - ref_o_slc).abs().max()
    is_close_slc = torch.allclose(triton_o_slc, ref_o_slc, atol=tol, rtol=1e-3)
    
    print(f"\n[Comparison] Output SLC:")
    print(f"  Max Diff: {diff_slc.item():.6f}")
    print(f"  Pass: {is_close_slc}")
    
    if not is_close_slc:
        print("  -> Debug: Sample Diff")
        print("  Triton:", triton_o_slc[0,0,:5])
        print("  Ref:   ", ref_o_slc[0,0,:5])

    # Final Result
    if is_close_cmp and is_close_slc:
        print("\n✅ TEST PASSED: Triton implementation matches Golden Reference.")
    else:
        print("\n❌ TEST FAILED: Mismatch detected.")

if __name__ == "__main__":
    test_correctness()