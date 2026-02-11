import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl

# ==========================================
# 1. FBGEMM 替代算子
# ==========================================
class FBGEMM_Ops:
    @staticmethod
    def jagged_to_padded_dense(values, offsets, max_lengths, padding_value=0.0):
        offset_tensor = offsets[0]
        max_len = max_lengths[0]
        B = offset_tensor.size(0) - 1
        rest_shape = values.shape[1:] 
        device = values.device
        
        seq_lengths = offset_tensor[1:] - offset_tensor[:-1]
        mask = torch.arange(max_len, device=device)[None, :] < seq_lengths[:, None]
        
        out_shape = (B, max_len) + rest_shape
        out = torch.full(out_shape, padding_value, dtype=values.dtype, device=device)
        out[mask] = values
        return out

    @staticmethod
    def dense_to_jagged(padded_dense, offsets):
        offset_tensor = offsets[0]
        max_len = padded_dense.shape[1]
        seq_lengths = offset_tensor[1:] - offset_tensor[:-1]
        mask = torch.arange(max_len, device=padded_dense.device)[None, :] < seq_lengths[:, None]
        values = padded_dense[mask]
        return [values]

# ==========================================
# 2. Triton Kernels (修复了 Scale 和 Gate)
# ==========================================

@triton.jit
def _hstu_silu(x):
    return x * tl.sigmoid(x)

@triton.jit
def hstu_bsa_cmp_kernel(
    Q, K, V, G_cmp, Out, Offsets, 
    stride_q_t, stride_q_h, stride_q_d,
    stride_k_b, stride_k_blk, stride_k_h, stride_k_d,
    stride_v_b, stride_v_blk, stride_v_h, stride_v_d,
    stride_o_t, stride_o_h, stride_o_d,
    stride_g_t, stride_g_h,
    scale, inv_scale, # [Fix] Added inv_scale
    BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m, pid_h, pid_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq_start = tl.load(Offsets + pid_b)
    seq_len = tl.load(Offsets + pid_b + 1) - seq_start
    cmp_len = tl.cdiv(seq_len, BLOCK_SIZE)
    
    start_m_idx = pid_m * BLOCK_M
    if start_m_idx >= seq_len: return

    offs_m = start_m_idx + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_len
    
    q_ptrs = Q + (seq_start + offs_m[:, None]) * stride_q_t + pid_h * stride_q_h + tl.arange(0, HEAD_DIM)[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    g_ptrs = G_cmp + (seq_start + offs_m) * stride_g_t + pid_h * stride_g_h
    g = tl.load(g_ptrs, mask=mask_m, other=0.0)
    
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, cmp_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < cmp_len
        k_ptrs = K + pid_b * stride_k_b + offs_n[None, :] * stride_k_blk + pid_h * stride_k_h + tl.arange(0, HEAD_DIM)[:, None]
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        
        # [Fix] Logic: score = (Q @ K * scale)
        score = tl.dot(q, k) * scale
        
        is_causal = (offs_m[:, None] // BLOCK_SIZE) >= offs_n[None, :]
        
        # [Fix] Logic: SiLU(score) / scale
        p = _hstu_silu(score) * inv_scale 
        p = tl.where(is_causal & mask_m[:, None] & mask_n[None, :], p, 0.0)
        
        v_ptrs = V + pid_b * stride_v_b + offs_n[:, None] * stride_v_blk + pid_h * stride_v_h + tl.arange(0, HEAD_DIM)[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16))

    acc = acc * g[:, None]
    tl.store(Out + (seq_start + offs_m[:, None]) * stride_o_t + pid_h * stride_o_h + tl.arange(0, HEAD_DIM)[None, :], acc.to(Out.dtype.element_ty), mask=mask_m[:, None])

@triton.jit
def hstu_bsa_slc_kernel(
    Q, K, V, 
    # [Fix] Removed G_slc argument since it is NOT used in user's bsa_cal
    BlockIdx, Out, Offsets,
    stride_q_t, stride_q_h, stride_q_d,
    stride_idx_t, stride_idx_h, stride_idx_s,
    # stride_g_t, stride_g_h, # Removed
    scale, inv_scale, # [Fix] Added inv_scale
    S: tl.constexpr, BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr
):
    pid_m, pid_h, pid_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq_start = tl.load(Offsets + pid_b)
    seq_len = tl.load(Offsets + pid_b + 1) - seq_start
    start_m_idx = pid_m * BLOCK_M
    if start_m_idx >= seq_len: return
    
    offs_m = start_m_idx + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_len
    
    q = tl.load(Q + (seq_start + offs_m[:, None]) * stride_q_t + pid_h * stride_q_h + tl.arange(0, HEAD_DIM)[None, :], mask=mask_m[:, None], other=0.0)
    
    # [Fix] No Gate Loading
    
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for s_i in range(S):
        b_idx_ptr = BlockIdx + (seq_start + offs_m) * stride_idx_t + pid_h * stride_idx_h + s_i * stride_idx_s
        b_idx = tl.load(b_idx_ptr, mask=mask_m, other=-1)
        
        for blk_offset in range(BLOCK_SIZE):
            target_idx = seq_start + b_idx * BLOCK_SIZE + blk_offset
            is_valid = (b_idx >= 0) & (target_idx <= (seq_start + offs_m))
            
            k_val = tl.load(K + target_idx[:, None] * stride_q_t + pid_h * stride_q_h + tl.arange(0, HEAD_DIM)[None, :], mask=is_valid[:, None] & mask_m[:, None], other=0.0)
            score = tl.sum(q * k_val, axis=1) * scale
            
            # [Fix] SiLU(score) / scale
            p = _hstu_silu(score) * inv_scale
            p = tl.where(is_valid, p, 0.0)
            
            v_val = tl.load(V + target_idx[:, None] * stride_q_t + pid_h * stride_q_h + tl.arange(0, HEAD_DIM)[None, :], mask=is_valid[:, None] & mask_m[:, None], other=0.0)
            acc += p[:, None] * v_val

    # [Fix] No Gate Multiplication
    tl.store(Out + (seq_start + offs_m[:, None]) * stride_q_t + pid_h * stride_q_h + tl.arange(0, HEAD_DIM)[None, :], acc.to(Out.dtype.element_ty), mask=mask_m[:, None])

class HSTU_BSA_Triton(nn.Module):
    def __init__(self, num_heads, head_dim, block_size=32, block_counts=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.block_counts = block_counts

    def forward(self, q, k, v, u, x_offsets, gate_model):
        B = x_offsets.size(0) - 1
        seq_lens = x_offsets[1:] - x_offsets[:-1]
        max_seq_len = seq_lens.max().item()
        total_tokens = q.shape[0]
        scale = self.head_dim ** -0.5
        inv_scale = 1.0 / scale # [Fix]

        padded_q = FBGEMM_Ops.jagged_to_padded_dense(q, [x_offsets], [max_seq_len])
        padded_k = FBGEMM_Ops.jagged_to_padded_dense(k, [x_offsets], [max_seq_len])
        padded_v = FBGEMM_Ops.jagged_to_padded_dense(v, [x_offsets], [max_seq_len])
        g_cmp, g_slc, _ = gate_model(padded_q)
        
        num_blocks = math.ceil(max_seq_len / self.block_size)
        pad_len = num_blocks * self.block_size - max_seq_len
        pad_params = (0, 0, 0, 0, 0, pad_len) if pad_len > 0 else None
        
        padded_k_p = F.pad(padded_k, pad_params) if pad_params else padded_k
        padded_v_p = F.pad(padded_v, pad_params) if pad_params else padded_v
        
        k_cmp = padded_k_p.view(B, num_blocks, self.block_size, self.num_heads, self.head_dim).mean(dim=2)
        v_cmp = padded_v_p.view(B, num_blocks, self.block_size, self.num_heads, self.head_dim).mean(dim=2)

        # [Fix] TopK Selection Logic to Match Reference Exactly
        # Reference: attn_cmp = SiLU(Q*K*scale)/scale; mask 0; mask local 1.0; topk
        
        # 1. Raw Scores
        attn_scores = torch.einsum('bnhd,bmhd->bnhm', padded_q, k_cmp) * scale
        
        # 2. Causal Mask (Pre-activation)
        q_idx = torch.arange(max_seq_len, device=q.device)[:, None] // self.block_size
        k_idx = torch.arange(num_blocks, device=q.device)[None, :]
        causal_mask = q_idx >= k_idx
        # Reference fills 0 for masked values before SiLU (effectively 0 output)
        attn_scores.masked_fill_(~causal_mask.unsqueeze(0).unsqueeze(2), 0.0) 
        
        # 3. Activation & Scale
        attn_scores = F.silu(attn_scores) * inv_scale
        
        # 4. Local Mask (Post-activation) -> Fill 1.0
        local_mask = (q_idx == k_idx)
        attn_scores.masked_fill_(local_mask.unsqueeze(0).unsqueeze(2), 1.0)
        
        # 5. TopK
        S = min(self.block_counts, num_blocks)
        _, topk_indices = attn_scores.topk(S, dim=-1)
        topk_indices[:, :, :, 1::2] = topk_indices[:, :, :, 0::2]

        g_cmp_jag = FBGEMM_Ops.dense_to_jagged(g_cmp.unsqueeze(-1), [x_offsets])[0].squeeze(-1)
        g_slc_jag = FBGEMM_Ops.dense_to_jagged(g_slc.unsqueeze(-1), [x_offsets])[0].squeeze(-1) # Still computed but not used
        topk_jag = FBGEMM_Ops.dense_to_jagged(topk_indices.view(B, max_seq_len, -1), [x_offsets])[0].view(-1, self.num_heads, S).contiguous()

        o_cmp = torch.empty_like(q)
        o_slc = torch.empty_like(q)
        
        grid_dim = (triton.cdiv(max_seq_len, 32), self.num_heads, B)
        
        hstu_bsa_cmp_kernel[grid_dim](
            Q=q, K=k_cmp, V=v_cmp, G_cmp=g_cmp_jag, Out=o_cmp, Offsets=x_offsets,
            stride_q_t=q.stride(0), stride_q_h=q.stride(1), stride_q_d=q.stride(2),
            stride_k_b=k_cmp.stride(0), stride_k_blk=k_cmp.stride(1), stride_k_h=k_cmp.stride(2), stride_k_d=k_cmp.stride(3),
            stride_v_b=v_cmp.stride(0), stride_v_blk=v_cmp.stride(1), stride_v_h=v_cmp.stride(2), stride_v_d=v_cmp.stride(3),
            stride_o_t=o_cmp.stride(0), stride_o_h=o_cmp.stride(1), stride_o_d=o_cmp.stride(2),
            stride_g_t=g_cmp_jag.stride(0), stride_g_h=g_cmp_jag.stride(1),
            scale=scale, inv_scale=inv_scale, # [Fix]
            BLOCK_SIZE=self.block_size, HEAD_DIM=self.head_dim, BLOCK_M=32, BLOCK_N=32
        )
        
        hstu_bsa_slc_kernel[grid_dim](
            Q=q, K=k, V=v, BlockIdx=topk_jag, Out=o_slc, Offsets=x_offsets, # [Fix] Removed G_slc
            stride_q_t=q.stride(0), stride_q_h=q.stride(1), stride_q_d=q.stride(2),
            stride_idx_t=topk_jag.stride(0), stride_idx_h=topk_jag.stride(1), stride_idx_s=topk_jag.stride(2),
            # stride_g_t=g_slc_jag.stride(0), stride_g_h=g_slc_jag.stride(1),
            scale=scale, inv_scale=inv_scale, # [Fix]
            S=S, BLOCK_SIZE=self.block_size, HEAD_DIM=self.head_dim, BLOCK_M=32
        )

        hidden_size = self.num_heads * self.head_dim
        o_cmp = F.layer_norm(o_cmp.view(total_tokens, hidden_size), [hidden_size], eps=1e-6).view(total_tokens, self.num_heads, self.head_dim) * u
        o_slc = F.layer_norm(o_slc.view(total_tokens, hidden_size), [hidden_size], eps=1e-6).view(total_tokens, self.num_heads, self.head_dim) * u
        
        return (o_cmp + o_slc).view(total_tokens, -1)

# ==============================================================================
# Part 4: Verification Script (No Changes needed, just Run)
# ==============================================================================

# Mock Ops Setup
if not hasattr(torch.ops, "fbgemm"):
    class MockOps: pass
    torch.ops.fbgemm = MockOps()
torch.ops.fbgemm.jagged_to_padded_dense = FBGEMM_Ops.jagged_to_padded_dense
torch.ops.fbgemm.dense_to_jagged = FBGEMM_Ops.dense_to_jagged

class DeterministicGate(nn.Module):
    def __init__(self, g_cmp, g_slc, g_swa):
        super().__init__()
        self.register_buffer('g_cmp', g_cmp)
        self.register_buffer('g_slc', g_slc)
        self.register_buffer('g_swa', g_swa)
    def forward(self, x): return self.g_cmp, self.g_slc, self.g_swa

# --- Reference Implementation Helper Functions ---
def ref_layernorm(x, eps=1e-6):
    bsize, seq_len, num_heads, head_dim = x.shape
    return F.layer_norm(
        x.reshape(bsize, seq_len, -1), normalized_shape=[head_dim * num_heads], eps=eps
    ).reshape(bsize, seq_len, num_heads, head_dim)

def ref_compression(k, v, block_size):
    B, T, H = k.shape[:3]
    num_block = math.ceil(T / block_size)
    if k.shape[1] % block_size != 0:
        k = F.pad(k, (0, 0, 0, 0, 0, num_block * block_size - T))
        v = F.pad(v, (0, 0, 0, 0, 0, num_block * block_size - T))
    k_cmp = k.view(B, num_block, block_size, H, -1).mean(dim=2) 
    v_cmp = v.view(B, num_block, block_size, H, -1).mean(dim=2) 
    return k_cmp, v_cmp

def ref_bsa_compression(q, k, v, u, g_cmp, block_counts, block_size, scale):
    bsize, seq_len, num_heads, attn_dim = q.shape 
    BS = block_size
    q, k, v, u = map(lambda x: x.float(), (q, k, v, u))
    k_cmp, v_cmp = ref_compression(k, v, BS) 
    C = k_cmp.shape[1] 
    S = min(block_counts, C)

    casual_mask = ((torch.arange(seq_len) - BS + 1)[:, None] // BS < torch.arange(C)[None, :]).to(q.device)
    local_mask = (torch.arange(seq_len)[:, None] // BS == torch.arange(C)[None, :]).to(q.device)

    attn_cmp = torch.einsum('bqhd,bkhd->bhqk', q*scale, k_cmp)
    # [Fix] Reference uses 0 mask before SiLU
    attn_cmp = attn_cmp.masked_fill(~casual_mask.unsqueeze(0).unsqueeze(0), 0.0)
    attn_cmp = F.silu(attn_cmp) / scale
    o_cmp = torch.einsum('bhqk, bkhd -> bqhd', attn_cmp, v_cmp) * g_cmp.unsqueeze(-1)
    o_cmp = ref_layernorm(o_cmp)*u
    
    attn_select = attn_cmp.masked_fill(local_mask.unsqueeze(0).unsqueeze(0), float(1.0))
    block_indices = attn_select.topk(S, -1)[1]
    
    range_t = torch.arange(seq_len, device=q.device)
    block_indices = block_indices.masked_fill(block_indices > (range_t[None, None, :, None] // BS), -1)
    return block_indices, o_cmp.to(q.dtype)

def ref_bsa_cal(q, k, v, u, g_slc, block_indices, block_size, scale):
    bsize, seq_len, num_heads, head_dim = q.shape
    S = block_indices.shape[-1] 
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
    
    current_t = torch.arange(seq_len, device=q.device).view(1, seq_len, 1, 1)
    mask = (~valid_mask) | (gather_ids > current_t)
    
    attn_logits = attn_logits.masked_fill(mask, 0)
    attn_weights = F.silu(attn_logits) / scale
    
    o_slc = torch.matmul(attn_weights.unsqueeze(3), v_slc).squeeze(3)
    # [Check] User's code does NOT multiply g_slc here
    o_slc = ref_layernorm(o_slc)*u
    return o_slc

def ref_hstu_attention_with_bsa(num_heads, attention_dim, linear_dim, q, k, v, u, x_offsets, gate_model):
    B = x_offsets.size(0) - 1
    seq_lens = x_offsets[1:] - x_offsets[:-1]
    n = seq_lens.max().item()
    
    padded_q = FBGEMM_Ops.jagged_to_padded_dense(q, [x_offsets], [n], 0.0).view(B, n, num_heads, attention_dim)
    padded_k = FBGEMM_Ops.jagged_to_padded_dense(k, [x_offsets], [n], 0.0).view(B, n, num_heads, attention_dim)
    padded_v = FBGEMM_Ops.jagged_to_padded_dense(v, [x_offsets], [n], 0.0).view(B, n, num_heads, linear_dim)
    padded_u = FBGEMM_Ops.jagged_to_padded_dense(u, [x_offsets], [n], 0.0).view(B, n, num_heads, linear_dim)

    g_cmp, g_slc, g_swa = gate_model(padded_q)
    scale = attention_dim ** -0.5

    block_indices, o_cmp = ref_bsa_compression(padded_q, padded_k, padded_v, padded_u, g_cmp, 4, 32, scale)
    block_indices[:, :, :, 1::2] = block_indices[:, :, :, 0::2]
    o_slc = ref_bsa_cal(padded_q, padded_k, padded_v, padded_u, g_slc, block_indices, 32, scale)
    
    attn_output = o_cmp + o_slc
    attn_output = attn_output.reshape(B, n, num_heads * linear_dim)
    attn_output = FBGEMM_Ops.dense_to_jagged(attn_output, [x_offsets])[0]
    return attn_output

def run_comparison():
    print("=== HSTU BSA Correctness Test ===")
    if not torch.cuda.is_available(): return
    device = "cuda"
    torch.manual_seed(42)
    B, H, D = 2, 8, 64
    
    lengths = [64, 128]
    max_len = max(lengths)
    offsets = [0]
    for l in lengths: offsets.append(offsets[-1] + l)
    x_offsets = torch.tensor(offsets, dtype=torch.long, device=device)
    total_tokens = offsets[-1]
    
    q = torch.randn(total_tokens, H, D, device=device)
    k = torch.randn(total_tokens, H, D, device=device)
    v = torch.randn(total_tokens, H, D, device=device)
    u = torch.randn(total_tokens, H, D, device=device)

    g_cmp_val = torch.rand(B, max_len, H, device=device)
    g_slc_val = torch.rand(B, max_len, H, device=device)
    g_swa_val = torch.rand(B, max_len, H, device=device)
    gate_model = DeterministicGate(g_cmp_val, g_slc_val, g_swa_val)

    print("Running PyTorch Reference...")
    out_ref = ref_hstu_attention_with_bsa(H, D, D, q, k, v, u, x_offsets, gate_model)

    print("Running Triton...")
    triton_model = HSTU_BSA_Triton(H, D, 32, 4).to(device)
    out_triton = triton_model(q, k, v, u, x_offsets, gate_model)

    print("\n=== Results ===")
    diff = (out_ref - out_triton).abs()
    print(f"Max Diff: {diff.max().item():.6f}")
    if diff.max().item() < 1e-3: print("✅ PASSED")
    else: print("❌ FAILED")

if __name__ == "__main__":
    run_comparison()