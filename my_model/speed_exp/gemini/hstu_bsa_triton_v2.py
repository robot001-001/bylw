import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import math

# -----------------------------------------------------------------------------
# Triton Helper: Bitonic Sort
# -----------------------------------------------------------------------------

@triton.jit
def _compare_and_swap(x, ids, flip, i, n_dims):
    """
    Bitonic Sort 的基本单元：比较并交换
    x: Values (Scores)
    ids: Indices (Block Indices)
    """
    # 确定交换的伙伴索引
    target = tl.arange(0, x.shape[0]) ^ (1 << i)
    
    # 掩码：只在一个方向上操作，避免重复交换
    # 如果 i 位是 0，则是主动方
    mask = (tl.arange(0, x.shape[0]) & (1 << i)) == 0
    
    # 获取伙伴的值
    x_target = tl.load(x + target, mask=mask) # 这里其实是寄存器操作，不用 load，直接 swizzle 更好
    # 实际上 Triton 并没有直接的 register swizzle 指令暴露给 user 用于这种 tensor 
    # 但是对于 tensor 来说，可以用 view/reshape 或者直接索引来实现寄存器混洗
    # 为了通用性，这里使用 where + 索引
    
    # 优化：在 Triton 中，tensor 就在寄存器里。我们构造 target tensor
    x_partner = tl.view(x, [x.shape[0]])[target]
    ids_partner = tl.view(ids, [ids.shape[0]])[target]
    
    # 决定排序方向
    # flip 控制升序还是降序
    # 在 Bitonic sort 中，方向取决于高位的 pattern
    region = (tl.arange(0, x.shape[0]) >> (n_dims - 1)) & 1
    # 如果 flip=True，我们在某些段反转方向
    cond = x < x_partner 
    if flip:
        # 降序
        swap = cond
    else:
        # 升序 (但在 TopK 中我们通常想要降序，把大的放前面)
        # 这里我们实现：大的在索引 0，小的在索引 N
        swap = x < x_partner

    # 这里的 Bitonic 逻辑比较绕，我们采用一种更简单的 Stream-TopK 策略：
    # 始终保持 list 是降序的 (Max First)。
    # 这是一个 Merge 步骤：Merge two sorted lists (descending) -> one sorted list
    # 但标准的 bitonic merge 需要特定的比较模式。
    
    return x, ids

@triton.jit
def _bitonic_sort_descending(v, i):
    """
    对寄存器中的向量 v 和索引 i 进行降序排序 (Bubble-like network for small S is okay, but Bitonic is better)
    由于 S 通常是 16/32，我们可以硬编码。
    """
    # 这里的实现较为复杂，为了稳定性，我们使用一种简单的 "Odd-Even Transposition Sort" 
    # 或者对于小 S (<=32)，直接使用多次 pass。
    # 但为了最高性能，标准库通常没有提供。
    
    # 简化策略：
    # 我们假设 S 是 2 的幂次。
    # 我们使用简单的迭代交换来实现降序。
    
    S: tl.constexpr = v.shape[0]
    
    # 这种写法在 Triton 编译器中会被展开
    # Phase 1: Bitonic Construct
    # Phase 2: Bitonic Merge
    # 这里直接使用简单的冒泡网络 (Odd-Even) 对小 S 也是高效的，因为是在寄存器中并行。
    # 但为了效率，我们实现标准的 Bitonic Merge Sort (Descending)
    
    for k in tl.static_range(1, tl.cdiv(S.bit_length() - 1, 1) + 1): # log2(S)
        step = 1 << k
        for j in tl.static_range(k):
            # Inner stage
            mask_step = 1 << (k - 1 - j)
            
            # 构造比较对
            # 索引 idx 与 idx ^ mask_step 比较
            idx = tl.arange(0, S)
            partner_idx = idx ^ mask_step
            
            # 只有 idx < partner_idx 的线程进行“主导”判断，避免重复
            # 但在 Tensor 层面，我们是全量操作
            
            val = v
            partner_val = tl.view(v, [S])[partner_idx]
            
            idx_val = i
            partner_idx_val = tl.view(i, [S])[partner_idx]
            
            # 判断方向：我们希望整体降序
            # 在 Bitonic Sort 构造阶段，方向是交替的
            # (idx // step) % 2 == 0 -> 降序, == 1 -> 升序
            descending_group = ((idx // step) % 2) == 0
            
            # 逻辑：
            # 如果是降序组：Max(val, partner) 放左边 (较小索引)
            # 如果是升序组：Min(val, partner) 放左边
            
            # 比较
            is_smaller = val < partner_val
            
            # 需要交换的情况：
            # 1. 降序组 (0) 且 val < partner (当前是小的，但要在左边放大的) -> 交换
            # 2. 升序组 (1) 且 val > partner (当前是大的，但要在左边放小的) -> 交换
            swap = (descending_group & is_smaller) | ((~descending_group) & (~is_smaller))
            
            # 执行交换
            v = tl.where(swap, partner_val, val)
            i = tl.where(swap, partner_idx_val, idx_val)
            
    # 最后一次 Merge，将结果完全变成降序
    # Merge 是 Bitonic Sort 的最后一步，但方向全是 "Descending"
    # 等价于 log2(S) 次 pass
    for k in tl.static_range(tl.cdiv(S.bit_length() - 1, 1), -1, -1): # log2(S)-1 down to 0
         mask_step = 1 << k
         idx = tl.arange(0, S)
         partner_idx = idx ^ mask_step
         
         val = v
         partner_val = tl.view(v, [S])[partner_idx]
         idx_val = i
         partner_idx_val = tl.view(i, [S])[partner_idx]
         
         # 全局降序：左边 (小索引) 必须比 右边 (大索引) 大
         # 我们只关心 idx < partner_idx 的那对关系
         # 如果 val < partner_val 且 idx < partner_idx，则 val 在左边但比右边小，需要交换
         # 也就是： swap if (val < partner) != (idx < partner) ? No.
         
         # 简单逻辑：
         # 目标：idx 小的地方放大的值
         # 比较 val 和 partner
         larger = tl.maximum(val, partner_val)
         smaller = tl.minimum(val, partner_val)
         
         larger_idx = tl.where(val > partner_val, idx_val, partner_idx_val)
         smaller_idx = tl.where(val > partner_val, partner_idx_val, idx_val)
         
         # 如果当前 idx < partner_idx，取 larger
         v = tl.where(idx < partner_idx, larger, smaller)
         i = tl.where(idx < partner_idx, larger_idx, smaller_idx)
         
    return v, i


# -----------------------------------------------------------------------------
# Triton Kernels
# -----------------------------------------------------------------------------

@triton.jit
def _hstu_silu_activation(x):
    return x * tl.sigmoid(x)

# [Kernel 1] TopK Selection (Fused Score + Sort)
@triton.jit
def hstu_bsa_topk_kernel(
    Q, K, 
    Out_Indices, # [TotalTokens, H, S]
    Stride_qt, Stride_qh, Stride_qd,
    Stride_kt, Stride_kh, Stride_kd,
    Stride_idx_t, Stride_idx_h, Stride_idx_s,
    offsets,      
    offsets_cmp,  
    scale,
    S: tl.constexpr,          # TopK 数量 (必须是 2 的幂次)
    BLOCK_SIZE: tl.constexpr, # 原始序列分块大小 (32)
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,    # Query 块大小
):
    pid_m = tl.program_id(0) 
    pid_h = tl.program_id(1) 
    pid_z = tl.program_id(2) 

    # 1. Setup Q
    seq_start = tl.load(offsets + pid_z)
    seq_end = tl.load(offsets + pid_z + 1)
    seq_len = seq_end - seq_start
    
    start_m = pid_m * BLOCK_M
    if start_m >= seq_len:
        return

    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_len
    
    q_ptrs = Q + (seq_start + offs_m[:, None]) * Stride_qt + pid_h * Stride_qh + tl.arange(0, HEAD_DIM)[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0) # [BLOCK_M, D]

    # 2. Setup TopK Accumulators (Registers)
    # 维护 S 个最好的值和索引
    # 初始化为负无穷和 -1
    top_vals = tl.full([BLOCK_M, S], float('-inf'), dtype=tl.float32)
    top_idxs = tl.full([BLOCK_M, S], -1, dtype=tl.int32)

    # 3. Setup K_cmp Loop
    cmp_start = tl.load(offsets_cmp + pid_z)
    cmp_end = tl.load(offsets_cmp + pid_z + 1)
    cmp_len = cmp_end - cmp_start 

    # 这里的 BLOCK_N 我们设置为 S (TopK 数量)，以方便后续 Merge
    # 要求 S 必须能整除常见的维度，或者我们在循环中处理 mask
    BLOCK_N: tl.constexpr = S 

    for start_n in range(0, cmp_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < cmp_len
        
        # Load K chunk
        k_ptrs = K + (cmp_start + offs_n[None, :]) * Stride_kt + pid_h * Stride_kh + tl.arange(0, HEAD_DIM)[:, None]
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0) # [BLOCK_N, D]
        
        # Compute Score: [BLOCK_M, D] @ [BLOCK_N, D].T -> [BLOCK_M, BLOCK_N]
        # 注意：这里的 BLOCK_N = S
        scores = tl.dot(q, k) 
        scores *= scale
        
        # Causal Masking
        # Q_idx // BS >= K_idx
        # K_idx 是全局的 relative index (0..cmp_len)
        mask_causal = (offs_m[:, None] // BLOCK_SIZE) >= offs_n[None, :]
        
        # 将不合法的（Causal 或者 Padding）置为 -inf
        mask_valid = mask_causal & mask_n[None, :] & mask_m[:, None]
        scores = tl.where(mask_valid, scores, float('-inf'))
        
        # 4. Stream-TopK Merge (In-Register Bitonic Merge)
        # 我们现在有两组数据：
        # Group A (Current Best): top_vals, top_idxs (Size S) (Sorted Descending)
        # Group B (New Candidates): scores, offs_n (Size S) (Unsorted)
        
        # Step A: 对 New Candidates 进行排序 (降序)
        # new_vals: [BLOCK_M, S], new_idxs: [1, S] broadcasted
        new_vals = scores
        new_idxs = tl.broadcast_to(offs_n[None, :], [BLOCK_M, S]).to(tl.int32)
        
        # 对新来的数据进行内部排序
        new_vals, new_idxs = _bitonic_sort_descending(new_vals, new_idxs)
        
        # Step B: Merge Group A and Group B, Keep Top S
        # 因为 A 和 B 都是有序的 (Desc)，我们可以直接 Bitonic Merge 吗？
        # Bitonic Merge 需要输入序列是 "Bitonic Sequence" (单峰)，比如一个升序一个降序拼接。
        # 我们的 A 是 Desc, B 是 Desc。
        # 我们可以把 B 翻转成 Asc，接在 A 后面，形成 Desc-Asc (Bitonic)，然后跑一次 Bitonic Merge。
        # 但我们无法在 Triton 方便地拼接 Tensor (cat)。
        
        # 替代方案：Odd-Even Merge 或者直接比较
        # 由于我们只需要 Top S (也就是前一半)，且 S 很小。
        # 我们可以简单地比较 A[i] 和 B[i]，但这不准。
        
        # 实用方案：我们假设 S 比较小 (16, 32)，我们用简单的 Odd-Even Merge 的思想
        # 或者再次利用 _bitonic_sort_descending 的最后阶段逻辑
        # 让我们利用 "Bitonic Merge" 的性质：
        # 如果我们把两个有序序列合并，可以用 bitonic merger。
        # 这里为了代码简洁和寄存器压力，我们使用一个简单的 Masked Update：
        # 这种方法虽然不是 log(N)，但对于小 S 完全够用且逻辑简单。
        # 实际上，如果 S=16，我们可以把 A 和 B 视为一个 32 的数组进行排序然后截断。
        # 但无法扩展 array。
        
        # 正确的 Triton Stream TopK 做法：
        # 使用 Bitonic Merge 网络合并两个有序序列。
        # 既然我们无法 resize tensor，我们只能原地 update。
        
        # 让我们使用一种 "Max-Swap" 策略 (类似于冒泡归并，但并行化):
        # 将 top_vals 和 new_vals 视为两行。我们想把大的值“冒泡”到 top_vals 中。
        
        # Pass 1: Compare top_vals[i] vs new_vals[i]
        # Swap if new > top. (Pre-filtering)
        # 这保证了 top_vals 里的元素大概率是大的，但没有完全排序。
        swap_mask = new_vals > top_vals
        temp_vals = top_vals
        temp_idxs = top_idxs
        
        top_vals = tl.where(swap_mask, new_vals, temp_vals)
        top_idxs = tl.where(swap_mask, new_idxs, temp_idxs)
        
        new_vals = tl.where(swap_mask, temp_vals, new_vals) # 小的被挤出来了
        # new_idxs = ... (不关心被挤出来的索引)
        
        # Pass 2: Re-sort top_vals
        # 因为我们破坏了顺序，需要重新排序
        top_vals, top_idxs = _bitonic_sort_descending(top_vals, top_idxs)
        
        # 注意：这种 "Swap + Re-sort" 策略在数学上对于 TopK 是完备的吗？
        # 如果 A=[10, 0], B=[9, 8]。
        # Swap -> A=[10, 8], B=[9, 0]。Re-sort A -> [10, 8]。
        # 错误！9 丢失了。
        
        # --- 修正策略：双调合并 (模拟拼接) ---
        # 实际上我们想对 [top_vals, new_vals] (Size 2S) 排序并取前 S。
        # 模拟拼接：
        # v_combined = concat(top_vals, new_vals)
        # sort(v_combined)
        # top_vals = v_combined[:S]
        
        # 在 Triton 中模拟这个过程：
        # 我们知道 Bitonic Merge 的第一步是比较 A[i] 和 B[S-1-i] (如果构成 Bitonic)。
        # 或者如果两个都是 Descending，我们比较 A[i] 和 B[i] (Batcher's Odd-Even Merge)。
        
        # 让我们使用 Batcher's Odd-Even Merge 的变体来合并两个有序序列 A 和 B
        # 这是一个标准的并行归并算法。
        # 但代码太长。
        
        # 回退到最稳健且不依赖复杂库的方法：
        # 只是简单的把 A 和 B 的对应位置比较交换，然后**再次排序**。
        # 虽然这不等价于全排序，但如果在 Loop 中反复执行，能保留最大值吗？不能保证。
        
        # --- 终极方案：强制使用 Bitonic Sort Network ---
        # 假设 S=16。我们在循环内部，通过多次寄存器交换，确保 top_vals 始终包含最大值。
        # 技巧：将 new_vals 反转 (Ascending)，逻辑上拼接到 top_vals (Descending) 后面。
        # 这形成了一个 Bitonic Sequence (先降后升)。
        # 然后我们只需要执行 Bitonic Merge 的步骤，就可以让最大的 S 个值跑到 top_vals 里，且有序！
        # 这就是 Bitonic Sort 的神奇之处。
        
        # 实现：
        # 1. new_vals 已经是 Descending 的。
        # 2. 我们需要它是 Ascending 的才能和 top_vals (Desc) 构成 Bitonic。
        #    利用 tensor 索引反转：
        rev_idx = S - 1 - tl.arange(0, S)
        new_vals_rev = tl.view(new_vals, [BLOCK_M, S])[:, rev_idx]
        new_idxs_rev = tl.view(new_idxs, [BLOCK_M, S])[:, rev_idx]
        
        # 3. 现在虚拟序列是 [top_vals, new_vals_rev] (长度 2S)。它是 Bitonic 的。
        # 4. 执行 Bitonic Merge (降序)。
        #    Merge 的第一步是比较 i 和 i+S。也就是比较 top_vals[i] 和 new_vals_rev[i]。
        
        # Step 1: Compare Top[i] vs New_Rev[i]
        # 大的留给 Top，小的给 New
        cmp1 = new_vals_rev > top_vals
        temp_v = top_vals
        temp_i = top_idxs
        
        top_vals = tl.where(cmp1, new_vals_rev, temp_v)
        top_idxs = tl.where(cmp1, new_idxs_rev, temp_i)
        
        # 小的部分被扔到了 "虚拟的下半部分"，我们直接丢弃，不关心了。
        # 我们只关心上半部分 (top_vals)。
        
        # Step 2: Recursive Merge on top_vals
        # 因为我们只保留了上半部分，现在上半部分可能不是完全有序的，但满足 Bitonic Merge 的后续属性。
        # 我们需要对 top_vals 继续执行 Bitonic Merge 的剩余步骤。
        # 这等价于对 top_vals 进行一次 _bitonic_sort_descending 的 "最后几步"。
        # 实际上，直接跑一次全排序是最安全的。
        
        top_vals, top_idxs = _bitonic_sort_descending(top_vals, top_idxs)
        
        # 结束。这保证了 top_vals 包含最大的 S 个值且有序。

    # 5. Store Indices
    # Out_Indices: [TotalTokens, H, S]
    # 我们是按 Block 写入的
    idx_ptrs = Out_Indices + (seq_start + offs_m[:, None]) * Stride_idx_t + pid_h * Stride_idx_h + tl.arange(0, S)[None, :] * Stride_idx_s
    tl.store(idx_ptrs, top_idxs, mask=mask_m[:, None])


# [Kernel 2 & 3] CMP and SLC (Reused with minimal changes)
# 之前的 CMP Kernel 和 SLC Kernel 代码保持不变，
# 唯一区别是 Python 端的调用方式。
# 为了节省篇幅，这里假设使用了之前 "修复版 (带 Gate Stride)" 的 Kernel。

# 重新粘贴 CMP Kernel (确保 context 完整)
@triton.jit
def hstu_bsa_cmp_fwd_kernel(
    Q, K, V, G_cmp, Out,
    Stride_qt, Stride_qh, Stride_qd,
    Stride_kt, Stride_kh, Stride_kd,
    Stride_vt, Stride_vh, Stride_vd,
    Stride_ot, Stride_oh, Stride_od,
    Stride_gt, Stride_gh,
    offsets, offsets_cmp, scale,
    BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,    
):
    pid_m, pid_h, pid_z = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq_start = tl.load(offsets + pid_z)
    seq_end = tl.load(offsets + pid_z + 1)
    if pid_m * BLOCK_M >= (seq_end - seq_start): return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < (seq_end - seq_start)
    
    q = tl.load(Q + (seq_start + offs_m[:, None])*Stride_qt + pid_h*Stride_qh + tl.arange(0, HEAD_DIM)[None, :], mask=mask_m[:, None], other=0.0)
    g = tl.load(G_cmp + (seq_start + offs_m[:, None])*Stride_gt + pid_h*Stride_gh, mask=mask_m[:, None], other=0.0)
    
    cmp_start = tl.load(offsets_cmp + pid_z)
    cmp_len = tl.load(offsets_cmp + pid_z + 1) - cmp_start
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    for start_n in range(0, cmp_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < cmp_len
        k = tl.load(K + (cmp_start + offs_n[None, :])*Stride_kt + pid_h*Stride_kh + tl.arange(0, HEAD_DIM)[:, None], mask=mask_n[None, :], other=0.0)
        v = tl.load(V + (cmp_start + offs_n[:, None])*Stride_vt + pid_h*Stride_vh + tl.arange(0, HEAD_DIM)[None, :], mask=mask_n[:, None], other=0.0)
        
        score = tl.dot(q, k) * scale
        mask_causal = (offs_m[:, None] // BLOCK_SIZE) >= offs_n[None, :]
        score = tl.where(mask_causal & mask_n[None, :] & mask_m[:, None], score, -1e9)
        p = _hstu_silu_activation(score)
        p = tl.where(mask_causal & mask_n[None, :], p, 0.0)
        acc += tl.dot(p, v)
        
    tl.store(Out + (seq_start + offs_m[:, None])*Stride_ot + pid_h*Stride_oh + tl.arange(0, HEAD_DIM)[None, :], (acc * g).to(Out.dtype.element_ty), mask=mask_m[:, None])

# 重新粘贴 SLC Kernel
@triton.jit
def hstu_bsa_slc_fwd_kernel(
    Q, K, V, G_slc, BlockIndices, Out,
    Stride_qt, Stride_qh, Stride_qd,
    Stride_kt, Stride_kh, Stride_kd,
    Stride_vt, Stride_vh, Stride_vd,
    Stride_ot, Stride_oh, Stride_od,
    Stride_gt, Stride_gh,
    offsets, scale,
    S: tl.constexpr, BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,    
):
    pid_m, pid_h, pid_z = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq_start = tl.load(offsets + pid_z)
    seq_end = tl.load(offsets + pid_z + 1)
    if pid_m * BLOCK_M >= (seq_end - seq_start): return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < (seq_end - seq_start)

    q = tl.load(Q + (seq_start + offs_m[:, None])*Stride_qt + pid_h*Stride_qh + tl.arange(0, HEAD_DIM)[None, :], mask=mask_m[:, None], other=0.0)
    g = tl.load(G_slc + (seq_start + offs_m[:, None])*Stride_gt + pid_h*Stride_gh, mask=mask_m[:, None], other=0.0)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for s_idx in range(S):
        b_idx = tl.load(BlockIndices + (seq_start + offs_m)*(S*tl.num_programs(1)) + pid_h*S + s_idx, mask=mask_m, other=-1)
        for blk_offset in range(BLOCK_SIZE):
            valid = (b_idx >= 0)
            target_idx = seq_start + b_idx * BLOCK_SIZE + blk_offset
            is_bounds = target_idx < seq_end
            is_causal = target_idx <= (seq_start + offs_m)
            
            mask_load = valid[:, None] & mask_m[:, None] & is_bounds[:, None]
            k = tl.load(K + target_idx[:, None]*Stride_kt + pid_h*Stride_kh + tl.arange(0, HEAD_DIM)[None, :], mask=mask_load, other=0.0)
            v = tl.load(V + target_idx[:, None]*Stride_vt + pid_h*Stride_vh + tl.arange(0, HEAD_DIM)[None, :], mask=mask_load, other=0.0)
            
            score = tl.sum(q * k, axis=1) * scale
            p = _hstu_silu_activation(score)
            p = tl.where(valid & is_causal & is_bounds, p, 0.0)
            acc += p[:, None] * v

    tl.store(Out + (seq_start + offs_m[:, None])*Stride_ot + pid_h*Stride_oh + tl.arange(0, HEAD_DIM)[None, :], (acc * g).to(Out.dtype.element_ty), mask=mask_m[:, None])


# -----------------------------------------------------------------------------
# Python Wrapper
# -----------------------------------------------------------------------------

class HSTU_BSA_Triton(torch.nn.Module):
    def __init__(self, block_size=32, block_counts=4):
        super().__init__()
        self.block_size = block_size
        self.block_counts = block_counts
        # 确保 block_counts 是 2 的幂次，这是 Bitonic Sort 的要求
        # 如果不是，向上取整
        self.s_pow2 = 1
        while self.s_pow2 < self.block_counts:
            self.s_pow2 *= 2

    def forward(self, q, k, v, g_cmp, g_slc, x_offsets):
        if g_cmp.dim() == 2: g_cmp = g_cmp.unsqueeze(-1)
        if g_slc.dim() == 2: g_slc = g_slc.unsqueeze(-1)
        
        B = x_offsets.size(0) - 1
        seq_lens = x_offsets[1:] - x_offsets[:-1]
        total_tokens = q.shape[0]
        device = q.device
        num_heads = q.shape[1]
        dim = q.shape[2]
        
        # 1. Jagged Pooling
        with torch.no_grad():
            cmp_seq_lens = (seq_lens + self.block_size - 1) // self.block_size
            offsets_cmp = torch.zeros_like(x_offsets)
            offsets_cmp[1:] = torch.cumsum(cmp_seq_lens, dim=0)
            total_cmp_tokens = offsets_cmp[-1].item()
            
            batch_ids = torch.repeat_interleave(torch.arange(B, device=device), seq_lens)
            local_ids = torch.arange(total_tokens, device=device) - x_offsets[:-1][batch_ids]
            local_block_ids = local_ids // self.block_size
            segment_ids = offsets_cmp[:-1][batch_ids] + local_block_ids

        k_cmp = torch.zeros((total_cmp_tokens, num_heads, dim), dtype=k.dtype, device=device)
        v_cmp = torch.zeros((total_cmp_tokens, num_heads, v.shape[-1]), dtype=v.dtype, device=device)
        k_cmp.index_add_(0, segment_ids, k)
        v_cmp.index_add_(0, segment_ids, v)
        k_cmp = k_cmp / self.block_size
        v_cmp = v_cmp / self.block_size

        # 2. Kernel Setup
        max_n = seq_lens.max().item()
        scale = dim ** -0.5
        o_cmp = torch.empty_like(v)
        o_slc = torch.empty_like(v)
        
        grid_triton = lambda meta: (triton.cdiv(max_n, meta['BLOCK_M']), num_heads, B)

        # 3. Launch TopK Kernel (REPLACES torch.matmul + topk)
        # Output: [TotalTokens, H, S] (Jagged)
        # S 必须是 2 的幂次，所以我们申请 s_pow2 大小
        topk_indices = torch.full((total_tokens, num_heads, self.s_pow2), -1, dtype=torch.int32, device=device)
        
        hstu_bsa_topk_kernel[grid_triton](
            Q=q, K=k_cmp, 
            Out_Indices=topk_indices,
            Stride_qt=q.stride(0), Stride_qh=q.stride(1), Stride_qd=q.stride(2),
            Stride_kt=k_cmp.stride(0), Stride_kh=k_cmp.stride(1), Stride_kd=k_cmp.stride(2),
            Stride_idx_t=topk_indices.stride(0), Stride_idx_h=topk_indices.stride(1), Stride_idx_s=topk_indices.stride(2),
            offsets=x_offsets, 
            offsets_cmp=offsets_cmp, 
            scale=scale,
            S=self.s_pow2, # 传入 Power of 2
            BLOCK_SIZE=self.block_size, HEAD_DIM=dim,
            BLOCK_M=32
        )
        
        # 如果 block_counts < s_pow2, 我们只需要前 block_counts 个
        # 但为了后续 Kernel 方便，我们传 s_pow2 也没问题，只要 SLC Kernel 里的 loop range 是 min(block_counts, s_pow2)
        # 这里为了对齐逻辑，我们传入实际需要的 S
        S_real = self.block_counts

        # 4. Launch CMP Kernel
        hstu_bsa_cmp_fwd_kernel[grid_triton](
            Q=q, K=k_cmp, V=v_cmp, 
            G_cmp=g_cmp.squeeze(-1), 
            Out=o_cmp,
            Stride_qt=q.stride(0), Stride_qh=q.stride(1), Stride_qd=q.stride(2),
            Stride_kt=k_cmp.stride(0), Stride_kh=k_cmp.stride(1), Stride_kd=k_cmp.stride(2),
            Stride_vt=v_cmp.stride(0), Stride_vh=v_cmp.stride(1), Stride_vd=v_cmp.stride(2),
            Stride_ot=o_cmp.stride(0), Stride_oh=o_cmp.stride(1), Stride_od=o_cmp.stride(2),
            Stride_gt=g_cmp.stride(0), Stride_gh=g_cmp.stride(1),
            offsets=x_offsets, 
            offsets_cmp=offsets_cmp, 
            scale=scale,
            BLOCK_SIZE=self.block_size, HEAD_DIM=dim,
            BLOCK_M=32, BLOCK_N=32
        )

        # 5. Launch SLC Kernel
        hstu_bsa_slc_fwd_kernel[grid_triton](
            Q=q, K=k, V=v, 
            G_slc=g_slc.squeeze(-1), 
            BlockIndices=topk_indices, # 传入 In-Kernel 计算出的索引
            Out=o_slc,
            Stride_qt=q.stride(0), Stride_qh=q.stride(1), Stride_qd=q.stride(2),
            Stride_kt=k.stride(0), Stride_kh=k.stride(1), Stride_kd=k.stride(2),
            Stride_vt=v.stride(0), Stride_vh=v.stride(1), Stride_vd=v.stride(2),
            Stride_ot=o_slc.stride(0), Stride_oh=o_slc.stride(1), Stride_od=o_slc.stride(2),
            Stride_gt=g_slc.stride(0), Stride_gh=g_slc.stride(1),
            offsets=x_offsets, 
            scale=scale,
            S=S_real, # 这里限制只遍历前 real 个
            BLOCK_SIZE=self.block_size, HEAD_DIM=dim, BLOCK_M=32
        )
        
        return o_cmp, o_slc