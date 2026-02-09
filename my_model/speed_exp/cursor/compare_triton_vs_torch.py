# -*- coding: utf-8 -*-
"""
对比脚本：验证 Triton 实现与 PyTorch 实现在相同输入下的输出一致性。
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.nsa.nsa_bsa_interleave import block_sparse_attn
from model.HSTU_bsa_pretrain_interleave import _hstu_attention_with_bsa
from model.GateModel import GateModel
from speed_exp.triton_bsa.triton_bsa_attention import triton_block_sparse_attn
from speed_exp.triton_bsa.triton_hstu_bsa import triton_hstu_attention_with_bsa


def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, name: str = "", rtol: float = 1e-4, atol: float = 1e-5, verbose: bool = True):
    """
    比较两个张量是否相同
    
    Args:
        tensor1: 第一个张量
        tensor2: 第二个张量
        name: 张量名称（用于报告）
        rtol: 相对误差容忍度
        atol: 绝对误差容忍度
        verbose: 是否打印详细信息
    
    Returns:
        is_close: 是否在容忍范围内
        max_diff: 最大差异
        mean_diff: 平均差异
        stats: 统计信息字典
    """
    if tensor1.shape != tensor2.shape:
        if verbose:
            print(f"✗ {name}: 形状不匹配 - {tensor1.shape} vs {tensor2.shape}")
        return False, None, None, {}
    
    # 转换为 float32 进行比较
    t1 = tensor1.float().cpu()
    t2 = tensor2.float().cpu()
    
    # 计算差异
    diff = torch.abs(t1 - t2)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    std_diff = diff.std().item()
    
    # 计算相对误差
    rel_diff = diff / (torch.abs(t1) + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    # 统计超过阈值的元素数量
    num_elements = diff.numel()
    num_exceed_rtol = (rel_diff > rtol).sum().item()
    num_exceed_atol = (diff > atol).sum().item()
    
    # 检查是否在容忍范围内
    is_close = torch.allclose(t1, t2, rtol=rtol, atol=atol)
    
    stats = {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
        "num_elements": num_elements,
        "num_exceed_rtol": num_exceed_rtol,
        "num_exceed_atol": num_exceed_atol,
        "tensor1_range": [t1.min().item(), t1.max().item()],
        "tensor2_range": [t2.min().item(), t2.max().item()],
    }
    
    if verbose:
        if is_close:
            print(f"✓ {name}: 匹配")
            print(f"  - max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, std_diff={std_diff:.6e}")
            print(f"  - max_rel_diff={max_rel_diff:.6e}, mean_rel_diff={mean_rel_diff:.6e}")
        else:
            print(f"✗ {name}: 不匹配")
            print(f"  - max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, std_diff={std_diff:.6e}")
            print(f"  - max_rel_diff={max_rel_diff:.6e}, mean_rel_diff={mean_rel_diff:.6e}")
            print(f"  - 超过 rtol 的元素: {num_exceed_rtol}/{num_elements} ({100*num_exceed_rtol/num_elements:.2f}%)")
            print(f"  - 超过 atol 的元素: {num_exceed_atol}/{num_elements} ({100*num_exceed_atol/num_elements:.2f}%)")
            print(f"  - tensor1 range: [{t1.min().item():.6f}, {t1.max().item():.6f}]")
            print(f"  - tensor2 range: [{t2.min().item():.6f}, {t2.max().item():.6f}]")
            
            # 找出差异最大的位置
            max_idx = diff.argmax()
            max_idx_unraveled = np.unravel_index(max_idx.item(), diff.shape)
            print(f"  - 最大差异位置: {max_idx_unraveled}")
            print(f"  - tensor1[{max_idx_unraveled}] = {t1.flatten()[max_idx].item():.6f}")
            print(f"  - tensor2[{max_idx_unraveled}] = {t2.flatten()[max_idx].item():.6f}")
            print(f"  - 差异值 = {diff.flatten()[max_idx].item():.6f}")
    
    return is_close, max_diff, mean_diff, stats


def test_block_sparse_attn():
    """测试 block_sparse_attn 函数"""
    print("=" * 80)
    print("测试 block_sparse_attn")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置随机种子
    set_seed(42)
    
    # 创建测试数据
    B = 2
    T = 128
    H = 8
    ATTN_DIM = 32
    LINEAR_DIM = 64
    block_counts = 4
    block_size = 32
    
    q = torch.randn(B, T, H, ATTN_DIM, device=device)
    k = torch.randn(B, T, H, ATTN_DIM, device=device)
    v = torch.randn(B, T, H, LINEAR_DIM, device=device)
    u = torch.randn(B, T, H, LINEAR_DIM, device=device)
    g_cmp = torch.randn(B, T, H, device=device)
    g_slc = torch.randn(B, T, H, device=device)
    g_swa = None
    
    # PyTorch 实现
    print("\n运行 PyTorch 实现...")
    set_seed(42)
    q_torch = q.clone()
    k_torch = k.clone()
    v_torch = v.clone()
    u_torch = u.clone()
    g_cmp_torch = g_cmp.clone()
    g_slc_torch = g_slc.clone()
    
    try:
        o_torch, block_indices_torch = block_sparse_attn(
            q=q_torch,
            k=k_torch,
            v=v_torch,
            u=u_torch,
            g_cmp=g_cmp_torch,
            g_slc=g_slc_torch,
            g_swa=g_swa,
            block_counts=block_counts,
            block_size=block_size,
            window_size=0,
        )
        print(f"✓ PyTorch 实现完成")
        print(f"  - o_torch shape: {o_torch.shape}")
        print(f"  - block_indices_torch shape: {block_indices_torch.shape}")
    except Exception as e:
        print(f"✗ PyTorch 实现失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Triton 实现
    print("\n运行 Triton 实现...")
    set_seed(42)
    q_triton = q.clone()
    k_triton = k.clone()
    v_triton = v.clone()
    u_triton = u.clone()
    g_cmp_triton = g_cmp.clone()
    g_slc_triton = g_slc.clone()
    
    try:
        o_triton, block_indices_triton = triton_block_sparse_attn(
            q=q_triton,
            k=k_triton,
            v=v_triton,
            u=u_triton,
            g_cmp=g_cmp_triton,
            g_slc=g_slc_triton,
            g_swa=g_swa,
            block_counts=block_counts,
            block_size=block_size,
            window_size=0,
        )
        print(f"✓ Triton 实现完成")
        print(f"  - o_triton shape: {o_triton.shape}")
        print(f"  - block_indices_triton shape: {block_indices_triton.shape}")
    except Exception as e:
        print(f"✗ Triton 实现失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 比较结果
    print("\n比较结果...")
    o_match, o_max_diff, o_mean_diff, o_stats = compare_tensors(
        o_torch, o_triton, "输出 o", rtol=1e-3, atol=1e-4
    )
    
    # block_indices 可能不完全相同（因为 topk 可能有平局），但应该相似
    block_match, block_max_diff, block_mean_diff, block_stats = compare_tensors(
        block_indices_torch.float(), block_indices_triton.float(), 
        "block_indices", rtol=1e-2, atol=1.0
    )
    
    success = o_match
    print(f"\n{'='*80}")
    if success:
        print("✓ block_sparse_attn 测试通过！")
    else:
        print("✗ block_sparse_attn 测试失败！")
    print(f"{'='*80}\n")
    
    return success


def test_hstu_attention_with_bsa():
    """测试 _hstu_attention_with_bsa 函数"""
    print("=" * 80)
    print("测试 _hstu_attention_with_bsa")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置随机种子
    set_seed(42)
    
    # 创建测试数据
    B = 2
    N = 128
    num_heads = 8
    attention_dim = 32
    linear_dim = 64
    
    total_tokens = B * N
    
    q = torch.randn(total_tokens, num_heads * attention_dim, device=device)
    k = torch.randn(total_tokens, num_heads * attention_dim, device=device)
    v = torch.randn(total_tokens, num_heads * linear_dim, device=device)
    u = torch.randn(total_tokens, num_heads * linear_dim, device=device)
    
    x_offsets = torch.tensor([0, N, 2 * N], dtype=torch.long, device=device)
    invalid_attn_mask = torch.zeros(B, N, N, device=device)
    
    # 创建 gate model
    gate_model = GateModel(in_features=attention_dim, hidden_dim=64).to(device)
    
    # PyTorch 实现
    print("\n运行 PyTorch 实现...")
    set_seed(42)
    q_torch = q.clone()
    k_torch = k.clone()
    v_torch = v.clone()
    u_torch = u.clone()
    gate_model_torch = GateModel(in_features=attention_dim, hidden_dim=64).to(device)
    gate_model_torch.load_state_dict(gate_model.state_dict())
    
    try:
        attn_output_torch, padded_q_torch, padded_k_torch = _hstu_attention_with_bsa(
            num_heads=num_heads,
            attention_dim=attention_dim,
            linear_dim=linear_dim,
            q=q_torch,
            k=k_torch,
            v=v_torch,
            u=u_torch,
            x_offsets=x_offsets,
            invalid_attn_mask=invalid_attn_mask,
            gate_model=gate_model_torch,
        )
        print(f"✓ PyTorch 实现完成")
        print(f"  - attn_output_torch shape: {attn_output_torch.shape}")
        print(f"  - padded_q_torch shape: {padded_q_torch.shape}")
        print(f"  - padded_k_torch shape: {padded_k_torch.shape}")
    except Exception as e:
        print(f"✗ PyTorch 实现失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Triton 实现
    print("\n运行 Triton 实现...")
    set_seed(42)
    q_triton = q.clone()
    k_triton = k.clone()
    v_triton = v.clone()
    u_triton = u.clone()
    gate_model_triton = GateModel(in_features=attention_dim, hidden_dim=64).to(device)
    gate_model_triton.load_state_dict(gate_model.state_dict())
    
    try:
        attn_output_triton, padded_q_triton, padded_k_triton = triton_hstu_attention_with_bsa(
            num_heads=num_heads,
            attention_dim=attention_dim,
            linear_dim=linear_dim,
            q=q_triton,
            k=k_triton,
            v=v_triton,
            u=u_triton,
            x_offsets=x_offsets,
            invalid_attn_mask=invalid_attn_mask,
            gate_model=gate_model_triton,
            block_counts=4,
            block_size=32,
            window_size=0,
        )
        print(f"✓ Triton 实现完成")
        print(f"  - attn_output_triton shape: {attn_output_triton.shape}")
        print(f"  - padded_q_triton shape: {padded_q_triton.shape}")
        print(f"  - padded_k_triton shape: {padded_k_triton.shape}")
    except Exception as e:
        print(f"✗ Triton 实现失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 比较结果
    print("\n比较结果...")
    attn_match, attn_max_diff, attn_mean_diff, attn_stats = compare_tensors(
        attn_output_torch, attn_output_triton, "attn_output", rtol=1e-3, atol=1e-4
    )
    
    padded_q_match, _, _, _ = compare_tensors(
        padded_q_torch, padded_q_triton, "padded_q", rtol=1e-3, atol=1e-4
    )
    
    padded_k_match, _, _, _ = compare_tensors(
        padded_k_torch, padded_k_triton, "padded_k", rtol=1e-3, atol=1e-4
    )
    
    success = attn_match and padded_q_match and padded_k_match
    print(f"\n{'='*80}")
    if success:
        print("✓ _hstu_attention_with_bsa 测试通过！")
    else:
        print("✗ _hstu_attention_with_bsa 测试失败！")
    print(f"{'='*80}\n")
    
    return success


def test_multiple_configurations():
    """测试多个配置"""
    print("=" * 80)
    print("测试多个配置")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    configs = [
        {"B": 1, "T": 64, "H": 4, "ATTN_DIM": 16, "LINEAR_DIM": 32, "block_size": 16},
        {"B": 2, "T": 128, "H": 8, "ATTN_DIM": 32, "LINEAR_DIM": 64, "block_size": 32},
        {"B": 4, "T": 256, "H": 8, "ATTN_DIM": 32, "LINEAR_DIM": 64, "block_size": 32},
    ]
    
    all_passed = True
    
    for i, config in enumerate(configs):
        print(f"\n配置 {i+1}/{len(configs)}: {config}")
        print("-" * 80)
        
        set_seed(42 + i)
        
        B = config["B"]
        T = config["T"]
        H = config["H"]
        ATTN_DIM = config["ATTN_DIM"]
        LINEAR_DIM = config["LINEAR_DIM"]
        block_size = config["block_size"]
        block_counts = 4
        
        q = torch.randn(B, T, H, ATTN_DIM, device=device)
        k = torch.randn(B, T, H, ATTN_DIM, device=device)
        v = torch.randn(B, T, H, LINEAR_DIM, device=device)
        u = torch.randn(B, T, H, LINEAR_DIM, device=device)
        g_cmp = torch.randn(B, T, H, device=device)
        g_slc = torch.randn(B, T, H, device=device)
        
        # PyTorch
        set_seed(42 + i)
        try:
            o_torch, _ = block_sparse_attn(
                q=q.clone(), k=k.clone(), v=v.clone(), u=u.clone(),
                g_cmp=g_cmp.clone(), g_slc=g_slc.clone(), g_swa=None,
                block_counts=block_counts, block_size=block_size, window_size=0,
            )
        except Exception as e:
            print(f"✗ PyTorch 失败: {e}")
            all_passed = False
            continue
        
        # Triton
        set_seed(42 + i)
        try:
            o_triton, _ = triton_block_sparse_attn(
                q=q.clone(), k=k.clone(), v=v.clone(), u=u.clone(),
                g_cmp=g_cmp.clone(), g_slc=g_slc.clone(), g_swa=None,
                block_counts=block_counts, block_size=block_size, window_size=0,
            )
        except Exception as e:
            print(f"✗ Triton 失败: {e}")
            all_passed = False
            continue
        
        # 比较
        match, max_diff, mean_diff, stats = compare_tensors(
            o_torch, o_triton, f"配置 {i+1} 输出", rtol=1e-3, atol=1e-4
        )
        
        if not match:
            all_passed = False
    
    print(f"\n{'='*80}")
    if all_passed:
        print("✓ 所有配置测试通过！")
    else:
        print("✗ 部分配置测试失败！")
    print(f"{'='*80}\n")
    
    return all_passed


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("Triton vs PyTorch 实现对比测试")
    print("=" * 80 + "\n")
    
    if not torch.cuda.is_available():
        print("警告: CUDA 不可用，某些功能可能无法正常工作")
        print("建议在支持 CUDA 的环境中运行此测试\n")
    
    results = []
    
    # 测试 1: block_sparse_attn
    try:
        result1 = test_block_sparse_attn()
        results.append(("block_sparse_attn", result1))
    except Exception as e:
        print(f"测试 block_sparse_attn 时发生错误: {e}")
        import traceback
        traceback.print_exc()
        results.append(("block_sparse_attn", False))
    
    # 测试 2: _hstu_attention_with_bsa
    try:
        result2 = test_hstu_attention_with_bsa()
        results.append(("_hstu_attention_with_bsa", result2))
    except Exception as e:
        print(f"测试 _hstu_attention_with_bsa 时发生错误: {e}")
        import traceback
        traceback.print_exc()
        results.append(("_hstu_attention_with_bsa", False))
    
    # 测试 3: 多个配置
    try:
        result3 = test_multiple_configurations()
        results.append(("multiple_configurations", result3))
    except Exception as e:
        print(f"测试多个配置时发生错误: {e}")
        import traceback
        traceback.print_exc()
        results.append(("multiple_configurations", False))
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    for test_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    print("=" * 80)
    if all_passed:
        print("✓ 所有测试通过！")
        return 0
    else:
        print("✗ 部分测试失败！")
        return 1


if __name__ == "__main__":
    exit(main())

