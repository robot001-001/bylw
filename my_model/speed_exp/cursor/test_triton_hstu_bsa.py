# -*- coding: utf-8 -*-
"""
Test script for Triton HSTU BSA implementation.
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.GateModel import GateModel
from speed_exp.triton_bsa.triton_hstu_bsa import triton_hstu_attention_with_bsa


def test_triton_hstu_bsa():
    """Test Triton HSTU BSA implementation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Parameters
    B = 2  # batch size
    N = 128  # sequence length
    num_heads = 8
    attention_dim = 32
    linear_dim = 64
    embedding_dim = 128
    
    # Create dummy data
    total_tokens = B * N
    q = torch.randn(total_tokens, num_heads * attention_dim, device=device)
    k = torch.randn(total_tokens, num_heads * attention_dim, device=device)
    v = torch.randn(total_tokens, num_heads * linear_dim, device=device)
    u = torch.randn(total_tokens, num_heads * linear_dim, device=device)
    
    # Create offsets
    x_offsets = torch.tensor([0, N, 2 * N], dtype=torch.long, device=device)
    
    # Create attention mask
    invalid_attn_mask = torch.zeros(B, N, N, device=device)
    
    # Create gate model
    gate_model = GateModel(in_features=attention_dim, hidden_dim=64).to(device)
    
    # Test Triton implementation
    print("Testing Triton HSTU BSA...")
    try:
        attn_output, padded_q, padded_k = triton_hstu_attention_with_bsa(
            num_heads=num_heads,
            attention_dim=attention_dim,
            linear_dim=linear_dim,
            q=q,
            k=k,
            v=v,
            u=u,
            x_offsets=x_offsets,
            invalid_attn_mask=invalid_attn_mask,
            gate_model=gate_model,
            block_counts=4,
            block_size=32,
            window_size=0,
        )
        
        print(f"✓ Triton implementation successful!")
        print(f"  Output shape: {attn_output.shape}")
        print(f"  Padded Q shape: {padded_q.shape}")
        print(f"  Padded K shape: {padded_k.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Triton implementation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_triton_vs_pytorch():
    """Benchmark Triton vs PyTorch implementation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return
    
    # Parameters
    B = 4
    N = 256
    num_heads = 8
    attention_dim = 32
    linear_dim = 64
    
    # Create dummy data
    total_tokens = B * N
    q = torch.randn(total_tokens, num_heads * attention_dim, device=device)
    k = torch.randn(total_tokens, num_heads * attention_dim, device=device)
    v = torch.randn(total_tokens, num_heads * linear_dim, device=device)
    u = torch.randn(total_tokens, num_heads * linear_dim, device=device)
    
    x_offsets = torch.tensor([0, N, 2 * N, 3 * N, 4 * N], dtype=torch.long, device=device)
    invalid_attn_mask = torch.zeros(B, N, N, device=device)
    gate_model = GateModel(in_features=attention_dim, hidden_dim=64).to(device)
    
    # Warmup
    for _ in range(10):
        _ = triton_hstu_attention_with_bsa(
            num_heads=num_heads,
            attention_dim=attention_dim,
            linear_dim=linear_dim,
            q=q,
            k=k,
            v=v,
            u=u,
            x_offsets=x_offsets,
            invalid_attn_mask=invalid_attn_mask,
            gate_model=gate_model,
            block_counts=4,
            block_size=32,
        )
    
    # Benchmark Triton
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        _ = triton_hstu_attention_with_bsa(
            num_heads=num_heads,
            attention_dim=attention_dim,
            linear_dim=linear_dim,
            q=q,
            k=k,
            v=v,
            u=u,
            x_offsets=x_offsets,
            invalid_attn_mask=invalid_attn_mask,
            gate_model=gate_model,
            block_counts=4,
            block_size=32,
        )
    end.record()
    torch.cuda.synchronize()
    
    triton_time = start.elapsed_time(end) / 100
    
    print(f"Triton implementation: {triton_time:.3f} ms per iteration")


if __name__ == "__main__":
    print("=" * 60)
    print("Triton HSTU BSA Test")
    print("=" * 60)
    
    success = test_triton_hstu_bsa()
    
    if success and torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("Benchmarking...")
        print("=" * 60)
        benchmark_triton_vs_pytorch()

