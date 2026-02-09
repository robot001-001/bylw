# -*- coding: utf-8 -*-
"""
速度对比脚本：比较原版 HSTU Triton 实现与魔改 HSTU (BSA) Triton 实现的性能。
"""

import torch
import torch.nn as nn
import time
import sys
import os
import json
from typing import Dict, List, Tuple, Optional
import numpy as np

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../baseline/hstu'))

try:
    from generative_recommenders.common import HammerKernel
    from generative_recommenders.ops.hstu_attention import hstu_mha as baseline_hstu_mha
    BASELINE_AVAILABLE = True
except ImportError:
    print("警告: 无法导入 baseline HSTU，将跳过原版 HSTU 测试")
    BASELINE_AVAILABLE = False

from model.GateModel import GateModel
from speed_exp.triton_bsa.triton_hstu_bsa import triton_hstu_attention_with_bsa


class BenchmarkResult:
    """基准测试结果"""
    def __init__(self, name: str):
        self.name = name
        self.times: List[float] = []
        self.memory_allocated: List[float] = []
        self.memory_reserved: List[float] = []
        
    def add_timing(self, time_ms: float, mem_allocated: float = 0, mem_reserved: float = 0):
        self.times.append(time_ms)
        if mem_allocated > 0:
            self.memory_allocated.append(mem_allocated)
        if mem_reserved > 0:
            self.memory_reserved.append(mem_reserved)
    
    def get_stats(self) -> Dict:
        if not self.times:
            return {}
        
        times_array = np.array(self.times)
        return {
            "mean_ms": float(np.mean(times_array)),
            "std_ms": float(np.std(times_array)),
            "min_ms": float(np.min(times_array)),
            "max_ms": float(np.max(times_array)),
            "median_ms": float(np.median(times_array)),
            "p95_ms": float(np.percentile(times_array, 95)),
            "p99_ms": float(np.percentile(times_array, 99)),
            "mean_mem_allocated_mb": float(np.mean(self.memory_allocated)) if self.memory_allocated else 0,
            "mean_mem_reserved_mb": float(np.mean(self.memory_reserved)) if self.memory_reserved else 0,
        }
    
    def __str__(self):
        stats = self.get_stats()
        if not stats:
            return f"{self.name}: 无数据"
        
        return (
            f"{self.name}:\n"
            f"  平均时间: {stats['mean_ms']:.3f} ms (±{stats['std_ms']:.3f})\n"
            f"  中位数: {stats['median_ms']:.3f} ms\n"
            f"  最小/最大: {stats['min_ms']:.3f} / {stats['max_ms']:.3f} ms\n"
            f"  P95/P99: {stats['p95_ms']:.3f} / {stats['p99_ms']:.3f} ms"
        )


def benchmark_function(
    func,
    args: tuple,
    kwargs: dict,
    num_warmup: int = 10,
    num_iterations: int = 100,
    measure_memory: bool = True,
) -> BenchmarkResult:
    """
    基准测试函数
    
    Args:
        func: 要测试的函数
        args: 位置参数
        kwargs: 关键字参数
        num_warmup: 预热迭代次数
        num_iterations: 测试迭代次数
        measure_memory: 是否测量内存使用
    
    Returns:
        BenchmarkResult: 测试结果
    """
    result = BenchmarkResult(func.__name__ if hasattr(func, '__name__') else 'unknown')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 预热
    for _ in range(num_warmup):
        try:
            _ = func(*args, **kwargs)
        except Exception as e:
            print(f"预热失败: {e}")
            return result
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 测试迭代
    for _ in range(num_iterations):
        if measure_memory and device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        start_event = None
        end_event = None
        
        if device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.perf_counter()
        
        try:
            _ = func(*args, **kwargs)
        except Exception as e:
            print(f"迭代失败: {e}")
            continue
        
        if device.type == 'cuda':
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            
            if measure_memory:
                mem_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
                mem_reserved = torch.cuda.max_memory_reserved() / 1024**2  # MB
            else:
                mem_allocated = 0
                mem_reserved = 0
        else:
            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time) * 1000
            mem_allocated = 0
            mem_reserved = 0
        
        result.add_timing(elapsed_ms, mem_allocated, mem_reserved)
    
    return result


def benchmark_baseline_hstu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    max_seq_len: int,
    alpha: float,
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> Optional[BenchmarkResult]:
    """基准测试原版 HSTU Triton 实现"""
    if not BASELINE_AVAILABLE:
        return None
    
    def run_baseline():
        return baseline_hstu_mha(
            max_seq_len=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            kernel=HammerKernel.TRITON,
            enable_tma=False,
        )
    
    return benchmark_function(
        run_baseline,
        (),
        {},
        num_warmup=num_warmup,
        num_iterations=num_iterations,
    )


def benchmark_bsa_hstu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    u: torch.Tensor,
    x_offsets: torch.Tensor,
    invalid_attn_mask: torch.Tensor,
    gate_model: nn.Module,
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    block_counts: int = 4,
    block_size: int = 32,
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> BenchmarkResult:
    """基准测试魔改 HSTU (BSA) Triton 实现"""
    
    def run_bsa():
        return triton_hstu_attention_with_bsa(
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
            block_counts=block_counts,
            block_size=block_size,
            window_size=0,
        )
    
    return benchmark_function(
        run_bsa,
        (),
        {},
        num_warmup=num_warmup,
        num_iterations=num_iterations,
    )


def run_single_benchmark(
    config: Dict,
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> Dict:
    """
    运行单个配置的基准测试
    
    Args:
        config: 配置字典
        num_warmup: 预热次数
        num_iterations: 测试次数
    
    Returns:
        包含测试结果的字典
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    B = config["B"]
    N = config["N"]
    num_heads = config["num_heads"]
    attention_dim = config["attention_dim"]
    linear_dim = config["linear_dim"]
    block_size = config.get("block_size", 32)
    block_counts = config.get("block_counts", 4)
    
    total_tokens = B * N
    
    # 创建输入数据
    torch.manual_seed(42)
    q = torch.randn(total_tokens, num_heads * attention_dim, device=device)
    k = torch.randn(total_tokens, num_heads * attention_dim, device=device)
    v = torch.randn(total_tokens, num_heads * linear_dim, device=device)
    u = torch.randn(total_tokens, num_heads * linear_dim, device=device)
    
    x_offsets = torch.arange(0, total_tokens + 1, N, dtype=torch.long, device=device)
    seq_offsets = x_offsets.clone()
    invalid_attn_mask = torch.zeros(B, N, N, device=device)
    
    # 创建 gate model
    gate_model = GateModel(in_features=attention_dim, hidden_dim=64).to(device)
    
    # 计算 alpha
    alpha = 1.0 / (attention_dim ** 0.5)
    max_seq_len = N
    
    results = {
        "config": config,
        "baseline": None,
        "bsa": None,
    }
    
    # 测试原版 HSTU
    if BASELINE_AVAILABLE:
        print(f"  测试原版 HSTU Triton...")
        try:
            # 转换为原版 HSTU 需要的格式 [L, H, D]
            q_baseline = q.view(total_tokens, num_heads, attention_dim)
            k_baseline = k.view(total_tokens, num_heads, attention_dim)
            v_baseline = v.view(total_tokens, num_heads, linear_dim)
            
            baseline_result = benchmark_baseline_hstu(
                q=q_baseline,
                k=k_baseline,
                v=v_baseline,
                seq_offsets=seq_offsets,
                max_seq_len=max_seq_len,
                alpha=alpha,
                num_warmup=num_warmup,
                num_iterations=num_iterations,
            )
            results["baseline"] = baseline_result.get_stats() if baseline_result else None
        except Exception as e:
            print(f"    原版 HSTU 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 测试魔改 HSTU (BSA)
    print(f"  测试魔改 HSTU (BSA) Triton...")
    try:
        bsa_result = benchmark_bsa_hstu(
            q=q,
            k=k,
            v=v,
            u=u,
            x_offsets=x_offsets,
            invalid_attn_mask=invalid_attn_mask,
            gate_model=gate_model,
            num_heads=num_heads,
            attention_dim=attention_dim,
            linear_dim=linear_dim,
            block_counts=block_counts,
            block_size=block_size,
            num_warmup=num_warmup,
            num_iterations=num_iterations,
        )
        results["bsa"] = bsa_result.get_stats()
    except Exception as e:
        print(f"    魔改 HSTU (BSA) 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 计算加速比
    if results["baseline"] and results["bsa"]:
        baseline_mean = results["baseline"]["mean_ms"]
        bsa_mean = results["bsa"]["mean_ms"]
        speedup = baseline_mean / bsa_mean if bsa_mean > 0 else 0
        results["speedup"] = speedup
        results["baseline_slower_by"] = (bsa_mean / baseline_mean - 1) * 100 if baseline_mean > 0 else 0
    
    return results


def print_results(results: Dict):
    """打印测试结果"""
    config = results["config"]
    print(f"\n配置: B={config['B']}, N={config['N']}, H={config['num_heads']}, "
          f"ATTN_DIM={config['attention_dim']}, LINEAR_DIM={config['linear_dim']}")
    print("-" * 80)
    
    if results["baseline"]:
        baseline = results["baseline"]
        print(f"原版 HSTU Triton:")
        print(f"  平均时间: {baseline['mean_ms']:.3f} ms (±{baseline['std_ms']:.3f})")
        print(f"  中位数: {baseline['median_ms']:.3f} ms")
        print(f"  最小/最大: {baseline['min_ms']:.3f} / {baseline['max_ms']:.3f} ms")
        if baseline['mean_mem_allocated_mb'] > 0:
            print(f"  平均内存分配: {baseline['mean_mem_allocated_mb']:.2f} MB")
    
    if results["bsa"]:
        bsa = results["bsa"]
        print(f"魔改 HSTU (BSA) Triton:")
        print(f"  平均时间: {bsa['mean_ms']:.3f} ms (±{bsa['std_ms']:.3f})")
        print(f"  中位数: {bsa['median_ms']:.3f} ms")
        print(f"  最小/最大: {bsa['min_ms']:.3f} / {bsa['max_ms']:.3f} ms")
        if bsa['mean_mem_allocated_mb'] > 0:
            print(f"  平均内存分配: {bsa['mean_mem_allocated_mb']:.2f} MB")
    
    if "speedup" in results:
        speedup = results["speedup"]
        if speedup > 1:
            print(f"\n✓ 魔改 HSTU (BSA) 更快: {speedup:.2f}x 加速")
        elif speedup < 1:
            slower_by = results.get("baseline_slower_by", 0)
            print(f"\n✗ 魔改 HSTU (BSA) 更慢: {1/speedup:.2f}x (慢 {slower_by:.1f}%)")
        else:
            print(f"\n= 性能相当")


def run_all_benchmarks(
    num_warmup: int = 10,
    num_iterations: int = 100,
    save_results: bool = True,
) -> List[Dict]:
    """运行所有配置的基准测试"""
    
    configs = [
        {"B": 1, "N": 64, "num_heads": 4, "attention_dim": 16, "linear_dim": 32, "block_size": 16, "block_counts": 2},
        {"B": 2, "N": 128, "num_heads": 8, "attention_dim": 32, "linear_dim": 64, "block_size": 32, "block_counts": 4},
        {"B": 4, "N": 256, "num_heads": 8, "attention_dim": 32, "linear_dim": 64, "block_size": 32, "block_counts": 4},
        {"B": 8, "N": 512, "num_heads": 8, "attention_dim": 32, "linear_dim": 64, "block_size": 32, "block_counts": 4},
        {"B": 16, "N": 1024, "num_heads": 8, "attention_dim": 32, "linear_dim": 64, "block_size": 32, "block_counts": 4},
    ]
    
    all_results = []
    
    print("=" * 80)
    print("HSTU Triton 实现速度对比")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("警告: CUDA 不可用，性能测试可能不准确")
        print("建议在支持 CUDA 的环境中运行此测试\n")
    
    device_info = ""
    if torch.cuda.is_available():
        device_info = f"GPU: {torch.cuda.get_device_name(0)}"
        print(f"设备信息: {device_info}\n")
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] 测试配置 {i+1}...")
        try:
            result = run_single_benchmark(
                config,
                num_warmup=num_warmup,
                num_iterations=num_iterations,
            )
            print_results(result)
            all_results.append(result)
        except Exception as e:
            print(f"配置 {i+1} 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存结果
    if save_results:
        results_file = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "device_info": device_info,
                "num_warmup": num_warmup,
                "num_iterations": num_iterations,
                "results": all_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {results_file}")
    
    # 打印总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    
    if all_results:
        speedups = [r["speedup"] for r in all_results if "speedup" in r]
        if speedups:
            avg_speedup = np.mean(speedups)
            print(f"平均加速比: {avg_speedup:.2f}x")
            print(f"最大加速比: {max(speedups):.2f}x")
            print(f"最小加速比: {min(speedups):.2f}x")
    
    return all_results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HSTU Triton 实现速度对比")
    parser.add_argument("--warmup", type=int, default=10, help="预热迭代次数")
    parser.add_argument("--iterations", type=int, default=100, help="测试迭代次数")
    parser.add_argument("--no-save", action="store_true", help="不保存结果到文件")
    
    args = parser.parse_args()
    
    run_all_benchmarks(
        num_warmup=args.warmup,
        num_iterations=args.iterations,
        save_results=not args.no_save,
    )


if __name__ == "__main__":
    main()

