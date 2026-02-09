# HSTU Triton 实现速度对比脚本

## 概述

`benchmark_triton_hstu.py` 是一个性能基准测试脚本，用于对比：
1. **原版 HSTU Triton 实现**（来自 `bylw/baseline/hstu`）
2. **魔改 HSTU (BSA) Triton 实现**（我们实现的 Block Sparse Attention 版本）

## 功能特性

- **执行时间测量**: 使用 CUDA Events 精确测量 GPU 执行时间
- **内存使用统计**: 记录峰值内存分配和保留
- **多配置测试**: 测试不同批次大小、序列长度等配置
- **统计分析**: 提供平均值、标准差、中位数、P95/P99 等统计信息
- **结果保存**: 将结果保存为 JSON 格式便于后续分析
- **加速比计算**: 自动计算性能提升比例

## 使用方法

### 基本使用

```bash
cd bylw/my_model/speed_exp
python benchmark_triton_hstu.py
```

### 自定义参数

```bash
# 设置预热和迭代次数
python benchmark_triton_hstu.py --warmup 20 --iterations 200

# 不保存结果文件
python benchmark_triton_hstu.py --no-save
```

### 参数说明

- `--warmup`: 预热迭代次数（默认: 10）
- `--iterations`: 测试迭代次数（默认: 100）
- `--no-save`: 不保存结果到 JSON 文件

## 测试配置

脚本会测试以下配置：

1. **小配置**: B=1, N=64, H=4, ATTN_DIM=16, LINEAR_DIM=32
2. **中等配置**: B=2, N=128, H=8, ATTN_DIM=32, LINEAR_DIM=64
3. **大配置**: B=4, N=256, H=8, ATTN_DIM=32, LINEAR_DIM=64
4. **超大配置**: B=8, N=512, H=8, ATTN_DIM=32, LINEAR_DIM=64
5. **极大配置**: B=16, N=1024, H=8, ATTN_DIM=32, LINEAR_DIM=64

每个配置都会测试：
- 原版 HSTU Triton 实现
- 魔改 HSTU (BSA) Triton 实现

## 输出说明

### 控制台输出

脚本会在控制台输出详细的性能统计：

```
配置: B=2, N=128, H=8, ATTN_DIM=32, LINEAR_DIM=64
--------------------------------------------------------------------------------
原版 HSTU Triton:
  平均时间: 12.345 ms (±0.123)
  中位数: 12.300 ms
  最小/最大: 12.100 / 12.800 ms
  平均内存分配: 45.67 MB

魔改 HSTU (BSA) Triton:
  平均时间: 8.901 ms (±0.089)
  中位数: 8.850 ms
  最小/最大: 8.700 / 9.200 ms
  平均内存分配: 38.90 MB

✓ 魔改 HSTU (BSA) 更快: 1.39x 加速
```

### JSON 结果文件

结果会保存到 `benchmark_results.json`，包含：

```json
{
  "device_info": "GPU: NVIDIA GeForce RTX 3090",
  "num_warmup": 10,
  "num_iterations": 100,
  "results": [
    {
      "config": {
        "B": 2,
        "N": 128,
        "num_heads": 8,
        "attention_dim": 32,
        "linear_dim": 64,
        "block_size": 32,
        "block_counts": 4
      },
      "baseline": {
        "mean_ms": 12.345,
        "std_ms": 0.123,
        "min_ms": 12.100,
        "max_ms": 12.800,
        "median_ms": 12.300,
        "p95_ms": 12.650,
        "p99_ms": 12.750,
        "mean_mem_allocated_mb": 45.67,
        "mean_mem_reserved_mb": 128.00
      },
      "bsa": {
        "mean_ms": 8.901,
        "std_ms": 0.089,
        "min_ms": 8.700,
        "max_ms": 9.200,
        "median_ms": 8.850,
        "p95_ms": 9.100,
        "p99_ms": 9.180,
        "mean_mem_allocated_mb": 38.90,
        "mean_mem_reserved_mb": 120.00
      },
      "speedup": 1.387,
      "baseline_slower_by": -27.9
    }
  ]
}
```

## 性能指标说明

### 时间指标

- **mean_ms**: 平均执行时间（毫秒）
- **std_ms**: 标准差
- **min_ms / max_ms**: 最小/最大执行时间
- **median_ms**: 中位数执行时间
- **p95_ms / p99_ms**: 95/99 百分位数

### 内存指标

- **mean_mem_allocated_mb**: 平均峰值内存分配（MB）
- **mean_mem_reserved_mb**: 平均峰值内存保留（MB）

### 加速比

- **speedup**: 加速倍数（baseline_time / bsa_time）
  - `> 1`: BSA 更快
  - `< 1`: BSA 更慢
  - `= 1`: 性能相当

## 注意事项

1. **CUDA 要求**: 
   - 需要 CUDA 支持的 GPU
   - Triton 实现需要 CUDA 环境

2. **预热**: 
   - 预热迭代用于"预热"GPU 和 JIT 编译
   - 预热结果不计入统计

3. **随机性**: 
   - 使用固定随机种子（42）确保可重复性
   - 但 GPU 执行时间仍可能有波动

4. **内存测量**: 
   - 内存统计可能不完全准确
   - 建议多次运行取平均值

5. **基线实现**: 
   - 如果无法导入 baseline HSTU，将跳过原版测试
   - 确保 `bylw/baseline/hstu` 路径正确

## 故障排除

### 无法导入 baseline HSTU

如果看到 "警告: 无法导入 baseline HSTU"，请检查：

1. `bylw/baseline/hstu` 目录是否存在
2. 是否正确安装了相关依赖
3. Python 路径是否正确设置

### CUDA 错误

如果遇到 CUDA 相关错误：

1. 检查 CUDA 是否可用：`torch.cuda.is_available()`
2. 检查 GPU 内存是否充足
3. 尝试减小批次大小或序列长度

### 性能异常

如果性能结果异常：

1. 增加预热次数（`--warmup 50`）
2. 增加测试迭代次数（`--iterations 500`）
3. 确保没有其他进程占用 GPU
4. 检查 GPU 温度是否过高（可能导致降频）

## 结果分析

### 如何解读加速比

- **1.2x - 1.5x**: 中等加速，值得使用
- **1.5x - 2.0x**: 显著加速，强烈推荐
- **> 2.0x**: 大幅加速，非常推荐
- **< 1.0x**: BSA 更慢，可能需要优化

### 内存使用对比

BSA 实现通常使用更少内存，因为：
- Block Sparse Attention 只处理选中的 blocks
- 减少了注意力矩阵的计算和存储

### 不同配置的表现

- **小序列**: BSA 可能没有明显优势（overhead 较大）
- **中等序列**: BSA 开始显示优势
- **长序列**: BSA 优势最明显（稀疏性带来的收益）

## 扩展

### 添加新配置

在 `run_all_benchmarks()` 函数中添加新配置：

```python
configs = [
    # ... 现有配置 ...
    {"B": 32, "N": 2048, "num_heads": 16, "attention_dim": 64, "linear_dim": 128, ...},
]
```

### 自定义测试

可以修改脚本以测试特定配置：

```python
custom_config = {
    "B": 4,
    "N": 256,
    "num_heads": 8,
    "attention_dim": 32,
    "linear_dim": 64,
    "block_size": 32,
    "block_counts": 4,
}

result = run_single_benchmark(custom_config, num_warmup=20, num_iterations=200)
print_results(result)
```

## 示例输出

```
================================================================================
HSTU Triton 实现速度对比
================================================================================
设备信息: GPU: NVIDIA GeForce RTX 3090

[1/5] 测试配置 1...
  测试原版 HSTU Triton...
  测试魔改 HSTU (BSA) Triton...

配置: B=1, N=64, H=4, ATTN_DIM=16, LINEAR_DIM=32
--------------------------------------------------------------------------------
原版 HSTU Triton:
  平均时间: 2.345 ms (±0.023)
  中位数: 2.340 ms
  最小/最大: 2.300 / 2.400 ms
  平均内存分配: 12.34 MB

魔改 HSTU (BSA) Triton:
  平均时间: 2.456 ms (±0.025)
  中位数: 2.450 ms
  最小/最大: 2.400 / 2.500 ms
  平均内存分配: 11.23 MB

✗ 魔改 HSTU (BSA) 更慢: 0.95x (慢 4.7%)

...

================================================================================
总结
================================================================================
平均加速比: 1.25x
最大加速比: 1.89x
最小加速比: 0.95x

结果已保存到: benchmark_results.json
```

## 相关文件

- `compare_triton_vs_torch.py`: 数值正确性对比脚本
- `test_triton_hstu_bsa.py`: 功能测试脚本
- `benchmark_results.json`: 保存的基准测试结果

