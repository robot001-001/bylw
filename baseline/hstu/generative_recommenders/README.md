# HSTU Triton 加速实现

本文档介绍了在 `bylw/baseline/hstu` 目录下使用 Triton 加速 HSTU 实现的代码结构和主要功能。

## 目录结构

主要的 Triton 实现代码位于 `generative_recommenders/ops/triton/` 目录下：

```
generative_recommenders/ops/triton/
├── triton_hstu_attention.py              # HSTU Attention 的 Triton 实现
├── triton_hstu_preprocess_and_attention.py # 预处理和 Attention 的融合实现
├── triton_addmm.py                        # 矩阵乘法的 Triton 实现
├── triton_layer_norm.py                   # Layer Norm 的 Triton 实现
├── triton_position.py                     # 位置编码的 Triton 实现
├── triton_jagged_tensors.py               # Jagged Tensor 操作的 Triton 实现
└── triton_attention_utils.py               # Attention 工具函数
```

## 核心实现文件

### 1. triton_hstu_attention.py

**位置**: `generative_recommenders/ops/triton/triton_hstu_attention.py`

**主要功能**:
- 实现了 HSTU Attention 的前向和反向传播
- 使用 `@triton.jit` 装饰器编写 GPU kernel
- 支持 TMA (Tensor Memory Accelerator) 优化
- 支持 TLX (Triton Language eXtensions) 用于 H100 等新架构

**关键函数**:

- `triton_hstu_attention_fwd()`: 前向传播实现
- `triton_hstu_attention_bwd()`: 反向传播实现
- `triton_hstu_mha()`: 主要的 Multi-Head Attention 接口
- `triton_cached_hstu_mha()`: 缓存版本的 HSTU MHA（用于增量更新）

**核心 Kernel**:

- `_hstu_attn_fwd_one_block()`: 单个 attention block 的前向计算
- `_hstu_attn_fwd_compute()`: 前向计算主循环
- `_hstu_attn_fwd_compute_tlx()`: 使用 TLX 的前向计算（H100 优化）
- `_hstu_attn_bwd_one_block()`: 单个 attention block 的反向计算
- `_hstu_attn_bwd_one_col_block()`: 按列的反向计算

**特性**:
- 支持因果注意力（Causal Attention）
- 支持多目标（Multiple Targets）
- 支持上下文序列长度（Contextual Sequence Length）
- 支持最大注意力长度限制（Max Attention Length）
- 支持按长度排序优化（Sort by Length）
- 自动调优（Autotune）支持多种配置

### 2. triton_hstu_preprocess_and_attention.py

**位置**: `generative_recommenders/ops/triton/triton_hstu_preprocess_and_attention.py`

**主要功能**:
- 融合了 Layer Norm、矩阵乘法和 Attention 操作
- 减少内存访问，提高计算效率
- 完整的前向和反向传播实现

**关键函数**:

- `triton_hstu_preprocess_and_attention()`: 融合的预处理和注意力计算
- `_HSTUPreprocessAndAttentionFunction`: PyTorch autograd Function 实现

**计算流程**:
1. Layer Norm: `triton_weighted_layer_norm_fwd()`
2. 矩阵乘法: `maybe_triton_addmm_fwd()` (生成 U, V, Q, K)
3. Attention: `triton_hstu_attention_fwd()`
4. SiLU 激活: `F.silu(u)`

### 3. 辅助实现文件

#### triton_addmm.py
- 实现了矩阵乘法的 Triton kernel
- 支持 TMA 和 TLX 优化
- 提供前向和反向传播

#### triton_layer_norm.py
- 实现了加权 Layer Norm 的 Triton kernel
- 支持前向和反向传播

#### triton_position.py
- 实现了位置编码的 Triton kernel
- 支持时间戳位置嵌入

#### triton_jagged_tensors.py
- 实现了 Jagged Tensor 操作的 Triton kernel
- 支持不规则张量的高效操作

## 调用接口

### hstu_attention.py

**位置**: `generative_recommenders/ops/hstu_attention.py`

**主要函数**:

- `hstu_mha()`: 统一的 HSTU Multi-Head Attention 接口
- `delta_hstu_mha()`: 增量更新的 HSTU MHA

**使用方式**:

```python
from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.hstu_attention import hstu_mha

# 使用 Triton 实现
output = hstu_mha(
    max_seq_len=max_seq_len,
    alpha=alpha,
    q=q,
    k=k,
    v=v,
    seq_offsets=seq_offsets,
    kernel=HammerKernel.TRITON,  # 选择 Triton kernel
    enable_tma=True,  # 启用 TMA 优化
)
```

**Kernel 选择**:
- `HammerKernel.PYTORCH`: PyTorch 原生实现
- `HammerKernel.TRITON`: Triton 实现
- `HammerKernel.TRITON_CC`: Triton C++ 编译版本

## 性能优化特性

### 1. TMA (Tensor Memory Accelerator)
- 使用 Tensor Descriptor 进行高效的内存访问
- 适用于 H100 等新架构 GPU
- 通过 `enable_tma=True` 启用

### 2. TLX (Triton Language eXtensions)
- 使用异步加载和计算流水线
- 支持多缓冲区 ping-pong
- 针对 H100 (Compute Capability 9.0) 优化

### 3. 自动调优 (Autotune)
- 自动选择最优的 block 大小和 warp 数量
- 根据输入维度、序列长度等参数自动配置
- 支持多种硬件平台（NVIDIA、AMD）

### 4. 内存优化
- 支持按长度排序以减少 padding 浪费
- 支持增量更新（Delta Attention）
- 支持上下文序列长度限制

## 测试文件

- `generative_recommenders/ops/tests/hstu_attention_test.py`: Triton 实现的测试用例
- `generative_recommenders/ops/tests/hstu_attention_tma_test.py`: TMA 优化的测试用例
- `generative_recommenders/ops/tests/hstu_compute_test.py`: HSTU 计算的测试用例

## 使用示例

### 基本使用

```python
import torch
from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.hstu_attention import hstu_mha

# 准备输入
batch_size = 32
num_heads = 8
seq_len = 128
hidden_dim = 64

q = torch.randn(seq_len, num_heads, hidden_dim, device='cuda')
k = torch.randn(seq_len, num_heads, hidden_dim, device='cuda')
v = torch.randn(seq_len, num_heads, hidden_dim, device='cuda')
seq_offsets = torch.tensor([0, seq_len], dtype=torch.long, device='cuda')

# 使用 Triton 加速
output = hstu_mha(
    max_seq_len=seq_len,
    alpha=1.0 / (hidden_dim ** 0.5),
    q=q,
    k=k,
    v=v,
    seq_offsets=seq_offsets,
    kernel=HammerKernel.TRITON,
    enable_tma=True,
)
```

### 使用融合的预处理和注意力

```python
from generative_recommenders.ops.triton.triton_hstu_preprocess_and_attention import (
    triton_hstu_preprocess_and_attention
)

# 融合 Layer Norm + MatMul + Attention
silu_u, out = triton_hstu_preprocess_and_attention(
    x=x,
    norm_weight=norm_weight,
    norm_bias=norm_bias,
    norm_eps=1e-5,
    num_heads=num_heads,
    attn_dim=attn_dim,
    hidden_dim=hidden_dim,
    uvqk_weight=uvqk_weight,
    uvqk_bias=uvqk_bias,
    max_seq_len=max_seq_len,
    seq_offsets=seq_offsets,
    attn_alpha=attn_alpha,
    enable_tma=True,
)
```

## 注意事项

1. **CUDA 要求**: Triton 实现需要 CUDA 支持，确保所有张量在 CUDA 设备上
2. **TMA 支持**: TMA 优化需要 Triton 3.2.0+ 和兼容的 GPU（如 H100）
3. **TLX 支持**: TLX 优化需要 Triton 的额外扩展，主要用于 H100
4. **内存对齐**: TMA 需要特定的内存对齐（128 字节）
5. **数据类型**: 主要支持 float16 和 bfloat16

## 相关文件

- `generative_recommenders/ops/hstu_compute.py`: HSTU 计算的主要接口
- `generative_recommenders/modules/hstu_transducer.py`: HSTU Transducer 模块
- `generative_recommenders/modules/dlrm_hstu.py`: DLRM HSTU 模块

## 参考资料

- [Triton 官方文档](https://triton-lang.org/)
- Meta HSTU 开源实现

