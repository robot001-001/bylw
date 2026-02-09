# HSTU BSA Triton 加速实现

本目录包含使用 Triton 加速的 HSTU (Hierarchical Sequential Transduction Unit) 与 Block Sparse Attention (BSA) 的实现。

## 目录结构

```
speed_exp/
├── triton/
│   ├── __init__.py                    # 模块初始化
│   ├── triton_bsa_attention.py        # BSA Attention 的 Triton 实现
│   └── triton_hstu_bsa.py             # HSTU BSA 的完整 Triton 实现
├── test_triton_hstu_bsa.py            # 测试脚本
└── README.md                          # 本文档
```

## 实现概述

### 1. triton_bsa_attention.py

实现了 Block Sparse Attention 的核心 Triton kernels：

- **`_bsa_compression_kernel`**: 压缩注意力计算 kernel
  - 计算压缩的 k_cmp 和 v_cmp（通过 mean pooling）
  - 计算压缩注意力分数
  - 使用 SiLU 激活函数

- **`_bsa_selection_kernel`**: 选择注意力计算 kernel
  - 根据 block_indices 选择特定的 blocks
  - 计算选择注意力输出
  - 应用门控和 u 向量

- **`triton_block_sparse_attn`**: 完整的 BSA 前向传播
  - 结合压缩和选择两个阶段
  - 支持 interleave 模式

### 2. triton_hstu_bsa.py

实现了完整的 HSTU BSA 模块：

- **`TritonHSTUBSAAttention`**: 完整的 HSTU BSA 注意力模块
  - Layer Normalization（Triton 加速）
  - UVQK 投影（Triton 加速）
  - Block Sparse Attention
  - 输出投影

- **`triton_hstu_attention_with_bsa`**: 函数式接口
  - 与原始 PyTorch 实现兼容的接口
  - 支持 jagged tensor 输入输出

## 关键特性

### 1. Block Sparse Attention

- **压缩阶段**: 将序列分成 blocks，对每个 block 进行 mean pooling 压缩
- **选择阶段**: 根据注意力分数选择 top-k blocks
- **Interleave 模式**: 支持交替 block 共享索引

### 2. SiLU 激活函数

使用 SiLU (Swish) 激活函数替代传统的 softmax：
```
SiLU(x) = x * sigmoid(x)
```

### 3. 门控机制

- `g_cmp`: 压缩注意力的门控分数
- `g_slc`: 选择注意力的门控分数
- `g_swa`: 滑动窗口注意力的门控分数（可选）

### 4. U 向量门控

使用 u 向量对注意力输出进行门控：
```
output = attention_output * u
```

## 使用方法

### 基本使用

```python
from speed_exp.triton import triton_hstu_attention_with_bsa
from model.GateModel import GateModel

# 准备输入
q = torch.randn(total_tokens, num_heads * attention_dim, device='cuda')
k = torch.randn(total_tokens, num_heads * attention_dim, device='cuda')
v = torch.randn(total_tokens, num_heads * linear_dim, device='cuda')
u = torch.randn(total_tokens, num_heads * linear_dim, device='cuda')
x_offsets = torch.tensor([0, N, 2*N], dtype=torch.long, device='cuda')
gate_model = GateModel(in_features=attention_dim, hidden_dim=64).cuda()

# 调用 Triton 实现
attn_output, padded_q, padded_k = triton_hstu_attention_with_bsa(
    num_heads=num_heads,
    attention_dim=attention_dim,
    linear_dim=linear_dim,
    q=q,
    k=k,
    v=v,
    u=u,
    x_offsets=x_offsets,
    invalid_attn_mask=torch.zeros(B, N, N, device='cuda'),
    gate_model=gate_model,
    block_counts=4,
    block_size=32,
    window_size=0,
)
```

### 使用完整模块

```python
from speed_exp.triton import TritonHSTUBSAAttention
from model.GateModel import GateModel

# 创建模块
module = TritonHSTUBSAAttention(
    embedding_dim=128,
    linear_hidden_dim=64,
    attention_dim=32,
    num_heads=8,
    block_counts=4,
    block_size=32,
).cuda()

gate_model = GateModel(in_features=32, hidden_dim=64).cuda()

# 前向传播
x = torch.randn(total_tokens, 128, device='cuda')
x_offsets = torch.tensor([0, N, 2*N], dtype=torch.long, device='cuda')
invalid_attn_mask = torch.zeros(B, N, N, device='cuda')

output, padded_q, padded_k = module(
    x=x,
    x_offsets=x_offsets,
    invalid_attn_mask=invalid_attn_mask,
    gate_model=gate_model,
)
```

## 测试

运行测试脚本：

```bash
cd bylw/my_model/speed_exp
python test_triton_hstu_bsa.py
```

测试脚本会：
1. 验证 Triton 实现的正确性
2. 进行性能基准测试（如果 CUDA 可用）

## 性能优化

### 1. 自动调优

Triton kernels 使用 `@triton.autotune` 装饰器自动选择最优配置：
- Block 大小（BLOCK_M, BLOCK_N）
- Warp 数量
- Pipeline stages

### 2. 内存优化

- 使用 block pointer 进行高效内存访问
- 支持 jagged tensor，减少 padding 浪费
- 融合操作减少中间结果存储

### 3. 计算优化

- 使用 Triton 的矩阵乘法指令
- 向量化操作
- 减少分支和条件判断

## 与原始实现的对比

### 原始实现 (`HSTU_bsa_pretrain_interleave.py`)

- 使用 PyTorch 的 einsum 和索引操作
- 在 CPU/GPU 上运行，但可能不是最优的
- 支持完整的梯度计算

### Triton 实现

- 使用 Triton 编写自定义 GPU kernels
- 针对特定硬件优化
- 更高的计算效率
- 需要 CUDA 支持

## 注意事项

1. **CUDA 要求**: 需要 CUDA 支持的 GPU
2. **Triton 版本**: 建议使用 Triton 2.0+
3. **内存对齐**: 某些操作可能需要特定的内存对齐
4. **数据类型**: 主要支持 float16 和 float32
5. **梯度计算**: 当前实现仅支持前向传播，反向传播需要额外实现

## 未来改进

- [ ] 实现反向传播（backward pass）
- [ ] 支持滑动窗口注意力（sliding window attention）
- [ ] 优化 block selection 算法（使用 bitonic sort）
- [ ] 支持 TMA (Tensor Memory Accelerator) 优化
- [ ] 添加更多性能基准测试

## 参考资料

- [Triton 官方文档](https://triton-lang.org/)
- HSTU 原始实现: `bylw/my_model/model/HSTU_bsa_pretrain_interleave.py`
- NSA Triton 实现: `bylw/nsa/refer/nsa_kernel.py`
- HSTU Triton 实现: `bylw/baseline/hstu/generative_recommenders/ops/triton/`

