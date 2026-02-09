# Triton vs PyTorch 实现对比脚本

## 概述

`compare_triton_vs_torch.py` 是一个对比验证脚本，用于确保 Triton 实现与 PyTorch 实现在相同输入下产生相同的输出。

## 功能

1. **block_sparse_attn 对比测试**
   - 测试 BSA (Block Sparse Attention) 的核心函数
   - 比较输出张量和 block_indices

2. **_hstu_attention_with_bsa 对比测试**
   - 测试完整的 HSTU BSA 注意力函数
   - 比较注意力输出、padded_q 和 padded_k

3. **多配置测试**
   - 测试不同批次大小、序列长度、头数等配置
   - 确保在各种情况下都能正确工作

## 使用方法

### 基本使用

```bash
cd bylw/my_model/speed_exp
python compare_triton_vs_torch.py
```

### 输出说明

脚本会输出详细的对比结果：

- ✓ 表示测试通过（输出在容忍范围内）
- ✗ 表示测试失败（输出超出容忍范围）

对于失败的测试，会显示：
- 最大差异和平均差异
- 相对误差统计
- 超过阈值的元素数量和百分比
- 差异最大的位置和数值

## 容忍度设置

默认的数值容忍度：
- `rtol` (相对误差): 1e-3 (0.1%)
- `atol` (绝对误差): 1e-4

这些值可以根据需要调整。对于某些操作（如 block_indices），容忍度会适当放宽。

## 测试配置

脚本会测试以下配置：

1. **基础配置**
   - B=2, T=128, H=8, ATTN_DIM=32, LINEAR_DIM=64, block_size=32

2. **多配置测试**
   - 小配置: B=1, T=64, H=4, ATTN_DIM=16, LINEAR_DIM=32, block_size=16
   - 中等配置: B=2, T=128, H=8, ATTN_DIM=32, LINEAR_DIM=64, block_size=32
   - 大配置: B=4, T=256, H=8, ATTN_DIM=32, LINEAR_DIM=64, block_size=32

## 注意事项

1. **随机种子**: 脚本使用固定的随机种子（42）确保可重复性
2. **CUDA 要求**: Triton 实现需要 CUDA 支持
3. **数值精度**: 由于浮点运算的顺序和实现差异，可能存在微小的数值差异
4. **block_indices**: 由于 topk 操作在平局情况下可能产生不同结果，block_indices 的对比容忍度会放宽

## 故障排除

### 如果测试失败

1. **检查数值差异大小**
   - 如果差异很小（< 1e-5），可能是正常的浮点误差
   - 如果差异很大，需要检查实现逻辑

2. **检查形状匹配**
   - 确保两个实现的输出形状相同

3. **检查边界情况**
   - 确保正确处理序列边界和 padding

4. **检查随机种子**
   - 确保两个实现使用相同的随机种子

### 常见问题

**Q: 为什么 block_indices 不完全相同？**
A: topk 操作在遇到相同分数时可能返回不同的索引顺序，这是正常的。只要选择的 blocks 相似即可。

**Q: 为什么输出有微小的差异？**
A: Triton 和 PyTorch 的浮点运算顺序可能不同，导致累积误差。只要差异在容忍范围内即可。

**Q: 如何调整容忍度？**
A: 修改 `compare_tensors` 函数调用中的 `rtol` 和 `atol` 参数。

## 示例输出

```
================================================================================
测试 block_sparse_attn
================================================================================
使用设备: cuda

运行 PyTorch 实现...
✓ PyTorch 实现完成
  - o_torch shape: torch.Size([2, 128, 8, 64])
  - block_indices_torch shape: torch.Size([2, 8, 128, 4])

运行 Triton 实现...
✓ Triton 实现完成
  - o_triton shape: torch.Size([2, 128, 8, 64])
  - block_indices_triton shape: torch.Size([2, 8, 128, 4])

比较结果...
✓ 输出 o: 匹配
  - max_diff=1.234567e-05, mean_diff=2.345678e-07, std_diff=1.234567e-06
  - max_rel_diff=1.234567e-04, mean_rel_diff=2.345678e-06
✓ block_indices: 匹配

================================================================================
✓ block_sparse_attn 测试通过！
================================================================================
```

## 扩展

如果需要添加新的测试用例，可以在 `main()` 函数中添加新的测试函数，并按照现有模式实现。

