# 2025-12-17
* 修正环境配置 -- torchrec版本不兼容问题
* 代码开发
    * 跑通自建baseline代码
* todo
    * 现在使用的是`interleave`数据，HSTU的max_seq_len没有乘二，需修正
    * 以及tgt尚未拼接到输入序列中

# 2025-12-18
* 数据处理流程done
    * 数据处理完成后(preprocess后)，其形式为[item1,rating1,item2,rating2,...,tgt,fake_tgt_rating,...], mask只到tgt
* todo
    * 训练pipe
        * loss 重写

# 2025-12-21
* 训练pipe done
    * 定义loss
    * 定义eval
    * baseline 指标过低
* todo
    * 尝试从召回模型embedding做embedding冷启动，只训上游模型参数
    * 尝试加大参数量、数据量，看是否能解决冷启问题
    * 直接follow原脚本训hstu显存占用12g，但是自定义脚本显存占用2g，分析问题所在

# 2026-01-11
* 跑通hstu_nsa、hstu_interleave代码
* 完成binary数据清洗与训练脚本
* todo
    * hstu模型中endboundaries计算有问题，排查

# 2026-01-12
* 验证`endboundaries`取值
    * 应为 past_lengths -2
    * 是因为 past_lengths 已经*2了，然后需要移除末位元素-1，索引从0开始-1
* hstu_baseline中新加入 item/action signal encoding
    * hstu_baseline中有收益
    * hstu_interleave中无收益
* TODO:
    * 时间戳/preprocess/hstu长度有问题，统一修改成402截断而不是401

# 2026-01-13
* 时间戳/preprocess/hstu长度修正
    * 没涨点
* 测试bceloss
    * 没涨点
* 测试调参
    * 没涨点
* TODO：
    * 检查代码中mask的取值是否符合预期

# 2026-01-17
* 检查HSTU中的attn_mask
    * 符合预期
* 手动复现hstu源代码
    * 效果与开源一致
* 使用训练集eval
    * 发现问题，模型在训练集上acc能做到99.9%
    * 推测是数据量太少，模型记住了所有item对应取值
* TODO：
    * 修改数据集，实现pretrain逻辑