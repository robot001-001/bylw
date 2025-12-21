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