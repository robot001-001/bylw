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