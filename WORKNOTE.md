# 2025-12-17
* 修正环境配置 -- torchrec版本不兼容问题
* 代码开发
    * 跑通自建baseline代码
* todo
    * 现在使用的是`interleave`数据，HSTU的max_seq_len没有乘二，需修正
    * 以及tgt尚未拼接到输入序列中