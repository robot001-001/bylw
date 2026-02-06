# 1. 点击序列embedding
# 2. 曝光序列embedding
# 3. 时间戳embedding
# 4. 纵向拼接点击序列、时间戳
# 5. 纵向拼接曝光序列、时间戳
# 6. 横向拼接点击序列、<SEP>、曝光序列
# 7. uid embedding
# 8. itemid embedding

import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

import logging




class OneTransEmb(nn.Module):
    def __init__(
        self,
        max_itemid: int,
        max_uid: int,
        d_model: int
    ):
        super().__init__()
        self.exposure_emb = nn.Embedding(max_itemid, d_model)
        self.click_emb = nn.Embedding(max_itemid, d_model)
        self.uid_emb = nn.Embedding(max_uid, d_model)
        self.timestamp_y_emb = nn.Embedding(3000, d_model)
        self.timestamp_m_emb = nn.Embedding(20, d_model)
        self.timestamp_d_emb = nn.Embedding(50, d_model)
        self.timestamp_hh_emb = nn.Embedding(30, d_model)
        self.timestamp_mm_emb = nn.Embedding(100, d_model)
        self.timestamp_ss_emb = nn.Embedding(100, d_model)

    def forward(self):
        return