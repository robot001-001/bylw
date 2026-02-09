# rankmixer
# 1. 底层使用sim加工用户序列 -- 软搜曝光序列与点击序列
# 2. 顶层接上rankmixer做scaling

import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

import logging
import math


class RankMixerEmb(nn.Module):
    def __init__(
        self,
        max_itemid: int,
        max_uid: int,
        d_model: int,
        device='cpu'
    ):
        super().__init__()
        self.item_emb = nn.Embedding(max_itemid+1, d_model)
        self.user_emb = nn.Embedding(max_uid+1, d_model)
        self.timestamp_fc = nn.Linear(1, d_model)
        self.exposure_fc = nn.Linear(2*d_model, d_model)
        self.click_fc = nn.Linear(2*d_model, d_model)

        self.device = device


    def forward(self, row):
        high_items_pad = row[0].to(self.device)
        high_times_pad = row[1].to(self.device)
        high_len = row[2].to(self.device)
        user_id_tensor = row[3].to(self.device).unsqueeze(1)
        item_id_tensor = row[4].to(self.device)[:, -1].unsqueeze(1)
        item_rating_tensor = row[5].to(self.device)[:, -1]
        item_time_pad = row[6].to(self.device)[:, -1]
        seq_items_pad = row[4].to(self.device)[:, :-1]
        seq_ratings_pad = row[5].to(self.device)[:, :-1]
        seq_times_pad = row[6].to(self.device)[:, :-1]
        seq_len = row[7].to(self.device)

        high_times_gap = item_time_pad.unsqueeze(1) - high_times_pad
        seq_times_gap = item_time_pad.unsqueeze(1) - seq_times_pad
        
        high_items_emb = self.item_emb(high_items_pad)
        high_times_emb = self.timestamp_fc(torch.clamp(torch.log(high_times_gap.unsqueeze(2)+1.0), min=0))
        click_emb = torch.cat([high_items_emb, high_times_emb], dim=2)
        click_emb = self.click_fc(click_emb)

        seq_items_emb = self.item_emb(seq_items_pad)
        seq_times_emb = self.timestamp_fc(torch.clamp(torch.log(seq_times_gap.unsqueeze(2)+1.0), min=0))
        exposure_emb = torch.cat([seq_items_emb, seq_times_emb], dim=2)
        exposure_emb = self.exposure_fc(exposure_emb)

        uid_emb = self.user_emb(user_id_tensor)
        tgt_emb = self.item_emb(item_id_tensor)
        return uid_emb, tgt_emb, \
            click_emb, high_len, \
            exposure_emb, seq_len, \
            item_rating_tensor