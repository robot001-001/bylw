# rankmixer
# 1. 底层使用sim加工用户序列 -- 软搜曝光序列与点击序列
# 2. 顶层接上rankmixer做scaling

import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

import logging
import math
from typing import Tuple

from model.MainTower import MainTowerMLP as MLP


class SIM(nn.Module):
    def __init__(
        self,
        topk: int,
        max_seq_len: int,
        d_model: int,
        main_tower_units: Tuple[int]
    ):
        super().__init__()
        self.topk = topk
        self.max_seq_len = max_seq_len
        self.mlp = MLP(2*d_model, main_tower_units)


    def _generate_left_padding_mask(self, seq_len, max_len, device):
        batch_size = seq_len.shape[0]
        range_tensor = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, max_len)
        start_indices = max_len - seq_len.unsqueeze(1)
        mask = range_tensor >= start_indices
        return mask

    def _soft_search(self, tgt_emb, seq, seq_len):
        # 实现SIM架构的软搜索功能
        # shape
        #   tgt_emb: [batch_size, d_model]
        #   seq: [batch_size, max_seq_len, d_model]
        #   seq_len: [batch_size, ]
        # 其中，seq是左padding成完整形状的矩阵，其真实长度由seq_len给出
        B, L, D = seq.shape
        K = self.topk
        query = tgt_emb.unsqueeze(1)
        scores = torch.bmm(query, seq.transpose(1, 2)).squeeze(1)
        mask = self._generate_left_padding_mask(seq_len, L, seq.device)
        scores = scores.masked_fill(~mask, -1e9)
        topk_scores, topk_indices = torch.topk(scores, k=K, dim=1)
        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        topk_emb = torch.gather(seq, 1, gather_indices)
        attention_weights = F.softmax(topk_scores, dim=1).unsqueeze(-1)
        final_vector = torch.sum(topk_emb * attention_weights, dim=1)
        return final_vector.unsqueeze(1)


    def forward(
        self, 
        tgt_emb, 
        click_emb, click_len,
        exposure_emb, exposure_len
    ):
        click_out = self._soft_search(tgt_emb, click_emb, click_len)
        exposure_out = self._soft_search(tgt_emb, exposure_emb, exposure_len)
        out = torch.cat([click_out, exposure_out], dim=2)
        out = self.mlp(out)
        return out