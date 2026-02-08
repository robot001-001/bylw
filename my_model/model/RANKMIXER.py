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

from model.MainTower import MainTowerMLP


class RankMixerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_experts: int = 4,
        expert_dim: int = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_experts = num_experts
        if expert_dim is None:
            expert_dim = d_model * 4

        self.router_weight = nn.Parameter(torch.randn(num_heads, d_model, num_experts))

        self.experts_w1 = nn.Parameter(torch.randn(num_heads, num_experts, d_model, expert_dim))
        self.experts_b1 = nn.Parameter(torch.zeros(num_heads, num_experts, expert_dim))

        self.experts_w2 = nn.Parameter(torch.randn(num_heads, num_experts, expert_dim, d_model))
        self.experts_b2 = nn.Parameter(torch.zeros(num_heads, num_experts, d_model))

        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        init.xavier_uniform_(self.router_weight)
        init.xavier_uniform_(self.experts_w1)
        init.xavier_uniform_(self.experts_w2)

    def _token_mixing(self, input_embeddings):
        B, T, D = input_embeddings.shape
        head_dim = D // self.num_heads
        ret = input_embeddings.reshape(B, T, self.num_heads, head_dim).transpose(1, 2).reshape(B, T, D)
        return ret
    
    def _smoe(self, x):
        gate_logits = torch.einsum('btd, tde -> bte', x, self.router_weight)
        gates = F.relu(gate_logits) 
        hidden = torch.einsum('btd, tedh -> bteh', x, self.experts_w1) + self.experts_b1
        hidden = F.gelu(hidden)
        expert_out = torch.einsum('bteh, tehd -> bted', hidden, self.experts_w2) + self.experts_b2
        final_out = torch.sum(gates.unsqueeze(-1) * expert_out, dim=2)
        return final_out

    def forward(
        self,
        input_embeddings
    ):
        mix_out = self._token_mixing(input_embeddings) + input_embeddings
        mix_out = self.norm1(mix_out)
        out = self._smoe(mix_out) + mix_out
        out = self.norm2(out)
        return out
    

class RANKMIXER(nn.Module):
    def __init__(
        self,
        num_blocks,
        d_model,
        num_heads,
        main_tower_units,
        num_experts=4,
        expert_dim=None,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.rankmixer = nn.ModuleList()
        for idx in range(num_blocks):
            self.rankmixer.append(
                RankMixerBlock(
                    d_model, num_heads, num_experts, expert_dim
                )
            )
        self.main_tower = MainTowerMLP(
            d_model, main_tower_units
        )

    def forward(
        self, 
        uid_emb,
        itemid_emb,
        sim_out,
    ):
        B, _ = uid_emb.shape
        device = uid_emb.device
        sim_out = sim_out.view(B, -1, self.d_model)
        uid_emb = uid_emb.unsqueeze(1)
        itemid_emb = itemid_emb.unsqueeze(1)
        input_embeddings = torch.cat([itemid_emb, uid_emb, sim_out], dim=1)
        for layer in self.rankmixer:
            input_embeddings = layer(input_embeddings)

        out = input_embeddings.mean(dim=1)
        out = self.main_tower(out)
        return out