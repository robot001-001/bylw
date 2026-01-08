import torch
from torch import nn
import logging

class GateModel(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_dim,
    ):
        super().__init__()
        self.mlp1 = nn.Linear(in_features=in_features, out_features=hidden_dim)
        self.atv1 = nn.ReLU()
        self.mlp2 = nn.Linear(in_features=hidden_dim, out_features=3) # g_cmp, g_slc, g_swa
        self.atv2 = nn.Sigmoid()
        logging.info(f'successfully init gate model')

    def forward(self, x): # x: [bsize, seq_len, head_num, head_dim]
        x = self.mlp1(x)
        x = self.atv1(x)
        x = self.mlp2(x)
        x = self.atv2(x)# x: [bsize, seq_len, head_num, 3]
        g_cmp, g_slc, g_swa = x.unbind(dim=-1)
        
        return g_cmp, g_slc, g_swa
