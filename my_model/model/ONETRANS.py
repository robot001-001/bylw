import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

import logging
from typing import Tuple, List
import math

from model.MainTower import MainTowerMLP
from model.positional_embedding.rope import RotaryEmbedding, apply_rotary_pos_emb



class OneTransBlock(nn.Module):
    # 左padding
    def __init__(
        self,
        max_seq_len: int,    # 当前层输入的最大 S 序列长度
        out_seq_len: int,    # 当前层输出的 S 序列长度 (剪枝目标)
        ns_seq_len: int,     # NS 序列长度
        d_model: int,
        num_heads: int,
        ffn_layer_hidden_dim: int
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.out_seq_len = out_seq_len
        self.ns_seq_len = ns_seq_len
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, 
            max_seq_len=max_seq_len+ns_seq_len
        )
        causal_mask = self._create_static_causal_mask()
        self.register_buffer('static_causal_mask', causal_mask)
        self.w_s_qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.w_ns_q = nn.Parameter(torch.empty(self.ns_seq_len, d_model, d_model))
        self.w_ns_k = nn.Parameter(torch.empty(self.ns_seq_len, d_model, d_model))
        self.w_ns_v = nn.Parameter(torch.empty(self.ns_seq_len, d_model, d_model))
        self.b_ns_q = nn.Parameter(torch.zeros(self.ns_seq_len, d_model))
        self.b_ns_k = nn.Parameter(torch.zeros(self.ns_seq_len, d_model))
        self.b_ns_v = nn.Parameter(torch.zeros(self.ns_seq_len, d_model))
        self.num_heads = num_heads
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn_layer_hidden_dim = ffn_layer_hidden_dim
        self._init_mixed_ffn()
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.w_ns_q)
        init.xavier_uniform_(self.w_ns_k)
        init.xavier_uniform_(self.w_ns_v)
        init.xavier_uniform_(self.w1_ns_ffn)
        init.xavier_uniform_(self.w2_ns_ffn)

    def _init_mixed_ffn(self):
        self.ffn_s = nn.Sequential(
            nn.Linear(self.d_model, self.ffn_layer_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.ffn_layer_hidden_dim, self.d_model)
        )
        self.w1_ns_ffn = nn.Parameter(torch.empty(self.ns_seq_len, self.d_model, self.ffn_layer_hidden_dim))
        self.b1_ns_ffn = nn.Parameter(torch.zeros(self.ns_seq_len, self.ffn_layer_hidden_dim))
        self.w2_ns_ffn = nn.Parameter(torch.empty(self.ns_seq_len, self.ffn_layer_hidden_dim, self.d_model))
        self.b2_ns_ffn = nn.Parameter(torch.zeros(self.ns_seq_len, self.d_model))

    def _create_static_causal_mask(self):
        total_len = self.max_seq_len + self.ns_seq_len
        full_mask = torch.tril(torch.ones(total_len, total_len))
        s_query_start = self.max_seq_len - self.out_seq_len
        pruned_mask = full_mask[s_query_start:, :]
        return pruned_mask
    
    def get_valid_mask(self, in_seq_len: torch.Tensor):
        batch_size = in_seq_len.size(0)
        device = in_seq_len.device
        total_len = self.max_seq_len + self.ns_seq_len
        key_indices = torch.arange(total_len, device=device).unsqueeze(0)
        valid_start_indices = (self.max_seq_len - in_seq_len).unsqueeze(1)
        is_ns = key_indices >= self.max_seq_len
        is_valid_s = key_indices >= valid_start_indices
        key_padding_mask = (is_ns | is_valid_s).view(batch_size, 1, 1, total_len)
        causal_mask = self.static_causal_mask.unsqueeze(0).unsqueeze(0)
        final_mask_bool = (causal_mask > 0.5) & key_padding_mask
        attention_mask = torch.zeros_like(final_mask_bool, dtype=torch.float)
        attention_mask.masked_fill_(~final_mask_bool, float("-inf"))
        return attention_mask
        
    def _mha(self, mask, q, k, v, num_heads):
        batch_size = q.size(0)
        d_model = q.size(-1)
        head_dim = d_model // num_heads
        q = q.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        scores = scores + mask
        attn_probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(batch_size, -1, d_model)
        return attn_out
    
    def mixed_ffn(self, x):
        x_s = x[:, :-self.ns_seq_len, :]
        x_ns = x[:, -self.ns_seq_len:, :]
        out_s = self.ffn_s(x_s)
        out_ns = torch.einsum('bni,nio->bno', x_ns, self.w1_ns_ffn) + self.b1_ns_ffn
        out_ns = F.relu(out_ns)
        out_ns = torch.einsum('bni,nio->bno', out_ns, self.w2_ns_ffn) + self.b2_ns_ffn
        return torch.cat([out_s, out_ns], dim=1)

    def forward(self, x, in_seq_len: torch.Tensor):
        device = x.device
        mask = self.get_valid_mask(in_seq_len)
        batch_size = x.size(0)
        residual_1 = x
        x_norm = self.norm1(x)
        x_s = x_norm[:, :-self.ns_seq_len, :]
        x_ns = x_norm[:, -self.ns_seq_len:, :]
        out_s_qkv = self.w_s_qkv(x_s)
        q_s, k_s, v_s = out_s_qkv.chunk(3, dim=-1)
        q_ns = torch.einsum('bni,nio->bno', x_ns, self.w_ns_q) + self.b_ns_q
        k_ns = torch.einsum('bni,nio->bno', x_ns, self.w_ns_k) + self.b_ns_k
        v_ns = torch.einsum('bni,nio->bno', x_ns, self.w_ns_v) + self.b_ns_v
        # q_s = q_s[:, -self.out_seq_len:, :]
        if self.out_seq_len > 0:
            q_s = q_s[:, -self.out_seq_len:, :]
        else:
            q_s = q_s[:, :0, :]
        q = torch.cat([q_s, q_ns], dim=1)
        k = torch.cat([k_s, k_ns], dim=1)
        v = torch.cat([v_s, v_ns], dim=1)
        q_rope = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_rope = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        total_k_len = k.size(1)
        k_pos_ids = torch.arange(total_k_len, device=device)
        total_q_len = q.size(1)
        q_pos_ids = torch.arange(total_k_len - total_q_len, total_k_len, device=device)
        freqs_cis = self.rotary_emb(k_rope)
        q_rope = apply_rotary_pos_emb(q_rope, q_pos_ids, freqs_cis)
        k_rope = apply_rotary_pos_emb(k_rope, k_pos_ids, freqs_cis)
        q_out = q_rope.transpose(1, 2).flatten(2)
        k_out = k_rope.transpose(1, 2).flatten(2)
        attn_out = self._mha(mask, q_out, k_out, v, num_heads=self.num_heads)
        logging.info(f'attn_out: {attn_out}')
        current_valid_len = self.out_seq_len + self.ns_seq_len
        residual_1 = residual_1[:, -current_valid_len:, :]
        x = residual_1 + attn_out
        residual_2 = x
        x_norm = self.norm2(x)
        x = self.mixed_ffn(x_norm) + residual_2
        out_seq_len_tensor = torch.clamp(in_seq_len, max=self.out_seq_len)
        return x, out_seq_len_tensor



class ONETRANS(nn.Module):
    def __init__(
        self,
        num_layers: int,
        max_seq_len: List[int], # 长度=num_layers
        ns_seq_len: int,
        d_model: int,
        num_heads: int,
        ffn_layer_hidden_dim: int,
        main_tower_units: List[int],
    ):
        super().__init__()
        self.num_layers = num_layers
        max_seq_len.append(0)
        self.onetrans = nn.ModuleList()
        for i in range(num_layers):
            self.onetrans.append( 
                OneTransBlock(
                    max_seq_len=max_seq_len[i],
                    out_seq_len=max_seq_len[i+1],
                    ns_seq_len=ns_seq_len,
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_layer_hidden_dim=ffn_layer_hidden_dim
                )
            )
        self.main_tower = MainTowerMLP(ns_seq_len*d_model, main_tower_units)

    def forward(self, x, in_seq_len):
        for i in range(self.num_layers):
            # print(f'running layer {i}')
            x, in_seq_len = self.onetrans[i](x, in_seq_len)
            logging.info(f'layer: {i}, x: {x.shape}, {x[3, :, :2]}')
        output_embedding = x.reshape(x.size(0), -1)
        out = self.main_tower(output_embedding)
        return out
    





if __name__ == "__main__":
    bsize = 2
    num_layers = 5
    max_seq_len = [128, 64, 32, 16, 8]
    ns_seq_len = 2
    d_model = 50
    num_heads = 2
    ffn_layer_hidden_dim = 128
    main_tower_units = [128, 2]
    x = torch.rand(bsize, max_seq_len[0]+ns_seq_len, d_model)
    in_seq_len = torch.tensor([19, 128])

    model = ONETRANS(
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        ns_seq_len=ns_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        ffn_layer_hidden_dim=ffn_layer_hidden_dim,
        main_tower_units=main_tower_units
    )
    ret = model(x, in_seq_len)