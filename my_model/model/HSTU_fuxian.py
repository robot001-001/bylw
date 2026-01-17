import torch
from torch import nn
import torch.nn.functional as F
torch.set_printoptions(
    precision=4,        # 显示的小数位数
    threshold=float('inf'),     # 触发折叠（显示省略号）的元素总数阈值
    edgeitems=3,        # 折叠时，开头和结尾显示的元素个数
    linewidth=80,       # 每行的字符宽度
    profile=None,       # 使用预设配置 ('default', 'short', 'full')
    sci_mode=True       # 是否使用科学计数法 (True/False)
)

import logging



class HSTULayer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        linear_dim,
        attention_dim,
        num_heads,
    ):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._linear_dim = linear_dim
        self._attention_dim = attention_dim
        self._num_heads = num_heads
        self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(
                (
                    self._embedding_dim,
                    self._linear_dim * 2 * self._num_heads
                    + self._attention_dim * self._num_heads * 2,
                )
            ).normal_(mean=0, std=0.02),
        )
        self._input_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self._attn_out_norm = nn.LayerNorm(self._linear_dim*self._num_heads, eps=1e-6)

    def forward(
        self, 
        user_embedding,
        attn_mask
    ):
        normed_x = self._input_norm(user_embedding)
        logging.info(f'normed_x: {normed_x.shape}')
        logging.info(f'self._uvqk: {self._uvqk.shape}')
        batched_mm_output = torch.matmul(normed_x, self._uvqk)
        u, v, q, k = torch.split(
            batched_mm_output,
            [
                self._linear_dim * self._num_heads,
                self._linear_dim * self._num_heads,
                self._attention_dim * self._num_heads,
                self._attention_dim * self._num_heads,
            ],
            dim=1,
        )
        attn_out = self._hstu_attention(q, k, v, u, attn_mask)
        return user_embedding + attn_out

    def _hstu_attention(
        self,
        q, k, v, u,
        attn_mask
    ):
        B, S, _ = u.shape
        q = q.view(B, S, self._num_heads, self._attention_dim)
        k = k.view(B, S, self._num_heads, self._attention_dim)
        v = v.view(B, S, self._num_heads, self._linear_dim)

        qk_attn = torch.einsum("bnhd,bmhd->bhnm", q, k)
        # TODO: PE
        qk_attn = F.silu(qk_attn)
        qk_attn *= attn_mask.unsqueeze(0).unsqueeze(0)
        attn_out = torch.einsum("bhnm,bmhd->bnhd", qk_attn, v)
        attn_out = attn_out.view(B, S, _)
        attn_out = self._attn_out_norm(attn_out)*u
        return attn_out





class HSTU(nn.Module):
    def __init__(
        self,
        max_seq_len,
        embedding_dim,
        dropout_rate,
        num_ratings,
        linear_dim,
        attention_dim,
        normalization,
        linear_config,
        linear_activation,
        num_blocks,
        num_heads,
        linear_dropout_rate,
        attn_dropout_rate,
        main_tower_units,
        concat_ua,
        enable_relative_attention_bias
    ):
        super().__init__()
        self._max_seq_len = max_seq_len
        self._embedding_dim = embedding_dim
        self._linear_dim = linear_dim
        self._attention_dim = attention_dim
        self._num_ratings = num_ratings
        self._num_heads = num_heads

        self.rating_emb = nn.Embedding(num_embeddings=self._num_ratings+2, embedding_dim=self._embedding_dim)
        self._hstu = nn.ModuleList()
        for idx in range(num_blocks):
            self._hstu.add_module(
                f'HSTULayer_{idx}', 
                HSTULayer(
                    embedding_dim=self._embedding_dim, 
                    linear_dim=self._linear_dim, 
                    attention_dim=self._attention_dim, 
                    num_heads=self._num_heads
                )
            )
        self.get_attn_mask()
        

    def forward(
        self,
        past_lengths,
        past_ids,
        past_embeddings,
        past_payloads
    ):
        logging.info(f'past_lengths: {past_lengths}')
        logging.info(f'past_ids: {past_ids}')
        logging.info(f'past_embeddings: {past_embeddings.shape}')
        device = past_lengths.device
        float_dtype = past_embeddings.dtype

        timestamps = past_payloads['timestamps']
        ratings = past_payloads['ratings']
        rating_embeddings = self.rating_emb(ratings)
        logging.info(f'timestamps: {timestamps}')
        logging.info(f'ratings: {ratings}')
        logging.info(f'rating_embeddings: {rating_embeddings.shape}')

        B, S, D = past_embeddings.shape
        user_embeddings = torch.stack([past_embeddings, rating_embeddings], dim=2).reshape(B, S*2, D)
        logging.info(f'past_embeddings: {past_embeddings[..., 0]}')
        logging.info(f'rating_embeddings: {rating_embeddings[..., 0]}')
        logging.info(f'user_embeddings: {user_embeddings[..., 0]}')

        user_embeddings = self.hstu_forward(user_embeddings, float_dtype)
        return user_embeddings
    
    def hstu_forward(
        self,
        user_embeddings,
        float_dtype
    ):
        for idx, layer in enumerate(self._hstu):
            user_embeddings = layer(user_embeddings, 1.0 - self._attn_mask.to(float_dtype))
        return user_embeddings
    
    def get_attn_mask(self):
        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones(
                    (
                        self._max_seq_len*2+2,
                        self._max_seq_len*2+2,
                    ),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )