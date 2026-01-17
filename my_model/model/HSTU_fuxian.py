import torch
from torch import nn

import logging

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
        self.rating_emb = nn.Embedding(num_embeddings=num_ratings+2, embedding_dim=embedding_dim)
        

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
        return
    
    