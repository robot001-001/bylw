from generative_recommenders.ops.hstu_attention import hstu_mha

import time
from typing import Tuple

import torch



def generate_random_jagged_qkv(
    batch_size: int, 
    max_seq_len: int, 
    num_heads: int, 
    dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    lengths = torch.randint(1, max_seq_len + 1, (batch_size,))
    seq_offsets = torch.cat([torch.tensor([0]), torch.cumsum(lengths, dim=0)]).to(torch.int32)
    total_L = seq_offsets[-1].item()
    q = torch.randn(total_L, num_heads, dim)
    k = torch.randn(total_L, num_heads, dim)
    v = torch.randn(total_L, num_heads, dim)
    return q, k, v, seq_offsets, max_seq_len


def speed_exp(
    Bsize, max_seq_len, num_heads, emb_dim
):
    ALPHA = 1.0 / (emb_dim ** 0.5)
    q, k, v, seq_offsets, max_seq_len = generate_random_jagged_qkv(Bsize, max_seq_len, num_heads, emb_dim)
    now = time.time()
    ret = hstu_mha(max_seq_len, ALPHA, q, k, v, seq_offsets, kernel="TRITON")
    time_cost = time.time()-now
    return time_cost


if __name__ == "__main__":
    time_cost = speed_exp(32, 256, 8, 512)
    print(time_cost)