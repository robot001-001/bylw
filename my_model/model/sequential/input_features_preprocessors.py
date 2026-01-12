# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

import abc
import math
from typing import Dict, Tuple

import torch

from model.initialization import truncated_normal


class InputFeaturesPreprocessorModule(torch.nn.Module):
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class LearnablePositionalEmbeddingInputFeaturesPreprocessor(
    InputFeaturesPreprocessorModule
):
    def __init__(
        self,
        max_sequence_len: int,
        embedding_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = embedding_dim
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.reset_state()

    def debug_str(self) -> str:
        return f"posi_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()
        D = past_embeddings.size(-1)

        user_embeddings = past_embeddings * (self._embedding_dim**0.5) + self._pos_emb(
            torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]
        user_embeddings *= valid_mask
        return past_lengths, user_embeddings, valid_mask


class LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor(
    InputFeaturesPreprocessorModule
):
    def __init__(
        self,
        max_sequence_len: int,
        item_embedding_dim: int,
        dropout_rate: float,
        rating_embedding_dim: int,
        num_ratings: int,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = item_embedding_dim + rating_embedding_dim
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_ratings,
            rating_embedding_dim,
        )
        self.reset_state()

    def debug_str(self) -> str:
        return f"posir_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )
        truncated_normal(
            self._rating_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()

        user_embeddings = torch.cat(
            [past_embeddings, self._rating_emb(past_payloads["ratings"].int())],
            dim=-1,
        ) * (self._embedding_dim**0.5) + self._pos_emb(
            torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]
        user_embeddings *= valid_mask
        return past_lengths, user_embeddings, valid_mask


class CombinedItemAndRatingInputFeaturesPreprocessor(InputFeaturesPreprocessorModule):
    def __init__(
        self,
        max_sequence_len: int,
        item_embedding_dim: int,
        dropout_rate: float,
        num_ratings: int,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = item_embedding_dim
        # Due to [item_0, rating_0, item_1, rating_1, ...]
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len * 2,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_ratings,
            item_embedding_dim,
        )
        self.reset_state()

    def debug_str(self) -> str:
        return f"combir_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )
        truncated_normal(
            self._rating_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def get_preprocessed_ids(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns (B, N * 2,) x int64.
        """
        B, N = past_ids.size()
        return torch.cat(
            [
                past_ids.unsqueeze(2),  # (B, N, 1)
                past_payloads["ratings"].to(past_ids.dtype).unsqueeze(2),
            ],
            dim=2,
        ).reshape(B, N * 2)

    def get_preprocessed_masks(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns (B, N * 2,) x bool.
        """
        B, N = past_ids.size()
        return (past_ids != 0).unsqueeze(2).expand(-1, -1, 2).reshape(B, N * 2)

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()
        D = past_embeddings.size(-1)

        user_embeddings = torch.cat(
            [
                past_embeddings,  # (B, N, D)
                self._rating_emb(past_payloads["ratings"].int()),
            ],
            dim=2,
        ) * (self._embedding_dim**0.5)
        user_embeddings = user_embeddings.view(B, N * 2, D)
        user_embeddings = user_embeddings + self._pos_emb(
            torch.arange(N * 2, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (
            self.get_preprocessed_masks(
                past_lengths,
                past_ids,
                past_embeddings,
                past_payloads,
            )
            .unsqueeze(2)
            .float()
        )  # (B, N * 2, 1,)
        user_embeddings *= valid_mask
        return past_lengths * 2, user_embeddings, valid_mask


class CombinedItemAndRatingInputFeaturesPreprocessorV1(InputFeaturesPreprocessorModule):
    def __init__(
        self,
        max_sequence_len: int,
        item_embedding_dim: int,
        dropout_rate: float,
        num_ratings: int,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = item_embedding_dim
        # Due to [item_0, rating_0, item_1, rating_1, ...]
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len * 2,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_ratings+2, # 评分1-5要能取到5, 所以num_embs=num_ratings+1, 由于mask掉的tgt_rating用的值是6, 这里再+1防止越界
            item_embedding_dim,
        )
        self.reset_state()

    def debug_str(self) -> str:
        return f"combir_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )
        truncated_normal(
            self._rating_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def get_preprocessed_ids(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns (B, N * 2,) x int64.
        """
        B, N = past_ids.size()
        return torch.cat(
            [
                past_ids.unsqueeze(2),  # (B, N, 1)
                past_payloads["ratings"].to(past_ids.dtype).unsqueeze(2),
            ],
            dim=2,
        ).reshape(B, N * 2)

    def get_preprocessed_masks(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_ratings: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns (B, N * 2,) x bool.
        """
        B, N = past_ids.size()
        past_ids_valid = (past_ids != 0)
        past_ratings_valid = (past_ratings !=0) & (past_ratings !=6)
        valid_mask = torch.stack([past_ids_valid, past_ratings_valid], dim=-1).flatten(start_dim=1)
        return valid_mask

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()
        D = past_embeddings.size(-1)

        past_ratings = past_payloads["ratings"].int()
        user_embeddings = torch.cat(
            [
                past_embeddings,  # (B, N, D)
                self._rating_emb(past_ratings),
            ],
            dim=2,
        ) * (self._embedding_dim**0.5)
        user_embeddings = user_embeddings.view(B, N * 2, D)
        user_embeddings = user_embeddings + self._pos_emb(
            torch.arange(N * 2, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (
            self.get_preprocessed_masks(
                past_lengths,
                past_ids,
                past_ratings,
                past_embeddings,
                past_payloads,
            )
            .unsqueeze(2)
            .float()
        )  # (B, N * 2, 1,)
        user_embeddings *= valid_mask
        return past_lengths * 2, user_embeddings, valid_mask
    

class CombinedItemAndRatingInputFeaturesPreprocessorV2(torch.nn.Module):
    # v2: 加入item/action标识embedding
    def __init__(
        self,
        max_sequence_len: int,
        item_embedding_dim: int,
        dropout_rate: float,
        num_ratings: int,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = item_embedding_dim
        self._max_sequence_len: int = max_sequence_len
        
        # 1. Positional Embedding (长度需要 x2)
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len * 2,
            self._embedding_dim,
        )
        
        # 2. Item/Action Type Embedding (0: Item, 1: Rating)
        self._iasig_emb: torch.nn.Embedding = torch.nn.Embedding(
            2,
            self._embedding_dim,
        )
        
        # --- [修复点 1] ---
        # 预先生成 010101 序列的索引，并注册为 buffer
        # 这样它会自动随模型做 .to(device) 操作，且不作为参数被优化器更新
        # 长度必须是 max_sequence_len * 2
        iasig_ids = torch.zeros((max_sequence_len * 2,), dtype=torch.long)
        iasig_ids[1::2] = 1 # 偶数位(Index 1,3...)设为1 (Rating)
        self.register_buffer("_iasig_ids", iasig_ids)
        
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        
        self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_ratings + 2, 
            item_embedding_dim,
        )
        self.reset_state()

    def debug_str(self) -> str:
        return f"combir_d{self._dropout_rate}"

    def reset_state(self) -> None:
        # 初始化权重
        for m in [self._pos_emb, self._rating_emb, self._iasig_emb]:
            torch.nn.init.trunc_normal_(
                m.weight.data,
                mean=0.0,
                std=math.sqrt(1.0 / self._embedding_dim),
            )

    def get_preprocessed_masks(
        self,
        past_ids: torch.Tensor,
        past_ratings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns (B, N * 2,) x bool.
        """
        # Interleave Mask 的逻辑：Item 非 0 且 Rating 有效
        past_ids_valid = (past_ids != 0)
        # 假设 6 是 mask token, 0 是 padding
        past_ratings_valid = (past_ratings != 0) & (past_ratings != 6) 
        
        # stack & flatten 得到 [Item_Mask, Rating_Mask, Item_Mask, ...]
        valid_mask = torch.stack([past_ids_valid, past_ratings_valid], dim=-1).flatten(start_dim=1)
        return valid_mask

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        B, N = past_ids.size()
        D = past_embeddings.size(-1)
        device = past_ids.device

        past_ratings = past_payloads["ratings"].int()

        # 1. 构造 Interleave 的 User Embeddings
        # [Item_Emb, Rating_Emb, Item_Emb, Rating_Emb ...]
        user_embeddings = torch.cat(
            [
                past_embeddings,  # (B, N, D)
                self._rating_emb(past_ratings), # (B, N, D)
            ],
            dim=2,
        ) * (self._embedding_dim**0.5)
        
        # (B, N, 2, D) -> (B, N*2, D)
        user_embeddings = user_embeddings.view(B, N * 2, D)

        # 2. 加上 Positional Embedding
        # 动态生成 0 ~ 2N-1
        positions = torch.arange(N * 2, device=device).unsqueeze(0) # (1, 2N)
        user_embeddings = user_embeddings + self._pos_emb(positions)

        # 3. 加上 Item/Action Type Embedding (0/1)
        # --- [修复点 2] ---
        # 动态切片，取前 2N 个 ID，并查表
        # self._iasig_ids 已经在正确的 device 上了
        type_ids = self._iasig_ids[:N * 2].unsqueeze(0) # (1, 2N)
        user_embeddings = user_embeddings + self._iasig_emb(type_ids)

        user_embeddings = self._emb_dropout(user_embeddings)

        # 4. Masking
        valid_mask = (
            self.get_preprocessed_masks(
                past_ids,
                past_ratings,
            )
            .unsqueeze(2)
            .float()
        )  # (B, N * 2, 1)
        
        user_embeddings *= valid_mask
        
        # 注意：Sequence Length 变为了 2 倍
        return past_lengths * 2, user_embeddings, valid_mask