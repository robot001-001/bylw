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

from dataclasses import dataclass
from typing import List
import logging
from tqdm import tqdm

import pandas as pd

import torch

from data.dataset import DatasetV2, MultiFileDatasetV2
from data.dataset_v3 import DatasetV3
from data.item_features import ItemFeatures
from data.preprocessor import get_common_preprocessors


@dataclass
class RecoDataset:
    max_sequence_length: int
    num_unique_items: int
    max_item_id: int
    all_item_ids: List[int]
    train_dataset: torch.utils.data.Dataset
    eval_dataset: torch.utils.data.Dataset

    def presort(self, block_size, emb_matrix, device):
        train_dataset = self.train_dataset
        for idx in tqdm(range(train_dataset.__len__()), desc='presort'):
            sample = train_dataset.__getitem__(idx)
            history_lengths = sample['history_lengths']
            historical_ids = sample['historical_ids']
            historical_ratings = sample['historical_ratings']
            historical_timestamps = sample['historical_timestamps']
            with torch.no_grad():
                historical_id_emb = emb_matrix.get_item_embeddings(historical_ids[:history_lengths].to(device))
            # logging.info(f'historical_ids.shape: {historical_ids.shape}')
            # logging.info(f'historical_id_emb.shape: {historical_id_emb.shape}')
            sorted_indices = self._group_vectors_by_similarity(historical_id_emb, block_size)
            sample['historical_ids'][:history_lengths] = historical_ids[sorted_indices]
            sample['historical_ratings'][:history_lengths] = historical_ratings[sorted_indices]
            sample['historical_timestamps'][:history_lengths] = historical_timestamps[sorted_indices]
            train_dataset._cache[idx] = sample
        self.train_dataset = train_dataset
        return
    
    def _group_vectors_by_similarity(self, tensor, block_size=16):
        seq_len, embed_dim = tensor.shape
        pad_len = (block_size - (seq_len % block_size)) % block_size
        if pad_len > 0:
            padding = torch.zeros(pad_len, embed_dim, device=tensor.device, dtype=tensor.dtype)
            tensor_padded = torch.cat([tensor, padding], dim=0)
        else:
            tensor_padded = tensor
        mean = tensor_padded.mean(dim=0, keepdim=True)
        centered = tensor_padded - mean
        try:
            _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
            first_pc = Vh[0]
        except RuntimeError:
            first_pc = torch.ones(embed_dim, device=tensor.device)
        scores = torch.matmul(centered, first_pc)
        sorted_indices = torch.argsort(scores)
        sorted_indices = sorted_indices[sorted_indices < seq_len]
        # logging.info(f'sorted_indices: {sorted_indices}')
        return sorted_indices.to('cpu')


def get_reco_dataset(
    dataset_name: str,
    max_sequence_length: int,
    chronological: bool,
    use_binary_ratings: bool,
    num_ratings: int,
    positional_sampling_ratio: float = 1.0,
) -> RecoDataset:
    if dataset_name == "ml-1m":
        dp = get_common_preprocessors()[dataset_name]
        train_dataset = DatasetV3(
            # ratings_file=dp.output_format_csv().replace('.csv', '_binary_augment.csv') if use_binary_ratings else dp.output_format_csv(),
            ratings_file=dp.output_format_csv().replace('.csv', '_binary.csv') if use_binary_ratings else dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            chronological=chronological,
            sample_ratio=positional_sampling_ratio,
            num_ratings=num_ratings,
        )
        eval_dataset = DatasetV3(
            ratings_file=dp.output_format_csv().replace('.csv', '_binary.csv') if use_binary_ratings else dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            chronological=chronological,
            sample_ratio=1.0,  # do not sample
            num_ratings=num_ratings,
        )
    elif dataset_name == "ml-20m":
        dp = get_common_preprocessors()[dataset_name]
        train_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            chronological=chronological,
        )
        eval_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            chronological=chronological,
        )
    elif dataset_name == "ml-3b":
        dp = get_common_preprocessors()[dataset_name]
        train_dataset = MultiFileDatasetV2(
            file_prefix="tmp/ml-3b/16x32",
            num_files=16,
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            chronological=chronological,
        )
        eval_dataset = MultiFileDatasetV2(
            file_prefix="tmp/ml-3b/16x32",
            num_files=16,
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            chronological=chronological,
        )
    elif dataset_name == "amzn-books":
        dp = get_common_preprocessors()[dataset_name]
        train_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            shift_id_by=1,  # [0..n-1] -> [1..n]
            chronological=chronological,
        )
        eval_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            shift_id_by=1,  # [0..n-1] -> [1..n]
            chronological=chronological,
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    if dataset_name == "ml-1m" or dataset_name == "ml-20m":
        items = pd.read_csv(dp.processed_item_csv(), delimiter=",")
        max_jagged_dimension = 16
        expected_max_item_id = dp.expected_max_item_id()
        assert expected_max_item_id is not None
        item_features: ItemFeatures = ItemFeatures(
            max_ind_range=[63, 16383, 511],
            num_items=expected_max_item_id + 1,
            max_jagged_dimension=max_jagged_dimension,
            lengths=[
                torch.zeros((expected_max_item_id + 1,), dtype=torch.int64),
                torch.zeros((expected_max_item_id + 1,), dtype=torch.int64),
                torch.zeros((expected_max_item_id + 1,), dtype=torch.int64),
            ],
            values=[
                torch.zeros(
                    (expected_max_item_id + 1, max_jagged_dimension),
                    dtype=torch.int64,
                ),
                torch.zeros(
                    (expected_max_item_id + 1, max_jagged_dimension),
                    dtype=torch.int64,
                ),
                torch.zeros(
                    (expected_max_item_id + 1, max_jagged_dimension),
                    dtype=torch.int64,
                ),
            ],
        )
        all_item_ids = []
        for df_index, row in items.iterrows():
            # print(f"index {df_index}: {row}")
            movie_id = int(row["movie_id"])
            genres = row["genres"].split("|")
            titles = row["cleaned_title"].split(" ")
            # print(f"{index}: genres{genres}, title{titles}")
            genres_vector = [hash(x) % item_features.max_ind_range[0] for x in genres]
            titles_vector = [hash(x) % item_features.max_ind_range[1] for x in titles]
            years_vector = [hash(row["year"]) % item_features.max_ind_range[2]]
            item_features.lengths[0][movie_id] = min(
                len(genres_vector), max_jagged_dimension
            )
            item_features.lengths[1][movie_id] = min(
                len(titles_vector), max_jagged_dimension
            )
            item_features.lengths[2][movie_id] = min(
                len(years_vector), max_jagged_dimension
            )
            for f, f_values in enumerate([genres_vector, titles_vector, years_vector]):
                for j in range(min(len(f_values), max_jagged_dimension)):
                    item_features.values[f][movie_id][j] = f_values[j]
            all_item_ids.append(movie_id)
        max_item_id = dp.expected_max_item_id()
        for x in all_item_ids:
            assert x > 0, "x in all_item_ids should be positive"
    else:
        # expected_max_item_id and item_features are not set for Amazon datasets.
        item_features = None
        max_item_id = dp.expected_num_unique_items()
        all_item_ids = [x + 1 for x in range(max_item_id)]  # pyre-ignore [6]

    return RecoDataset(
        max_sequence_length=max_sequence_length,
        num_unique_items=dp.expected_num_unique_items(),  # pyre-ignore [6]
        max_item_id=max_item_id,  # pyre-ignore [6]
        all_item_ids=all_item_ids,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
