# follow datasetv3 code
# 为数据集添加presort逻辑
import csv
import linecache

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import logging
import torch

from data.dataset_v3 import DatasetV3


class DatasetV4(torch.utils.data.Dataset):
    """In reverse chronological order."""

    def __init__(
        self,
        dataset_v3: DatasetV3
    ) -> None:
        """
        Args:
            csv_file (string): Path to the csv file.
        """
        super().__init__()
        self.dataset = dataset_v3
        self._cache = self.dataset._cache

    def __len__(self) -> int:
        return self.dataset.__len__()
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.dataset.__getitem__(idx)

    def presort(self, block_size, emb_matrix):
        for idx in range(self.__len__()):
            sample = self.__getitem__(idx)
            logging.info(f'sample: {sample}')
            break
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
        return sorted_indices