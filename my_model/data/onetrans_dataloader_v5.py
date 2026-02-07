import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import logging

class MovieLensFullDataset(Dataset):
    def __init__(self, csv_path, max_len=50):
        self.max_len = max_len
        self.df = pd.read_csv(csv_path)
        self.df = self.df.reset_index(drop=True)
        
        self.max_item_id = 3952
        self.max_user_id = 6040

    def __len__(self):
        return self.df.shape[0]
    
    def _process_seq(self, idx, seq_str, max_len):
        try:
            seq = [int(float(i)) for i in str(seq_str).split(',')]
        except:
            seq = []
        actual_len = min(len(seq), max_len)
        seq_trimmed = seq[-max_len:]
        pad_len = max_len - len(seq_trimmed)
        padded_seq = [0] * pad_len + seq_trimmed
        return torch.tensor(padded_seq, dtype=torch.long), torch.tensor(actual_len, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        high_items_pad, high_len = self._process_seq(idx, row['high_item_ids'], self.max_len)
        high_times_pad, _ = self._process_seq(idx, row['high_timestamps'], self.max_len)
        seq_items_pad, seq_len = self._process_seq(idx, row['sequence_item_ids'], self.max_len+1)
        seq_ratings_pad, _ = self._process_seq(idx, row['sequence_ratings'], self.max_len+1)
        seq_times_pad, _ = self._process_seq(idx, row['sequence_timestamps'], self.max_len+1)
        
        user_id_tensor = torch.tensor(row['user_id'], dtype=torch.long)

        return (
            high_items_pad,    # 1. 高评分item序列
            high_times_pad,    # 2. 高评分时间戳序列
            high_len,          # 3. 高评分序列真实长度
            user_id_tensor,    # 4. 用户ID
            seq_items_pad,     # 5. 全量item序列
            seq_ratings_pad,   # 6. 全量评分序列
            seq_times_pad,     # 7. 全量时间戳序列
            seq_len            # 8. 全量序列真实长度
        )