import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class MovieLensFullDataset(Dataset):
    def __init__(self, csv_path, max_len=50):
        self.max_len = max_len
        df = pd.read_csv(csv_path)
        
        def str_to_list(s):
            if pd.isna(s) or s == "": return []
            return [int(float(i)) for i in str(s).split(',')]
        self.user_ids = df['user_id'].astype(int).values
        self.high_item_ids = [str_to_list(x) for x in df['high_item_ids']]
        self.high_timestamps = [str_to_list(x) for x in df['high_timestamps']]
        self.seq_item_ids = [str_to_list(x) for x in df['sequence_item_ids']]
        self.seq_ratings = [str_to_list(x) for x in df['sequence_ratings']]
        self.seq_timestamps = [str_to_list(x) for x in df['sequence_timestamps']]

        current_max = 0
        for seq in self.seq_item_ids:
            if seq:
                local_max = max(seq)
                if local_max > current_max:
                    current_max = local_max
        self.max_item_id = current_max

    def __len__(self):
        return len(self.user_ids)

    def _process_seq(self, seq):
        actual_len = min(len(seq), self.max_len)
        
        seq_trimmed = seq[-self.max_len:]
        pad_len = self.max_len - len(seq_trimmed)
        padded_seq = [0] * pad_len + seq_trimmed
        
        return torch.tensor(padded_seq, dtype=torch.long), torch.tensor(actual_len, dtype=torch.long)

    def __getitem__(self, idx):
        high_items_pad, high_len = self._process_seq(self.high_item_ids[idx])
        high_times_pad, _ = self._process_seq(self.high_timestamps[idx])
        seq_items_pad, seq_len = self._process_seq(self.seq_item_ids[idx])
        seq_ratings_pad, _ = self._process_seq(self.seq_ratings[idx])
        seq_times_pad, _ = self._process_seq(self.seq_timestamps[idx])
        
        user_id_tensor = torch.tensor(self.user_ids[idx], dtype=torch.long)

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