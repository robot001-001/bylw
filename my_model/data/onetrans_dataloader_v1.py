import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

class MovieLensFullDataset(Dataset):
    def __init__(self, csv_path, max_len=50):
        self.max_len = max_len
        print(f"Loading CSV from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        num_samples = len(df)
        
        # 1. 预分配 NumPy 数组（内存连续，速度远快于 List）
        # 使用 object 类型存储列表，虽然还是 list，但减少了 pandas 索引开销
        self.user_ids = df['user_id'].astype(int).values
        
        # 定义需要处理的列
        target_cols = [
            'high_item_ids', 'high_timestamps', 
            'sequence_item_ids', 'sequence_ratings', 'sequence_timestamps'
        ]
        
        # 初始化存储字典
        self.data_store = {col: [] for col in target_cols}
        self.max_item_id = 0

        # 2. 【核心优化】单次循环处理所有列
        print("Parsing sequences...")
        # 预取 values 避开索引开销
        col_values = {col: df[col].values for col in target_cols}
        
        for i in tqdm(range(num_samples), desc="Processing rows"):
            for col in target_cols:
                val = col_values[col][i]
                if pd.isna(val) or val == "":
                    res = []
                else:
                    # 优化解析：先 split，再统一转 int
                    res = [int(float(x)) for x in str(val).split(',')]
                
                self.data_store[col].append(res)
                
                # 顺便统计 max_item_id，省去后续循环
                if col == 'sequence_item_ids' and res:
                    local_max = max(res)
                    if local_max > self.max_item_id:
                        self.max_item_id = local_max

        self.max_user_id = int(self.user_ids.max())

    def __len__(self):
        return len(self.user_ids)

    def _process_seq(self, seq):
        actual_len = min(len(seq), self.max_len)
        seq_trimmed = seq[-self.max_len:]
        pad_len = self.max_len - len(seq_trimmed)
        # 优化：直接在生成 tensor 时处理 padding，避免多次列表相加
        padded_seq = [0] * pad_len + seq_trimmed
        return torch.tensor(padded_seq, dtype=torch.long), torch.tensor(actual_len, dtype=torch.long)

    def __getitem__(self, idx):
        # 从 data_store 中直接提取
        h_items_pad, h_len = self._process_seq(self.data_store['high_item_ids'][idx])
        h_times_pad, _     = self._process_seq(self.data_store['high_timestamps'][idx])
        s_items_pad, s_len = self._process_seq(self.data_store['sequence_item_ids'][idx])
        s_ratings_pad, _   = self._process_seq(self.data_store['sequence_ratings'][idx])
        s_times_pad, _     = self._process_seq(self.data_store['sequence_timestamps'][idx])
        
        user_id_tensor = torch.tensor(self.user_ids[idx], dtype=torch.long)

        return (
            h_items_pad, h_times_pad, h_len,
            user_id_tensor,
            s_items_pad, s_ratings_pad, s_times_pad, s_len
        )