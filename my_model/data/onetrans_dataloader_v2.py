import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class MovieLensFullDataset(Dataset):
    def __init__(self, csv_path, max_len=50):
        self.max_len = max_len
        print(f"Reading CSV...")
        df = pd.read_csv(csv_path)
        
        # 1. 直接提取基础列
        self.user_ids = df['user_id'].astype(int).values

        # 2. 【批处理核心】定义解析函数
        def batch_parse_to_numpy(series, max_len):
            print(f"Batch parsing {series.name}...")
            # a. 炸开字符串：'1,2,3' -> [1, 2, 3]
            # expand=True 会生成一个新 DataFrame，列数等于最大序列长度
            split_df = series.str.split(',', expand=True)
            
            # b. 截断：只保留最后 max_len 列
            if split_df.shape[1] > max_len:
                split_df = split_df.iloc[:, -max_len:]
            
            # c. 转换为整数矩阵并填充 0
            # float 是为了兼容 "1.0" 这种字符串，然后再转 int
            matrix = split_df.fillna(0).astype(float).astype(int).values
            
            # d. 真实长度：计算每行非 0 的个数 (假设 ID 都是正数)
            # 或者通过 split_df 原本的非空状态计算
            actual_lens = (split_df.notna()).sum(axis=1).values
            
            # e. 统一宽度填充 (如果序列长度不足 max_len，前面补 0)
            if matrix.shape[1] < max_len:
                pad_width = max_len - matrix.shape[1]
                matrix = np.pad(matrix, ((0, 0), (pad_width, 0)), 'constant', constant_values=0)
            
            return matrix, actual_lens

        # 3. 批量处理所有关键列
        self.h_items, self.h_lens = batch_parse_to_numpy(df['high_item_ids'], max_len)
        self.h_times, _ = batch_parse_to_numpy(df['high_timestamps'], max_len)
        
        self.s_items, self.s_lens = batch_parse_to_numpy(df['sequence_item_ids'], max_len)
        self.s_ratings, _ = batch_parse_to_numpy(df['sequence_ratings'], max_len)
        self.s_times, _ = batch_parse_to_numpy(df['sequence_timestamps'], max_len)

        # 4. 统计 ID
        self.max_item_id = int(self.s_items.max())
        self.max_user_id = int(self.user_ids.max())

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        # 现在的 __getitem__ 极其轻量，全是内存索引，没有任何计算
        return (
            torch.from_numpy(self.h_items[idx]),
            torch.from_numpy(self.h_times[idx]),
            torch.tensor(self.h_lens[idx], dtype=torch.long),
            torch.tensor(self.user_ids[idx], dtype=torch.long),
            torch.from_numpy(self.s_items[idx]),
            torch.from_numpy(self.s_ratings[idx]),
            torch.from_numpy(self.s_times[idx]),
            torch.tensor(self.s_lens[idx], dtype=torch.long)
        )