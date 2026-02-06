import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# 辅助函数：必须放在类外面，否则 Multiprocessing 会报错
def _parse_single_col(val_str):
    if pd.isna(val_str) or val_str == "":
        return []
    return [int(float(x)) for x in str(val_str).split(',')]

class MovieLensFullDataset(Dataset):
    def __init__(self, csv_path, max_len=50, cache_path=None):
        self.max_len = max_len
        # 如果没指定缓存路径，默认在 CSV 同级目录生成
        if cache_path is None:
            cache_path = csv_path.replace('.csv', '_cache.npz')

        if os.path.exists(cache_path):
            print(f"检测到缓存文件，正在快速加载: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            self.user_ids = data['user_ids']
            self.max_item_id = int(data['max_item_id'])
            self.max_user_id = int(data['max_user_id'])
            # 加载已处理好的 list
            self.data_store = {
                'high_item_ids': data['h_items'].tolist(),
                'high_timestamps': data['h_times'].tolist(),
                'sequence_item_ids': data['s_items'].tolist(),
                'sequence_ratings': data['s_ratings'].tolist(),
                'sequence_timestamps': data['s_times'].tolist()
            }
        else:
            print("未发现缓存，开始并行解析（仅需一次）...")
            df = pd.read_csv(csv_path)
            self.user_ids = df['user_id'].astype(int).values
            
            target_cols = [
                'high_item_ids', 'high_timestamps', 
                'sequence_item_ids', 'sequence_ratings', 'sequence_timestamps'
            ]
            self.data_store = {}
            self.max_item_id = 0

            # 使用多进程并行处理每一列
            num_cores = min(cpu_count(), 8) # 建议开启 8 个进程
            for col in target_cols:
                print(f"正在并行解析列: {col}")
                with Pool(num_cores) as p:
                    # 使用 list(tqdm(...)) 只是为了看进度
                    column_data = list(tqdm(p.imap(_parse_single_col, df[col].values, chunksize=1000), 
                                           total=len(df), desc=col))
                self.data_store[col] = column_data
                
                # 顺便找 max_item_id
                if col == 'sequence_item_ids':
                    # 这里的计算很快
                    self.max_item_id = max([max(s) if s else 0 for s in column_data])

            self.max_user_id = int(self.user_ids.max())

            print("正在保存缓存以备下次使用...")
            np.savez_compressed(cache_path, 
                                h_items=np.array(self.data_store['high_item_ids'], dtype=object),
                                h_times=np.array(self.data_store['high_timestamps'], dtype=object),
                                s_items=np.array(self.data_store['sequence_item_ids'], dtype=object),
                                s_ratings=np.array(self.data_store['sequence_ratings'], dtype=object),
                                s_times=np.array(self.data_store['sequence_timestamps'], dtype=object),
                                user_ids=self.user_ids,
                                max_item_id=self.max_item_id,
                                max_user_id=self.max_user_id)

    def __len__(self):
        return len(self.user_ids)

    def _process_seq(self, seq):
        actual_len = min(len(seq), self.max_len)
        seq_trimmed = seq[-self.max_len:]
        padded_seq = [0] * (self.max_len - len(seq_trimmed)) + seq_trimmed
        return torch.tensor(padded_seq, dtype=torch.long), torch.tensor(actual_len, dtype=torch.long)

    def __getitem__(self, idx):
        h_items, h_len = self._process_seq(self.data_store['high_item_ids'][idx])
        h_times, _     = self._process_seq(self.data_store['high_timestamps'][idx])
        s_items, s_len = self._process_seq(self.data_store['sequence_item_ids'][idx])
        s_ratings, _   = self._process_seq(self.data_store['sequence_ratings'][idx])
        s_times, _     = self._process_seq(self.data_store['sequence_timestamps'][idx])
        
        return (h_items, h_times, h_len, torch.tensor(self.user_ids[idx]), 
                s_items, s_ratings, s_times, s_len)