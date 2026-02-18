# 选出长序列来做实验
import pandas as pd
import os

src_file = 'tmp/amzn_books/sasrec_format_binary.csv'
out_file = 'tmp/amzn_books/sasrec_format_binary_v1.csv'

data = pd.read_csv(src_file)
print(f'data.shape: {data.shape}')
data['seq_len'] = (data['sequence_ratings'].apply(len)-1)//2
data = data[data.seq_len >= 128]
del data['seq_len']
print(f'data.shape: {data.shape}')
data.to_csv(out_file, index=False)