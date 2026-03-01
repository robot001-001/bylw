import pandas as pd
from tqdm import tqdm
import os

# ================= 配置 =================
INPUT_FILE = 'tmp/kuai/data/sasrec_format_binary.csv'
OUTPUT_FILE = 'tmp/kuai/data/sasrec_format_binary_augment.csv'
MIN_SEQ_LEN = 5     # 只有长度 >= 5 的序列才会被保存
CHUNK_SIZE = 1000   # 每次读取 1000 个用户进行处理，可根据内存大小调整
# =======================================

# 如果输出文件已存在，先删除，防止重复运行导致数据堆叠
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

print(f"正在开始分块处理数据: {INPUT_FILE} ...")

# 1. 使用 chunksize 分块读取
reader = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE)
first_chunk = True
total_original_users = 0
total_augmented_rows = 0

# 

for chunk in tqdm(reader, desc="Processing Chunks"):
    chunk_new_rows = []
    total_original_users += len(chunk)
    
    for _, row in chunk.iterrows():
        # 2. 解析基础数据
        try:
            full_items = str(row['sequence_item_ids']).split(',')
            full_ratings = str(row['sequence_ratings']).split(',')
            full_timestamps = str(row['sequence_timestamps']).split(',')
        except Exception as e:
            continue

        seq_len = len(full_items)
        if seq_len < MIN_SEQ_LEN:
            continue
            
        # 3. 滑动窗口切分
        # 将原始行转为字典，处理起来比 row.copy() 更快
        base_dict = row.to_dict()
        
        for length in range(MIN_SEQ_LEN, seq_len + 1): # 注意：如果是包含完整长度，用 seq_len + 1
            new_row = base_dict.copy()
            
            new_row['sequence_item_ids'] = ",".join(full_items[:length])
            new_row['sequence_ratings'] = ",".join(full_ratings[:length])
            new_row['sequence_timestamps'] = ",".join(full_timestamps[:length])
            
            chunk_new_rows.append(new_row)

    # 4. 将当前块的处理结果写入文件
    if chunk_new_rows:
        aug_chunk_df = pd.DataFrame(chunk_new_rows)
        total_augmented_rows += len(aug_chunk_df)
        
        # 核心：如果是第一次写入，带上表头(header=True)；之后则追加并关掉表头(header=False)
        aug_chunk_df.to_csv(
            OUTPUT_FILE, 
            mode='a', 
            index=False, 
            header=first_chunk, 
            quotechar='"'
        )
        first_chunk = False
        
    # 主动释放内存（可选，Python 会自动处理，但显式清空列表有助于降低峰值）
    del chunk_new_rows

print(f"\n处理完成！")
print(f"原始用户总数: {total_original_users}")
print(f"增强后总样本数: {total_augmented_rows}")
print(f"文件已保存至: {OUTPUT_FILE}")