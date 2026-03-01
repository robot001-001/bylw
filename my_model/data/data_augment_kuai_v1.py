import pandas as pd
import os
import gc  # 引入垃圾回收模块
from tqdm import tqdm

# ================= 配置 =================
INPUT_FILE = 'tmp/kuai/data/sasrec_format_binary.csv'
OUTPUT_FILE = 'tmp/kuai/data/sasrec_format_binary_augment.csv'
MIN_SEQ_LEN = 5     # 只有长度 >= 5 的序列才会被保存
CHUNK_SIZE = 1000   # 每次读取 1000 个用户进行处理，可根据内存大小调整
# 新增：每次写入的批次大小（避免频繁IO，又不累积太多数据）
WRITE_BATCH_SIZE = 5000  
# =======================================

# 如果输出文件已存在，先删除
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

print(f"正在开始分块处理数据: {INPUT_FILE} ...")

# 1. 初始化读取器和统计变量
reader = pd.read_csv(
    INPUT_FILE, 
    chunksize=CHUNK_SIZE,
    # 显式指定列类型，减少Pandas自动推断的内存占用
    dtype={
        'sequence_item_ids': str,
        'sequence_ratings': str,
        'sequence_timestamps': str
    }
)
first_write = True
total_original_users = 0
total_augmented_rows = 0

# 2. 分块处理核心逻辑
for chunk in tqdm(reader, desc="Processing Chunks"):
    # 临时存储待写入的行，达到批次大小就写入
    batch_rows = []
    total_original_users += len(chunk)
    
    for _, row in chunk.iterrows():
        try:
            # 解析数据（提前过滤空值）
            item_str = str(row['sequence_item_ids']).strip()
            rating_str = str(row['sequence_ratings']).strip()
            ts_str = str(row['sequence_timestamps']).strip()
            
            # 跳过空序列
            if not item_str or item_str == 'nan':
                continue
                
            full_items = item_str.split(',')
            full_ratings = rating_str.split(',')
            full_timestamps = ts_str.split(',')
            
            # 校验三个序列长度一致（避免数据异常）
            if len(full_items) != len(full_ratings) or len(full_items) != len(full_timestamps):
                continue
                
            seq_len = len(full_items)
            if seq_len < MIN_SEQ_LEN:
                continue
                
            # 3. 滑动窗口扩增（边扩增边加入批次）
            base_dict = row.to_dict()
            for length in range(MIN_SEQ_LEN, seq_len + 1):
                new_row = base_dict.copy()
                new_row['sequence_item_ids'] = ",".join(full_items[:length])
                new_row['sequence_ratings'] = ",".join(full_ratings[:length])
                new_row['sequence_timestamps'] = ",".join(full_timestamps[:length])
                
                batch_rows.append(new_row)
                
                # 达到批次大小，立即写入并清空
                if len(batch_rows) >= WRITE_BATCH_SIZE:
                    # 转为DataFrame并写入
                    batch_df = pd.DataFrame(batch_rows)
                    batch_df.to_csv(
                        OUTPUT_FILE,
                        mode='a',
                        index=False,
                        header=first_write,
                        quotechar='"',
                        # 优化写入性能
                        chunksize=1000
                    )
                    # 更新标记和统计
                    total_augmented_rows += len(batch_df)
                    first_write = False
                    # 清空批次并释放内存
                    batch_rows.clear()
                    del batch_df
                    gc.collect()  # 强制触发垃圾回收
                    
        except Exception as e:
            # 打印异常但不终止程序
            print(f"处理行数据时出错: {e}")
            continue
    
    # 处理当前chunk剩余的批次数据
    if batch_rows:
        batch_df = pd.DataFrame(batch_rows)
        batch_df.to_csv(
            OUTPUT_FILE,
            mode='a',
            index=False,
            header=first_write,
            quotechar='"',
            chunksize=1000
        )
        total_augmented_rows += len(batch_df)
        first_write = False
        del batch_df
        batch_rows.clear()
    
    # 强制释放当前chunk的内存
    del chunk
    gc.collect()

# ================= 最终统计 =================
print(f"\n处理完成！")
print(f"原始用户总数: {total_original_users}")
print(f"增强后总样本数: {total_augmented_rows}")
print(f"文件已保存至: {OUTPUT_FILE}")