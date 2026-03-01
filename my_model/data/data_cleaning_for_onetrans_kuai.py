import pandas as pd
from tqdm import tqdm
import os

# ================= 配置 =================
CHUNK_SIZE = 5000  # 每次处理 5000 行
TASKS = [
    {
        'input': 'tmp/kuai/data/sasrec_format_binary_augment.csv',
        'output': 'tmp/kuai/data/sasrec_format_binary_augment_onetrans.csv'
    },
    {
        'input': 'tmp/kuai/data/sasrec_format_binary.csv',
        'output': 'tmp/kuai/data/sasrec_format_binary_onetrans_testset.csv'
    }
]
# =======================================

def filter_high_ratings(row):
    """提取评分为 '2' 的物品及其时间戳"""
    try:
        ratings = str(row['sequence_ratings']).split(',')
        items = str(row['sequence_item_ids']).split(',')
        times = str(row['sequence_timestamps']).split(',')
        
        # 逻辑：排除最后一个（通常是 Target），过滤前面评分等于 '2' 的索引
        high_indices = [i for i, r in enumerate(ratings[:-1]) if r == '2']
        
        high_items = [items[i] for i in high_indices]
        high_times = [times[i] for i in high_indices]
        
        return ",".join(high_items), ",".join(high_times)
    except Exception:
        return "", ""

def process_file(input_path, output_path):
    print(f"\n正在处理: {input_path} -> {output_path}")
    
    # 如果输出文件已存在，先删除
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # 1. 分块读取
    reader = pd.read_csv(input_path, chunksize=CHUNK_SIZE)
    first_chunk = True
    
    for chunk in tqdm(reader, desc="Processing chunks"):
        # 2. 应用过滤逻辑
        # 使用 zip(*) 将返回的元组拆分成两列
        chunk['high_item_ids'], chunk['high_timestamps'] = zip(*chunk.apply(filter_high_ratings, axis=1))
        
        # 3. 边计算边保存
        chunk.to_csv(
            output_path, 
            mode='a', 
            index=False, 
            header=first_chunk, 
            quotechar='"'
        )
        first_chunk = False

# 执行任务
for task in TASKS:
    if os.path.exists(task['input']):
        process_file(task['input'], task['output'])
    else:
        print(f"跳过：找不到输入文件 {task['input']}")

print("\n所有任务已完成！")