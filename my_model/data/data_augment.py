# 前置：data_cleaning.py
import pandas as pd
from tqdm import tqdm

# ================= 配置 =================
INPUT_FILE = 'tmp/ml-1m/sasrec_format_binary.csv'
OUTPUT_FILE = 'tmp/ml-1m/sasrec_format_binary_augment.csv'
MIN_SEQ_LEN = 5     # 只有长度 >= 5 的序列才会被保存（防止冷启动数据噪音）
# =======================================

print(f"正在读取数据: {INPUT_FILE} ...")
data = pd.read_csv(INPUT_FILE)

# 获取列名，确保输出列名完全一致
columns = data.columns.tolist()
new_rows = []

print("正在进行数据增强 (Sliding Window)...")

for _, row in tqdm(data.iterrows(), total=data.shape[0]):
    # 1. 解析基础数据
    try:
        # 转成列表
        full_items = [x for x in str(row['sequence_item_ids']).split(',')]
        full_ratings = [x for x in str(row['sequence_ratings']).split(',')]
        full_timestamps = [x for x in str(row['sequence_timestamps']).split(',')]
    except Exception as e:
        print(f"解析错误 UserID {row.get('user_id')}: {e}")
        continue

    seq_len = len(full_items)
    
    # 2. 滑动窗口切分
    # 从 MIN_SEQ_LEN 开始，逐步增加长度，直到完整长度
    # range(5, 100) -> 切片长度为 5, 6, 7 ... 99
    # 你的Dataset逻辑是：取最后一个做Target，前面做History
    # 所以切片长度为 N 时，Input长度为 N-1
    
    for length in range(MIN_SEQ_LEN, seq_len):
        
        # 构造这一行的新数据
        # 使用 .copy() 保持元数据 (sex, age, zip_code 等) 不变
        new_row = row.copy()
        
        # 截取对应长度的序列
        sub_items = full_items[:length]
        sub_ratings = full_ratings[:length]
        sub_timestamps = full_timestamps[:length]
        
        # 覆盖原来的序列列，保持格式依然是逗号分隔的字符串
        new_row['sequence_item_ids'] = ",".join(sub_items)
        new_row['sequence_ratings'] = ",".join(sub_ratings)
        new_row['sequence_timestamps'] = ",".join(sub_timestamps)
        
        # index 列如果不重要可以重置，或者保持原样（会有重复index）
        # 建议不管 index 列，pandas 保存时会处理
        
        new_rows.append(new_row)

# 3. 生成新的 DataFrame
aug_data = pd.DataFrame(new_rows, columns=columns)

print(f"原始用户数: {len(data)}")
print(f"增强后样本数: {len(aug_data)}")

# 4. 保存 (保持原格式)
# quotechar='"' 确保列表字符串被正确包裹，避免解析错误
aug_data.to_csv(OUTPUT_FILE, index=False, quotechar='"')
print(f"文件已保存至: {OUTPUT_FILE}")

# 5. 验证一下结构
print("\n数据样例 (前3行):")
print(aug_data[['user_id', 'sequence_item_ids']].head(3))
