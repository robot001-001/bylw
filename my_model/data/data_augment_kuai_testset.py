import pandas as pd
import os

# 定义路径
INPUT_FILE = 'tmp/kuai/data/sasrec_format_binary.csv'
OUTPUT_FILE = 'tmp/kuai/data/sasrec_format_binary_testset.csv'

# 读取原始数据
data = pd.read_csv(INPUT_FILE)

def remove_last_element(s):
    """移除逗号分隔字符串的最后一个元素"""
    if pd.isna(s) or not isinstance(s, str):
        return s
    parts = s.split(',')
    if len(parts) > 1:
        return ','.join(parts[:-1])
    return ""  # 若只剩一个元素，截断后返回空

def augment_data(df, iterations=5):
    """
    数据增强逻辑：
    每次从当前数据集中截断最后一个元素，并追加到结果集中。
    """
    all_frames = [df]
    current_df = df.copy()
    
    cols_to_process = ['sequence_item_ids', 'sequence_ratings', 'sequence_timestamps']
    
    for _ in range(iterations):
        # 复制当前最新的一批数据进行截断
        next_df = current_df.copy()
        
        for col in cols_to_process:
            next_df[col] = next_df[col].apply(remove_last_element)
            
        # 过滤掉序列已经变为空的行（长度不足以截断的情况）
        next_df = next_df[next_df['sequence_item_ids'] != ""].copy()
        
        if next_df.empty:
            break
            
        all_frames.append(next_df)
        # 将本次截断的结果作为下一次截断的输入
        current_df = next_df
        
    # 合并所有增强后的数据
    return pd.concat(all_frames, ignore_index=True)

# 执行增强处理
augmented_data = augment_data(data, iterations=5)

# 保存结果
# 确保输出目录存在
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
augmented_data.to_csv(OUTPUT_FILE, index=False)

print(f"处理完成！")
print(f"原始行数: {len(data)}")
print(f"增强后的总行数: {len(augmented_data)}")