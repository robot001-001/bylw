# 前置：data_cleaning.py
# 前置：data_augment.py
import pandas as pd

# 1. 加载数据
data_home = 'tmp/ml-1m/sasrec_format_binary_augment.csv'
data = pd.read_csv(data_home)

# 定义处理函数
def filter_high_ratings(row):
    ratings = str(row['sequence_ratings']).split(',')
    items = str(row['sequence_item_ids']).split(',')
    times = str(row['sequence_timestamps']).split(',')
    high_indices = [i for i, r in enumerate(ratings[:-1]) if r == '2']
    high_items = [items[i] for i in high_indices]
    high_times = [times[i] for i in high_indices]
    return ",".join(high_items), ",".join(high_times)


data['high_item_ids'], data['high_timestamps'] = zip(*data.apply(filter_high_ratings, axis=1))


print("处理后的列名:", data.columns)
print("\n数据样例（高评分过滤后）:")
print(data[['high_item_ids', 'high_timestamps']].head())


data.to_csv('tmp/ml-1m/sasrec_format_binary_augment_onetrans.csv', index=False)