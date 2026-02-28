import pandas as pd
import os

data_home = os.path.join(os.path.dirname(__file__), '../tmp/kuai/data')
fnames = ['log_random_4_22_to_5_08_1k.csv',
          'log_standard_4_08_to_4_21_1k.csv',
          'log_standard_4_22_to_5_08_1k.csv']

data = []
for fname in fnames:
    data.append(pd.read_csv(os.path.join(data_home, fname)))

data = pd.concat(data)

print(data.shape)
print(data.groupby('is_click').size())
print(len(data.user_id.unique()))
print(data.video_id.max())

data = data.sort_values(by=['user_id', 'time_ms'])
result = data.groupby('user_id').agg({
    'video_id': lambda x: ','.join(map(str, x)),
    'is_click': lambda x: ','.join(map(str, x)),
    'time_ms': lambda x: ','.join(map(str, x))
}).reset_index()
result.columns = ['user_id', 'sequence_item_ids', 'sequence_ratings', 'sequence_timestamps']
result['sequence_ratings'] = result['sequence_ratings'].apply(lambda x: x.replace('1', '2').replace('0', '1'))
result.to_csv(os.path.join(data_home, 'sasrec_format_binary.csv'), index=False)