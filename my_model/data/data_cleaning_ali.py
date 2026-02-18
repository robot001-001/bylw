import pandas as pd
import os

data_home = os.path.join(os.path.dirname(__file__), '../tmp/hm')
fname = os.path.join(data_home, 'transactions_train.csv')
out_fname = os.path.join(data_home, 'sasrec_format_binary.csv')

data = pd.read_csv(fname)
print(data.head())

# data = pd.read_csv(fname, header=None)
# data.columns = ['user_id', 'item_id', 'cate_id', 'behavior', 'timestamp']

# buy 一个item并不会产生 pv 该 item
# print(data.columns)
# print(data.shape)
# print(data['behavior'].unique())
# print(data[(data.user_id==100)&(data.item_id==1603476)])

