# 将sasrec_format.csv转化成binary
import pandas as pd

data = pd.read_csv('tmp/ml-1m/sasrec_format.csv')
sequence_ratings = data['sequence_ratings'].to_list()
sequence_ratings = [i.replace('1', '1').replace('2', '1').replace('3', '1').replace('4', '2').replace('5', '2') for i in sequence_ratings]
data['sequence_ratings'] = sequence_ratings

# print(data['sequence_ratings'].sample(1))
data.to_csv('tmp/ml-1m/sasrec_format_binary.csv', index=False)
