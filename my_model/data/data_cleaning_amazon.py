# 将sasrec_format.csv转化成binary
import pandas as pd

data = pd.read_csv('tmp/amzn_books/sasrec_format.csv')
sequence_ratings = data['sequence_ratings'].to_list()
sequence_ratings = [i.replace('1.0', '1').replace('2.0', '1').replace('3.0', '1').replace('4.0', '1').replace('5.0', '2') for i in sequence_ratings]
data['sequence_ratings'] = sequence_ratings

# print(data['sequence_ratings'].sample(1))
data.to_csv('tmp/amzn_books/sasrec_format_binary.csv', index=False)
