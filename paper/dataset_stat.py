import pandas as pd
import os

data_home = '../my_model/tmp/'

dataset_names = ['ml-1m', 'kuai/data']
sasrec_format = 'sasrec_format_binary.csv'
augment = 'sasrec_format_binary_augment_onetrans.csv'

def stat(dataset_name):
    data_seq = pd.read_csv(os.path.join(data_home, dataset_name, sasrec_format))
    data_seq['train_ratings'] = data_seq['sequence_ratings'].apply(lambda x: x[:-2])
    data_seq['test_ratings'] = data_seq['sequence_ratings'].apply(lambda x: x[-1])
    counts = data_seq['train_ratings'].str.split(',').explode().value_counts()
    counts_item = data_seq['sequence_item_ids'].str.split(',').explode().value_counts()
    print(f'train_num: {data_seq.shape[0]}')
    print(f'train_pos: {counts.get("2", 0)}')
    print(f'train_neg: {counts.get("1", 0)}')
    print(f'train_num_users: {data_seq.shape[0]}')
    print(f'train_num_items: {len(counts_item.keys())}')

    print(f'split_num: {counts.get("2", 0)+counts.get("1", 0)}')
    print(f'split_pos: {counts.get("2", 0)}')
    print(f'split_neg: {counts.get("1", 0)}')
    print(f'split_num_users: {data_seq.shape[0]}')
    print(f'split_num_items: {len(counts_item.keys())}')


    counts = data_seq['test_ratings'].str.split(',').explode().value_counts()        
    print(f'test_num: {data_seq.shape[0]}')
    print(f'test_pos: {counts.get("2", 0)}')
    print(f'test_neg: {counts.get("1", 0)}')
    print(f'test_num_users: {data_seq.shape[0]}')
    print(f'train_num_items: {len(counts_item.keys())}')


for dataset_name in dataset_names:
    print('-'*100)
    stat(dataset_name)