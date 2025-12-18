from generative_recommenders.research.data.dataset import DatasetV2
from generative_recommenders.research.data.dataset_v3 import DatasetV3


def main():
    batch_size = 1
    dataset = DatasetV3(
        ratings_file="tmp/ml-1m/sasrec_format.csv",
        padding_length=200 + 1,  # target
        ignore_last_n=1,
        chronological=True,
        sample_ratio=1.0,
    )
    ret = dataset.__getitem__(255)
    print(ret)





if __name__ == '__main__':
    main()