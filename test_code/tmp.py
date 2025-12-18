from generative_recommenders.research.data.reco_dataset import get_reco_dataset
from generative_recommenders.research.trainer.data_loader import create_data_loader
from generative_recommenders.research.modeling.sequential.features import (
    movielens_seq_features_from_row,
)
from generative_recommenders.research.data.dataset import DatasetV2


def main():
    batch_size = 1
    dataset = DatasetV2(
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