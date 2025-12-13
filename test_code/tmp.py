from generative_recommenders.research.data.reco_dataset import get_reco_dataset
from generative_recommenders.research.trainer.data_loader import create_data_loader
from generative_recommenders.research.modeling.sequential.features import (
    movielens_seq_features_from_row,
)



def main():
    batch_size = 1
    dataset = get_reco_dataset(
        dataset_name="ml-1m",
        max_sequence_length=200,
        chronological=True,
        positional_sampling_ratio=1.0,
    )
    train_data_sampler, train_data_loader = create_data_loader(
        dataset.train_dataset,
        batch_size=batch_size,
        world_size=1,
        rank=0,
        shuffle=True,
        drop_last=1 > 1,
    )
    eval_data_sampler, eval_data_loader = create_data_loader(
        dataset.eval_dataset,
        batch_size=batch_size,
        world_size=1,
        rank=0,
        shuffle=True,  # needed for partial eval
        drop_last=1 > 1,
    )
    batch_id = 0
    epoch = 0
    for epoch in range(3):
        if train_data_sampler is not None:
            train_data_sampler.set_epoch(epoch)
        if eval_data_sampler is not None:
            eval_data_sampler.set_epoch(epoch)
        for row in iter(eval_data_loader):
            seq_features, target_ids, target_ratings = movielens_seq_features_from_row(
                row,
                device=0,
                max_output_length=10 + 1,
            )
            print(f'row:\n{row}\n\n\nseq_features:\n{seq_features}\n\n\ntarget_ids:\n{target_ids}\n\n\ntarget_ratings:\n{target_ratings}')
            break





if __name__ == '__main__':
    main()