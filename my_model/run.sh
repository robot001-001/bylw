python main.py \
    --logging_dir bylw/log/ \
    --mode dev \
    --dataset_name 'ml-20m' \
    --max_sequence_length 200 \
    --positional_sampling_ratio 1.0 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --device 'cuda:0'