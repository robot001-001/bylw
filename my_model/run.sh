git pull
clear
CUDA_LAUNCH_BLOCKING=1 \
python main.py \
    --logging_dir log/ \
    --mode dev \
    --dataset_name 'ml-1m' \
    --max_seq_len 200 \
    --embedding_dim 50 \
    --positional_sampling_ratio 1.0 \
    --train_batch_size 5 \
    --eval_batch_size 1 \
    --device 'cuda:0' \
    --num_epochs 3 \
    --eval_interval 20 \
    --model_args '{"max_seq_len": 200, "embedding_dim": 50, "dropout_rate": 0.2, "num_ratings": 5, "linear_dim": 64, "attention_dim": 64, "normalization": "rel_bias", "linear_config": "uvqk", "linear_activation": "silu", "num_blocks": 2, "num_heads": 1, "linear_dropout_rate": 0.0, "attn_dropout_rate": 0.0, "main_tower_units": [128, 5], "concat_ua": false, "enable_relative_attention_bias": true}'