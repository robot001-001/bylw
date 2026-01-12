git pull
clear
CUDA_LAUNCH_BLOCKING=1 \
python main.py \
    --logging_dir log/ \
    --model HSTU \
    --mode dev \
    --dataset_name 'ml-1m' \
    --use_binary_ratings True \
    --num_ratings 2 \
    --max_seq_len 200 \
    --embedding_dim 50 \
    --positional_sampling_ratio 1.0 \
    --train_batch_size 2 \
    --accum_steps 1 \
    --eval_batch_size 128 \
    --device 'cuda:0' \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --eval_interval 20 \
    --model_args '{"max_seq_len": 200, "embedding_dim": 50, "dropout_rate": 0.2, "num_ratings": 2, "linear_dim": 25, "attention_dim": 25, "normalization": "rel_bias", "linear_config": "uvqk", "linear_activation": "silu", "num_blocks": 8, "num_heads": 2, "linear_dropout_rate": 0.2, "attn_dropout_rate": 0.0, "main_tower_units": [128, 2], "concat_ua": false, "enable_relative_attention_bias": true}'