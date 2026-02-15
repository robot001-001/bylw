git pull
clear
# CUDA_LAUNCH_BLOCKING=1 \
# python main.py \
#     --logging_dir log/exp_log/baseline/ \
#     --logging_file baseline_hstu_ml.log \
#     --trainer_type HSTUBaseTrainer \
#     --model HSTU_pretrain \
#     --mode dev \
#     --dataset_name 'ml-1m' \
#     --use_binary_ratings True \
#     --num_ratings 2 \
#     --max_seq_len 200 \
#     --embedding_dim 50 \
#     --positional_sampling_ratio 1.0 \
#     --train_batch_size 512 \
#     --accum_steps 1 \
#     --eval_batch_size 512 \
#     --device 'cuda:0' \
#     --learning_rate 3e-4 \
#     --num_epochs 100 \
#     --eval_interval 20 \
#     --model_args '{"max_seq_len": 200, "embedding_dim": 50, "dropout_rate": 0.2, "num_ratings": 2, "linear_dim": 25, "attention_dim": 25, "normalization": "rel_bias", "linear_config": "uvqk", "linear_activation": "silu", "num_blocks": 8, "num_heads": 2, "linear_dropout_rate": 0.2, "attn_dropout_rate": 0.0, "main_tower_units": [128, 2], "concat_ua": false, "enable_relative_attention_bias": true}'

# CUDA_LAUNCH_BLOCKING=1 \
# python main.py \
#     --logging_dir log/exp_log/ablation/ \
#     --logging_file ablation_nsa_ml.log \
#     --trainer_type HSTUBaseTrainer \
#     --model HSTU_nsa_pretrain \
#     --mode dev \
#     --dataset_name 'ml-1m' \
#     --use_binary_ratings True \
#     --num_ratings 2 \
#     --max_seq_len 200 \
#     --embedding_dim 50 \
#     --positional_sampling_ratio 1.0 \
#     --train_batch_size 64 \
#     --accum_steps 8 \
#     --eval_batch_size 64 \
#     --device 'cuda:0' \
#     --learning_rate 1e-4 \
#     --num_epochs 100 \
#     --eval_interval 20 \
#     --model_args '{"max_seq_len": 200, "embedding_dim": 50, "dropout_rate": 0.2, "num_ratings": 2, "linear_dim": 25, "attention_dim": 25, "normalization": "rel_bias", "linear_config": "uvqk", "linear_activation": "silu", "num_blocks": 8, "num_heads": 2, "linear_dropout_rate": 0.2, "attn_dropout_rate": 0.0, "main_tower_units": [128, 2], "concat_ua": false, "enable_relative_attention_bias": true}'

# CUDA_LAUNCH_BLOCKING=1 \
# python main.py \
#     --logging_dir log/exp_log/ablation/ \
#     --logging_file ablation_nsa_interleave_ml.log \
#     --trainer_type HSTUBaseTrainer \
#     --model HSTU_interleave_pretrain \
#     --mode dev \
#     --dataset_name 'ml-1m' \
#     --use_binary_ratings True \
#     --num_ratings 2 \
#     --max_seq_len 200 \
#     --embedding_dim 50 \
#     --positional_sampling_ratio 1.0 \
#     --train_batch_size 128 \
#     --accum_steps 4 \
#     --eval_batch_size 128 \
#     --device 'cuda:0' \
#     --learning_rate 1e-4 \
#     --num_epochs 100 \
#     --eval_interval 20 \
#     --model_args '{"max_seq_len": 200, "embedding_dim": 50, "dropout_rate": 0.2, "num_ratings": 2, "linear_dim": 25, "attention_dim": 25, "normalization": "rel_bias", "linear_config": "uvqk", "linear_activation": "silu", "num_blocks": 8, "num_heads": 2, "linear_dropout_rate": 0.2, "attn_dropout_rate": 0.0, "main_tower_units": [128, 2], "concat_ua": false, "enable_relative_attention_bias": true}'

# CUDA_LAUNCH_BLOCKING=1 \
# python main.py \
#     --logging_dir log/hstu_fuxian/ \
#     --trainer_type HSTUBaseTrainer \
#     --model HSTU_fuxian \
#     --mode dev \
#     --dataset_name 'ml-1m' \
#     --use_binary_ratings True \
#     --num_ratings 2 \
#     --max_seq_len 200 \
#     --embedding_dim 50 \
#     --positional_sampling_ratio 1.0 \
#     --train_batch_size 1024 \
#     --accum_steps 1 \
#     --eval_batch_size 128 \
#     --device 'cuda:0' \
#     --learning_rate 1e-4 \
#     --num_epochs 100 \
#     --eval_interval 20 \
#     --model_args '{"max_seq_len": 200, "embedding_dim": 50, "dropout_rate": 0.2, "num_ratings": 2, "linear_dim": 25, "attention_dim": 25, "normalization": "rel_bias", "linear_config": "uvqk", "linear_activation": "silu", "num_blocks": 8, "num_heads": 2, "linear_dropout_rate": 0.2, "attn_dropout_rate": 0.0, "main_tower_units": [128, 2], "concat_ua": false, "enable_relative_attention_bias": true}'


# CUDA_LAUNCH_BLOCKING=1 \
# python main.py \
#     --logging_dir log/exp_log/ablation/ \
#     --logging_file ablation_bsa_ml.log \
#     --trainer_type HSTUBaseTrainer \
#     --model HSTU_bsa_pretrain \
#     --mode dev \
#     --dataset_name 'ml-1m' \
#     --use_binary_ratings True \
#     --num_ratings 2 \
#     --max_seq_len 200 \
#     --embedding_dim 50 \
#     --positional_sampling_ratio 1.0 \
#     --train_batch_size 128 \
#     --accum_steps 4 \
#     --eval_batch_size 128 \
#     --device 'cuda:0' \
#     --learning_rate 3e-4 \
#     --num_epochs 100 \
#     --eval_interval 20 \
#     --model_args '{"max_seq_len": 200, "embedding_dim": 50, "dropout_rate": 0.2, "num_ratings": 2, "linear_dim": 25, "attention_dim": 25, "normalization": "rel_bias", "linear_config": "uvqk", "linear_activation": "silu", "num_blocks": 8, "num_heads": 2, "linear_dropout_rate": 0.2, "attn_dropout_rate": 0.0, "main_tower_units": [128, 2], "concat_ua": false, "enable_relative_attention_bias": true}'


# CUDA_LAUNCH_BLOCKING=1 \
python main.py \
    --logging_dir log/exp_log/exp/ \
    --logging_file exp_bsa_interleave_ml.log \
    --trainer_type HSTUBaseTrainer \
    --model HSTU_bsa_pretrain_interleave \
    --mode dev \
    --dataset_name 'ml-1m' \
    --use_binary_ratings True \
    --num_ratings 2 \
    --max_seq_len 200 \
    --embedding_dim 50 \
    --positional_sampling_ratio 1.0 \
    --train_batch_size 128 \
    --accum_steps 4 \
    --eval_batch_size 128 \
    --device 'cuda:0' \
    --learning_rate 3e-4 \
    --num_epochs 50 \
    --eval_interval 20 \
    --model_args '{"max_seq_len": 200, "embedding_dim": 50, "dropout_rate": 0.2, "num_ratings": 2, "linear_dim": 25, "attention_dim": 25, "normalization": "rel_bias", "linear_config": "uvqk", "linear_activation": "silu", "num_blocks": 8, "num_heads": 2, "linear_dropout_rate": 0.2, "attn_dropout_rate": 0.0, "main_tower_units": [128, 2], "concat_ua": false, "enable_relative_attention_bias": true}'




# CUDA_LAUNCH_BLOCKING=1 \
# python main.py \
#     --logging_dir log/hstu_bsa_interleave_presort/ \
#     --logging_file train.log \
#     --trainer_type HSTUBaseTrainer \
#     --model HSTU_bsa_pretrain_interleave \
#     --mode train_presort \
#     --presort_steps 10 \
#     --dataset_name 'ml-1m' \
#     --use_binary_ratings True \
#     --num_ratings 2 \
#     --max_seq_len 200 \
#     --embedding_dim 50 \
#     --positional_sampling_ratio 1.0 \
#     --train_batch_size 128 \
#     --accum_steps 4 \
#     --eval_batch_size 128 \
#     --device 'cuda:0' \
#     --learning_rate 3e-4 \
#     --num_epochs 50 \
#     --eval_interval 20 \
#     --model_args '{"max_seq_len": 200, "embedding_dim": 50, "dropout_rate": 0.2, "num_ratings": 2, "linear_dim": 25, "attention_dim": 25, "normalization": "rel_bias", "linear_config": "uvqk", "linear_activation": "silu", "num_blocks": 8, "num_heads": 2, "linear_dropout_rate": 0.2, "attn_dropout_rate": 0.0, "main_tower_units": [128, 2], "concat_ua": false, "enable_relative_attention_bias": true}'


# CUDA_LAUNCH_BLOCKING=1 \
# python main.py \
#     --logging_dir log/onetrans/ \
#     --logging_file train.log \
#     --trainer_type ONETRANSTrainer \
#     --model ONETRANS \
#     --mode train \
#     --dataset_name 'ml-1m' \
#     --use_binary_ratings True \
#     --num_ratings 2 \
#     --max_seq_len 200 \
#     --embedding_dim 64 \
#     --train_batch_size 2048 \
#     --accum_steps 1 \
#     --eval_batch_size 2048 \
#     --device 'cuda:0' \
#     --learning_rate 3e-4 \
#     --num_epochs 50 \
#     --eval_interval 20 \
#     --model_args '{"num_layers": 8, "max_seq_len": [512, 256, 128, 64, 32, 16, 8, 4], "ns_seq_len": 2, "d_model": 64, "num_heads": 2, "ffn_layer_hidden_dim": 128, "main_tower_units": [128, 2]}'


# CUDA_LAUNCH_BLOCKING=1 \
# python main.py \
#     --logging_dir log/rankmixer/ \
#     --logging_file train.log \
#     --trainer_type RANKMIXERTrainer \
#     --model RANKMIXER \
#     --mode train \
#     --dataset_name 'ml-1m' \
#     --use_binary_ratings True \
#     --num_ratings 2 \
#     --max_seq_len 200 \
#     --embedding_dim 64 \
#     --train_batch_size 2048 \
#     --accum_steps 1 \
#     --eval_batch_size 2048 \
#     --device 'cuda:0' \
#     --learning_rate 3e-4 \
#     --num_epochs 50 \
#     --eval_interval 20 \
#     --model_args '{"topk": 20, "max_seq_len": 200, "d_model": 64, "sim_main_tower_units": [512, 384], "num_blocks": 2, "num_heads": 8, "main_tower_units": [128, 2]}'




# CUDA_LAUNCH_BLOCKING=1 \
# python main.py \
#     --logging_dir log/hstu/ \
#     --logging_file baseline_hstu_am.log \
#     --trainer_type HSTUBaseTrainer \
#     --model HSTU_pretrain \
#     --mode dev \
#     --dataset_name 'amzn-books' \
#     --use_binary_ratings True \
#     --num_ratings 2 \
#     --max_seq_len 200 \
#     --embedding_dim 50 \
#     --positional_sampling_ratio 1.0 \
#     --train_batch_size 512 \
#     --accum_steps 1 \
#     --eval_batch_size 512 \
#     --device 'cuda:0' \
#     --learning_rate 3e-4 \
#     --num_epochs 100 \
#     --eval_interval 20 \
#     --model_args '{"max_seq_len": 200, "embedding_dim": 50, "dropout_rate": 0.2, "num_ratings": 2, "linear_dim": 25, "attention_dim": 25, "normalization": "rel_bias", "linear_config": "uvqk", "linear_activation": "silu", "num_blocks": 8, "num_heads": 2, "linear_dropout_rate": 0.2, "attn_dropout_rate": 0.0, "main_tower_units": [128, 2], "concat_ua": false, "enable_relative_attention_bias": true}'