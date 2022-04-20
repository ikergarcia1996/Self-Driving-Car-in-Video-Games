#!/bin/bash
#SBATCH --job-name=large
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --output=large.out
#SBATCH --error=large.err

source /ikerlariak/igarcia945/envs/pytorch-tximista/bin/activate

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

cd ../../

python3 train.py --train_new \
  --train_dir ../gtaai_datasets/train \
  --val_dir  ../gtaai_datasets/dev \
  --output_dir models/tedd_1104_large \
  --encoder_type transformer \
  --dataloader_num_workers 32 \
  --batch_size 32 \
  --accumulation_steps 1 \
  --max_epochs 10 \
  --cnn_model_name efficientnet_v2_l \
  --num_layers_encoder 6 \
  --embedded_size 512 \
  --learning_rate 5e-5 \
  --mask_prob 0.2 \
  --dropout_cnn_out 0.3 \
  --dropout_encoder 0.1 \
  --dropout_encoder_features 0.3 \
  --control_mode keyboard \
  --val_check_interval 0.5 \
  --precision "bf16" \
  --devices 4 \
  --strategy "ddp_find_unused_parameters_false"


