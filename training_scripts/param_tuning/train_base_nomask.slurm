#!/bin/bash
#SBATCH --job-name=model_NOmask
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --output=model_NOmask.out
#SBATCH --error=model_NOmask.err

source /ikerlariak/igarcia945/envs/pytorch-tximista/bin/activate

cd ../../

python3 train.py --train_new \
  --train_dir ../gtaai_datasets/train \
  --val_dir  ../gtaai_datasets/dev \
  --output_dir models/tedd_1104_base_NOmask \
  --encoder_type transformer \
  --batch_size 32 \
  --accumulation_steps 2 \
  --max_epochs 5 \
  --cnn_model_name efficientnet_v2_m \
  --num_layers_encoder 4 \
  --embedded_size 512 \
  --learning_rate 5e-5 \
  --mask_prob 0.0 \
  --dropout_cnn_out 0.3 \
  --dropout_encoder 0.1 \
  --dropout_encoder_features 0.3 \
  --control_mode keyboard \
  --val_check_interval 0.5 \
  --precision "bf16" \
  --devices 2 \
  --strategy "ddp"


