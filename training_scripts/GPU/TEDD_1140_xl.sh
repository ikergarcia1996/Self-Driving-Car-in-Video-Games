#!/bin/bash
#SBATCH --job-name=TEDD_XL
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --output=TEDD_XL.out
#SBATCH --error=TEDD_XL.err

source /ikerlariak/igarcia945/envs/pytorch-tximista/bin/activate

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export OMP_NUM_THREADS=16

cd ../../

python3 train.py --train_new \
  --train_dir ../gtaai_datasets/train \
  --val_dir  ../gtaai_datasets/dev \
  --output_dir models/tedd_1104_XL \
  --encoder_type transformer \
  --dataloader_num_workers 32 \
  --batch_size 16 \
  --accumulation_steps 1 \
  --max_epochs 20 \
  --cnn_model_name efficientnet_v2_l \
  --num_layers_encoder 4 \
  --embedded_size 512 \
  --learning_rate 1e-5 \
  --optimizer_name adafactor \
  --scheduler_name cosine \
  --warmup_factor 0.05 \
  --mask_prob 0.2 \
  --hide_map_prob 0.0 \
  --dropout_cnn_out 0.3 \
  --dropout_encoder 0.1 \
  --dropout_encoder_features 0.3 \
  --label_smoothing 0.1 \
  --control_mode keyboard \
  --val_check_interval 0.25 \
  --precision "16"  \
  --devices 2 \
  --strategy "ddp_find_unused_parameters_false"


