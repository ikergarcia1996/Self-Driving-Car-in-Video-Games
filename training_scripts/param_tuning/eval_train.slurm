#!/bin/bash
#SBATCH --job-name=eval_train
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=evaluate_train.out
#SBATCH --error=evaluate_train.err

source /ikerlariak/igarcia945/envs/pytorch-tximista/bin/activate

cd ../../

python3 eval.py \
  --checkpoint_path models/tedd_1104_XXL/epoch=1-step=79417.ckpt \
  --batch_size 64 \
  --test_dirs \
   /ikerlariak/igarcia945/gtaai_datasets/train \
  --output_path models/tedd_1104_XXL/XXL_eval_train.tsv \
  --experiment_name XXL_train


python3 eval.py \
  --checkpoint_path models/tedd_1104_XXL/epoch=19-last.ckpt \
  --batch_size 64 \
  --test_dirs \
   /ikerlariak/igarcia945/gtaai_datasets/train \
  --output_path models/tedd_1104_XXL/XXL_last_eval_train.tsv \
  --experiment_name XXL_last_train


python3 eval.py \
  --checkpoint_path models/tedd_1104_M/epoch=3-step=138981.ckpt \
  --batch_size 64 \
  --test_dirs \
   /ikerlariak/igarcia945/gtaai_datasets/train \
  --output_path models/tedd_1104_M/M_eval_train.tsv \
  --experiment_name M_train

  python3 eval.py \
  --checkpoint_path models/tedd_1104_S/epoch=4-step=198544.ckpt \
  --batch_size 64 \
  --test_dirs \
   /ikerlariak/igarcia945/gtaai_datasets/train \
  --output_path models/tedd_1104_S/S_eval_train.tsv \
  --experiment_name S_train
