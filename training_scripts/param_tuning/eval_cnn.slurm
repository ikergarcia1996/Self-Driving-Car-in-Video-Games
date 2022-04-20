#!/bin/bash
#SBATCH --job-name=arqmax_eval
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --output=arqmax_eval.out
#SBATCH --error=arqmax_eval.err

source /ikerlariak/igarcia945/envs/pytorch-tximista/bin/activate

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

cd ../../

python3 eval.py \
  --checkpoint_path models/convnext_large/TEDD1104/33puavzl/checkpoints/epoch=0-step=19854.ckpt \
  --batch_size 32 \
  --report_to wandb \
  --experiment_name convnext_large \
  --test_dirs \
   /ikerlariak/igarcia945/gtaai_datasets/dev \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_city_day_clear \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_city_day_rain \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_city_night_clear \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_city_night_rain \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_highway_day_clear \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_highway_day_rain \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_highway_night_clear \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_highway_night_rain \
  --output_path models/convnext_large/results.tsv

python3 eval.py \
  --checkpoint_path models/efficientnet_b4/TEDD1104/24rvjdoh/checkpoints/epoch=0-step=19854.ckpt \
  --batch_size 32 \
  --report_to wandb \
  --experiment_name efficientnet_b4 \
  --test_dirs \
   /ikerlariak/igarcia945/gtaai_datasets/dev \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_city_day_clear \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_city_day_rain \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_city_night_clear \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_city_night_rain \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_highway_day_clear \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_highway_day_rain \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_highway_night_clear \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_highway_night_rain \
  --output_path models/efficientnet_b4/results.tsv

python3 eval.py \
  --checkpoint_path models/efficientnet_b7/TEDD1104/2l98lu95/checkpoints/epoch=0-step=19854.ckpt \
  --batch_size 32 \
  --report_to wandb \
  --experiment_name efficientnet_b7 \
  --test_dirs \
   /ikerlariak/igarcia945/gtaai_datasets/dev \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_city_day_clear \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_city_day_rain \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_city_night_clear \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_city_night_rain \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_highway_day_clear \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_highway_day_rain \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_highway_night_clear \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_highway_night_rain \
  --output_path models/efficientnet_b7/results.tsv

python3 eval.py \
  --checkpoint_path models/efficientnet_v2_l/TEDD1104/3at1f7wa/checkpoints/epoch=0-step=19854.ckpt \
  --batch_size 32 \
  --report_to wandb \
  --experiment_name efficientnet_v2_l \
  --test_dirs \
   /ikerlariak/igarcia945/gtaai_datasets/dev \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_city_day_clear \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_city_day_rain \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_city_night_clear \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_city_night_rain \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_highway_day_clear \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_highway_day_rain \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_highway_night_clear \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_highway_night_rain \
  --output_path models/efficientnet_v2_l/results.tsv


python3 eval.py \
  --checkpoint_path models/efficientnet_v2_m/TEDD1104/sgwhvgcd/checkpoints/epoch=0-step=19854.ckpt \
  --batch_size 32 \
  --report_to wandb \
  --experiment_name efficientnet_v2_m \
  --test_dirs \
   /ikerlariak/igarcia945/gtaai_datasets/dev \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_city_day_clear \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_city_day_rain \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_city_night_clear \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_city_night_rain \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_highway_day_clear \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_highway_day_rain \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_highway_night_clear \
   /ikerlariak/igarcia945/gtaai_datasets/test/car_highway_night_rain \
  --output_path models/efficientnet_v2_m/results.tsv



