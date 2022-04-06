#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --output=eval.out
#SBATCH --error=eval.err

source /ikerlariak/igarcia945/envs/pytorch-tximista/bin/activate

cd ../../

python3 eval.py \
  --checkpoint_path models/tedd_1104_base_NOmask/TEDD1104/tedd_1104_base_mask/checkpoints/epoch=0-step=19854.ckpt \
  --batch_size 8 \
  --test_dirs \
   /data/gtaai_datasets/dev \
   /data/gtaai_datasets/test/car_city_day_clear \
   /data/gtaai_datasets/test/car_city_day_rain \
   /data/gtaai_datasets/test/car_city_night_clear \
   /data/gtaai_datasets/test/car_city_night_rain \
   /data/gtaai_datasets/test/car_highway_day_clear \
   /data/gtaai_datasets/test/car_highway_day_rain \
   /data/gtaai_datasets/test/car_highway_night_clear \
   /data/gtaai_datasets/test/car_highway_night_rain \
  --output_path models/tedd_1104_base_NOmask/TEDD1104/tedd_1104_base_mask/results.tsv


  python3 eval.py \
  --checkpoint_path models/tedd_1104_base_NOmask/TEDD1104/tedd_1104_base_NOmask/checkpoints/epoch=4-step=89347.ckpt \
  --batch_size 8 \
  --test_dirs \
   /data/gtaai_datasets/dev \
   /data/gtaai_datasets/test/car_city_day_clear \
   /data/gtaai_datasets/test/car_city_day_rain \
   /data/gtaai_datasets/test/car_city_night_clear \
   /data/gtaai_datasets/test/car_city_night_rain \
   /data/gtaai_datasets/test/car_highway_day_clear \
   /data/gtaai_datasets/test/car_highway_day_rain \
   /data/gtaai_datasets/test/car_highway_night_clear \
   /data/gtaai_datasets/test/car_highway_night_rain \
  --output_path models/tedd_1104_base_NOmask/TEDD1104/tedd_1104_base_NOmask/results.tsv