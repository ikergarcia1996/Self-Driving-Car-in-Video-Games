#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=eval.out
#SBATCH --error=eval.err

source /ikerlariak/igarcia945/envs/pytorch-tximista/bin/activate

cd ../../

python3 eval.py \
  --checkpoint_path models/tedd_1104_large_adafactor_no_mask/epoch=4-step=168763.ckpt \
  --batch_size 32 \
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
  --output_path models/tedd_1104_large_adafactor_no_mask/results.tsv \
  --experiment_name large_adafactor_no_mask



python3 eval.py \
  --checkpoint_path models/tedd_1104_large_adafactor/epoch=4-step=168763.ckpt \
  --batch_size 32 \
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
  --output_path models/tedd_1104_large_adafactor/results.tsv \
  --experiment_name large_adafactor


python3 eval.py \
--checkpoint_path models/tedd_1104_large_adamw/epoch=2-step=119126.ckpt \
--batch_size 32 \
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
--output_path models/tedd_1104_large_adamw/results.tsv \
--experiment_name large_adamw


########## LAST EPOCH



python3 eval.py \
  --checkpoint_path models/tedd_1104_large_adafactor_no_mask/epoch=7-last.ckpt \
  --batch_size 32 \
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
  --output_path models/tedd_1104_large_adafactor_no_mask/results_last.tsv \
  --experiment_name large_adafactor_no_mask_last



python3 eval.py \
  --checkpoint_path models/tedd_1104_large_adafactor/epoch=11-last.ckpt \
  --batch_size 32 \
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
  --output_path models/tedd_1104_large_adafactor/results_last.tsv \
  --experiment_name large_adafactor_last


python3 eval.py \
--checkpoint_path models/tedd_1104_large_adamw/epoch=14-last.ckpt \
--batch_size 32 \
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
--output_path models/tedd_1104_large_adamw/results_last.tsv \
--experiment_name large_adamw_last


