#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=evaluate.out
#SBATCH --error=evaluate.err

source /ikerlariak/igarcia945/envs/pytorch-tximista/bin/activate

cd ../../




### 32 bits eval


python3 eval.py \
  --checkpoint_path models/tedd_1104_XXL/epoch=1-step=79417.ckpt \
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
  --output_path models/tedd_1104_XXL/XXL_eval_32.tsv \
  --experiment_name XXL_32 \
  --precision 32


python3 eval.py \
  --checkpoint_path models/tedd_1104_XXL/epoch=19-last.ckpt \
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
  --output_path models/tedd_1104_XXL/XXL_last_eval_32.tsv \
  --experiment_name XXL_last_32 \
  --precision 32


python3 eval.py \
  --checkpoint_path models/tedd_1104_M/epoch=3-step=138981.ckpt \
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
  --output_path models/tedd_1104_M/M_eval_32.tsv \
  --experiment_name M_32 \
  --precision 32

  python3 eval.py \
  --checkpoint_path models/tedd_1104_S/epoch=4-step=198544.ckpt \
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
  --output_path models/tedd_1104_S/S_eval_32.tsv \
  --experiment_name S_32 \
  --precision 32

########## PRECISION 16


python3 eval.py \
  --checkpoint_path models/tedd_1104_XXL/epoch=1-step=79417.ckpt \
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
  --output_path models/tedd_1104_XXL/XXL_eval.tsv \
  --experiment_name XXL


python3 eval.py \
  --checkpoint_path models/tedd_1104_XXL/epoch=19-last.ckpt \
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
  --output_path models/tedd_1104_XXL/XXL_last_eval.tsv \
  --experiment_name XXL_last


python3 eval.py \
  --checkpoint_path models/tedd_1104_M/epoch=3-step=138981.ckpt \
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
  --output_path models/tedd_1104_M/M_eval.tsv \
  --experiment_name M

  python3 eval.py \
  --checkpoint_path models/tedd_1104_S/epoch=4-step=198544.ckpt \
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
  --output_path models/tedd_1104_S/S_eval.tsv \
  --experiment_name S

