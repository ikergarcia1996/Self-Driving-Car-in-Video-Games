### LAST TRY LARGE

python3 train.py --train_new \
  --train_dir ../gtaai_datasets/train \
  --val_dir  ../gtaai_datasets/dev \
  --output_dir models/tedd_1104_slarge \
  --encoder_type transformer \
  --dataloader_num_workers 20 \
  --batch_size 8 \
  --accumulation_steps 2 \
  --max_epochs 20 \
  --cnn_model_name efficientnet_v2_l \
  --num_layers_encoder 4 \
  --embedded_size 896 \
  --learning_rate 1e-3 \
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
  --devices 4 \
  --strategy "ddp_find_unused_parameters_false"


###########################################
python3 train.py --train_new \
  --train_dir ../gtaai_datasets/train \
  --val_dir  ../gtaai_datasets/dev \
  --output_dir models/tedd_1104_large_optim \
  --encoder_type transformer \
  --dataloader_num_workers 32 \
  --batch_size 8 \
  --accumulation_steps 1 \
  --max_epochs 20 \
  --cnn_model_name efficientnet_v2_l \
  --num_layers_encoder 6 \
  --embedded_size 512 \
  --learning_rate 3e-5 \
  --mask_prob 0.2 \
  --hide_map_prob 0.0 \
  --dropout_cnn_out 0.4 \
  --dropout_encoder 0.2 \
  --dropout_encoder_features 0.5 \
  --label_smoothing 0.1 \
  --control_mode keyboard \
  --val_check_interval 0.25 \
  --precision "16"  \
  --devices 4 \
  --strategy "ddp_find_unused_parameters_false"

######### OPTIM 2
python3 train.py --train_new \
  --train_dir ../gtaai_datasets/train \
  --val_dir  ../gtaai_datasets/dev \
  --output_dir models/tedd_1104_large_optim_mask \
  --encoder_type transformer \
  --dataloader_num_workers 32 \
  --batch_size 8 \
  --accumulation_steps 1 \
  --max_epochs 20 \
  --cnn_model_name efficientnet_v2_l \
  --num_layers_encoder 6 \
  --embedded_size 512 \
  --learning_rate 3e-5 \
  --mask_prob 0.2 \
  --hide_map_prob 0.0 \
  --dropout_cnn_out 0.3 \
  --dropout_encoder 0.1 \
  --dropout_encoder_features 0.5 \
  --label_smoothing 0.1 \
  --control_mode keyboard \
  --val_check_interval 0.25 \
  --precision "16"  \
  --devices 4 \
  --strategy "ddp_find_unused_parameters_false"

####### OPTIM 2 NO MASK

python3 train.py --train_new \
  --train_dir ../gtaai_datasets/train \
  --val_dir  ../gtaai_datasets/dev \
  --output_dir models/tedd_1104_large_optim2_nomask \
  --encoder_type transformer \
  --dataloader_num_workers 32 \
  --batch_size 8 \
  --accumulation_steps 2 \
  --max_epochs 20 \
  --cnn_model_name efficientnet_v2_l \
  --num_layers_encoder 4 \
  --embedded_size 384 \
  --learning_rate 3e-5 \
  --mask_prob 0.0 \
  --hide_map_prob 0.0 \
  --dropout_cnn_out 0.3 \
  --dropout_encoder 0.1 \
  --dropout_encoder_features 0.4 \
  --label_smoothing 0.1 \
  --control_mode keyboard \
  --val_check_interval 0.25 \
  --precision "16"  \
  --devices 2 \
  --strategy "ddp_find_unused_parameters_false"


########### BIG TRANSFORMER
python3 train.py --train_new \
  --train_dir ../gtaai_datasets/train \
  --val_dir  ../gtaai_datasets/dev \
  --output_dir models/tedd_1104_large_bigTransformer \
  --encoder_type transformer \
  --dataloader_num_workers 32 \
  --batch_size 8 \
  --accumulation_steps 2 \
  --max_epochs 20 \
  --cnn_model_name efficientnet_v2_l \
  --num_layers_encoder 4 \
  --embedded_size 896 \
  --learning_rate 3e-5 \
  --mask_prob 0.2 \
  --hide_map_prob 0.0 \
  --dropout_cnn_out 0.3 \
  --dropout_encoder 0.1 \
  --dropout_encoder_features 0.4 \
  --label_smoothing 0.1 \
  --control_mode keyboard \
  --val_check_interval 0.25 \
  --precision "16"  \
  --devices 2 \
  --strategy "ddp_find_unused_parameters_false"

###############


python3 eval.py \
  --checkpoint_path models/tedd_1104_large_optim/epoch=1-step=79417.ckpt \
  --batch_size 16 \
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
  --output_path models/tedd_1104_large_optim/results_best.tsv \
  --experiment_name large_optim_best \


python3 eval.py \
  --checkpoint_path models/tedd_1104_large_optim/epoch=7-last.ckpt \
  --batch_size 16 \
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
  --output_path models/tedd_1104_large_optim/results_last.tsv \
  --experiment_name large_optim_last