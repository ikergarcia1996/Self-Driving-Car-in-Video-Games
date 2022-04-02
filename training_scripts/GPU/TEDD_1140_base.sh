python3 train.py --train_new \
  --train_dir ../gtaai_datasets/train \
  --val_dir  ../gtaai_datasets/dev \
  --output_dir models/tedd_1104_base \
  --encoder_type transformer \
  --batch_size 16 \
  --accumulation_steps 4 \
  --max_epochs 40 \
  --cnn_model_name convnext_base \
  --num_layers_encoder 4 \
  --embedded_size 512 \
  --learning_rate 5e-5 \
  --mask_prob 0.2 \
  --dropout_cnn_out 0.3 \
  --dropout_encoder 0.1 \
  --dropout_encoder_features 0.3 \
  --control_mode keyboard \
  --val_check_interval 0.5 \
  --precision "bf16" \
  --devices 1


