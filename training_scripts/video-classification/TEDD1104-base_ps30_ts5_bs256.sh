#!/bin/bash
#SBATCH --job-name=video-classification-base_ps30_ts5_bs256
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --output=.slurm/video-classification-base_ps30_ts5_bs256.out.txt
#SBATCH --error=.slurm/video-classification-base_ps30_ts5_bs256.err.txt

source /ikerlariak/igarcia945/envs/pytorch2/bin/activate

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export WANDB_PROJECT=TEDD1104_vmae
export OMP_NUM_THREADS=16


torchrun --standalone --master_port 37227 --nproc_per_node=2 train_TEDD1104.py configs/video-classification/TEDD1104-base_ps30_ts5_bs256.yaml
torchrun --standalone --master_port 37227 --nproc_per_node=2 train_TEDD1104.py configs/video-classification/eval/TEDD1104-base_ps30_ts5_bs256.yaml
