#!/bin/bash
#SBATCH --job-name=pretrain+video-classification-large
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:2
#SBATCH --mem=480G
#SBATCH --output=.slurm/pretrain+video-classification-large.out.txt
#SBATCH --error=.slurm/pretrain+video-classification-large.err.txt

source /ikerlariak/igarcia945/envs/pytorch2/bin/activate

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export WANDB_PROJECT=TEDD1104_vmae
export OMP_NUM_THREADS=16

CONFIGS_FOLDER="configs/pretrain+video-classification"

torchrun --standalone --master_port 37223 --nproc_per_node=2 train_TEDD1104.py ${CONFIGS_FOLDER}/TEDD1104-large.yaml
