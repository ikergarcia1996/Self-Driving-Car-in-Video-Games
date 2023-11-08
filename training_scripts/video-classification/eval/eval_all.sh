#!/bin/bash
#SBATCH --job-name=video-classification-base
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=300G
#SBATCH --output=.slurm/video-classification-base.out.txt
#SBATCH --error=.slurm/video-classification-base.err.txt

source /ikerlariak/igarcia945/envs/pytorch2/bin/activate

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export WANDB_PROJECT=TEDD1104_vmae
export OMP_NUM_THREADS=16

CONFIGS_FOLDER="configs/video-classification/eval"

torchrun --standalone --master_port 37227 --nproc_per_node=2 train_TEDD1104.py ${CONFIGS_FOLDER}/TEDD1104-base.yaml
torchrun --standalone --master_port 37227 --nproc_per_node=2 train_TEDD1104.py ${CONFIGS_FOLDER}/TEDD1104-base_ps30.yaml
torchrun --standalone --master_port 37227 --nproc_per_node=2 train_TEDD1104.py ${CONFIGS_FOLDER}/TEDD1104-base_ts5.yaml
