#!/bin/bash
#SBATCH --job-name=video-classification-large_ps30_ts5
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --output=.slurm/video-classification-large_ps30_ts5.out.txt
#SBATCH --error=.slurm/video-classification-large_ps30_ts5.err.txt

source /ikerlariak/igarcia945/envs/pytorch2/bin/activate

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export WANDB_PROJECT=TEDD1104_vmae
export WANDB_ENTITY=igarciaf
export WANDB_PROJECT=TEDD1104_vmae2
export OMP_NUM_THREADS=16
export WANDB__SERVICE_WAIT=300

CONFIGS_FOLDER="configs/video-classification"

torchrun --standalone --master_port 37223 --nproc_per_node=4 train_TEDD1104.py ${CONFIGS_FOLDER}/TEDD1104-large_ps30_ts5.yaml
torchrun --standalone --master_port 37223 --nproc_per_node=4 train_TEDD1104.py ${CONFIGS_FOLDER}/eval/TEDD1104-large_ps30_ts5.yaml
