#!/bin/bash
#SBATCH --account=ixa
#SBATCH --partition=ixa
#SBATCH --job-name=video-classification-large
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=900G
#SBATCH --output=.slurm/video-classification-large.out.txt
#SBATCH --error=.slurm/video-classification-large.err.txt

module load Python
source /scratch/igarcia945/venvs/transformers/bin/activate


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
echo CUDA_VISIBLE_DEVICES "${CUDA_VISIBLE_DEVICES}"

CONFIGS_FOLDER="configs/video-classification"

torchrun --standalone --master_port 37223 --nproc_per_node=8 train_TEDD1104.py ${CONFIGS_FOLDER}/TEDD1104-large.yaml

