#!/bin/bash
#SBATCH --job-name=mistral-finetune
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="COMPUTE_CAPABILITY_7_5|COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|"
#SBATCH --partition=shared-gpu
#SBATCH --time=02:00:00
#SBATCH --mem=32G

module purge

mkdir -p logs outputs

echo "Starting training..."
make train CONFIG=configs/config.yaml

echo "Job finished."
