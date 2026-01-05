#!/bin/bash
#SBATCH --job-name=dqi-eval
#SBATCH --output=logs/dqi-eval-%j.out
#SBATCH --error=logs/dqi-eval-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="COMPUTE_CAPABILITY_7_5|COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9"
#SBATCH --partition=shared-gpu
#SBATCH --array=1-8
#SBATCH --time=02:00:00
#SBATCH --mem=16G

module purge

mkdir -p logs outputs

CONFIG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" slurm/array_configs.txt)

echo "Starting evaluation..."
srun make evaluate CONFIG=$CONFIG

echo "Job finished."