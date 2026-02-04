#!/bin/bash

#SBATCH --job-name=jupyter-notebook
#SBATCH --output=logs/jupyter-%j.out
#SBATCH --error=logs/jupyter-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=shared-gpu
#SBATCH --time=02:00:00
#SBATCH --mem=32G

module purge

mkdir -p logs outputs

IP=localhost
PORT=8888

srun jupyter notebook --no-browser --ip=$IP --port=$PORT
