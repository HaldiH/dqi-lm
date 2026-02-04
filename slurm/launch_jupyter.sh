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

APPTAINER_IMAGE="~/scratch/containers/unsloth.sif"
SCRATCH_DIR=$(realpath ~/scratch)
IP=localhost
PORT=8888


srun apptainer exec \
  --nv \
  --env-file .env \
  --mount type=bind,source=$SCRATCH_DIR,destination=$SCRATCH_DIR $APPTAINER_IMAGE \
    jupyter notebook --no-browser --ip=$IP --port=$PORT
