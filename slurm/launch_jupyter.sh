#!/bin/bash

#SBATCH --job-name=jupyter-lab
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

APPTAINER_IMAGE="$HOME/scratch/containers/llama.cpp.sif"
SCRATCH_DIR="$(realpath ~/scratch)"
HF_HOME="$SCRATCH_DIR/huggingface"
IP=localhost
PORT=8888

echo "Starting Jupyter Lab on $IP:$PORT on node $SLURMD_NODENAME ..."

srun apptainer exec \
  --nv \
  --env-file .env \
  --env HF_HOME=$HF_HOME \
  --mount type=bind,source=$SCRATCH_DIR,destination=$SCRATCH_DIR $APPTAINER_IMAGE \
    uv run jupyter lab --no-browser --ip=$IP --port=$PORT
