#!/bin/bash
#SBATCH --job-name=data-prep
#SBATCH --output=logs/data-prep-%j.out
#SBATCH --error=logs/data-prep-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=shared-cpu
#SBATCH --time=00:30:00
#SBATCH --mem=32G

module purge

mkdir -p logs outputs

if [ $# -gt 0 ]; then
  CONFIG=$1
else
  CONFIG="configs/config.yaml"
fi

echo "Starting data preparation..."
make data CONFIG=$CONFIG

echo "Job finished."
