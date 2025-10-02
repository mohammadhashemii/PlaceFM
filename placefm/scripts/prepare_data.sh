#!/bin/bash
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --partition=l40s-8-gm384-c192-m1536
#SBATCH --time=24:00:00
#SBATCH --job-name=prepare_data

# -----------------------------------------------------------------------------
# 1) ACTIVATE YOUR CONDA ENV
# -----------------------------------------------------------------------------
conda init bash >/dev/null 2>&1
source ~/.bashrc
cd PlaceFM/placefm
conda activate fm

# -----------------------------------------------------------------------------
# 2) DEFINE HYPERPARAMETER RANGES
# -----------------------------------------------------------------------------
DATASET="fsq"
METHOD="placefm"

# Manually store all US states in a Bash array
STATE=(
    "AR"
)

# -----------------------------------------------------------------------------
# 3) RUN CREATE DATASET
# -----------------------------------------------------------------------------
for state in "${STATE[@]}"; do
    echo "==== Creating POI graph dataset for state=${state} ===="

    python prepare_dataset.py \
        --dataset "${DATASET}" \
        --method  "${METHOD}" \
        --state   "${state}" >> ../checkpoints/logs/data/${state}.log 2> ../checkpoints/logs/data/${state}.err

    echo "==== Finished job for ${state} ===="
done

echo "Dataset creation finished."
