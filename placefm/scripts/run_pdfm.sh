#!/bin/bash
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --partition=h200-8-gm1128-c192-m2048
#SBATCH --time=24:00:00
#SBATCH --job-name=pdfm

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
METHOD="pdfm"                       # Model name
DATASET="fsq"                       # Dataset name
DT_MODEL=("rf")                     # Downstream task model          
SEED=1                              # Random seed

STATE=(
    "AL" "AZ" "AR" "CA" "CO" "CT" "DE" "FL" "GA"
    "ID" "IL" "IN" "IA" "KS" "KY" "LA" "ME" "MD"
    "MA" "MI" "MN" "MS" "MO" "MT" "NE" "NV" "NH" "NJ"
    "NM" "NY" "NC" "ND" "OH" "OK" "OR" "PA" "RI" "SC"
    "SD" "TN" "TX" "UT" "VT" "VA" "WA" "WV" "WI" "WY"
)

# -----------------------------------------------------------------------------
# 3) RUN TRAINING LOCALLY FOR EACH STATE
# -----------------------------------------------------------------------------
for state in "${STATE[@]}"; do
        echo "==== Starting job with state=${state}, method=${METHOD} ===="

        python train.py \
            --dataset       "${DATASET}" \
            --method         "${METHOD}" \
            --state          "${state}" \
            --seed          "${SEED}" \
            --dt_model     "${DT_MODEL}" \
            --eval >> ../checkpoints/logs/pdfm/hp_pd_${state}.log 2> ../checkpoints/logs/pdfm/hp_pd_${state}.err

        echo "==== Finished job ===="
done

echo "training finished."
