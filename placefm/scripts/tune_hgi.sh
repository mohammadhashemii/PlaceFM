#!/bin/bash
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --partition=h200-8-gm1128-c192-m2048
#SBATCH --time=48:00:00
#SBATCH --job-name=tune_hgi

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
METHOD="hgi"                        # Model name
DATASET="fsq"                       # Dataset name
STATE=(
    "AL" "AZ" "AR" "CA" "CO" "CT" "DE" "FL" "GA"
    "ID" "IL" "IN" "IA" "KS" "KY" "LA" "ME" "MD"
    "MA" "MI" "MN" "MS" "MO" "MT" "NE" "NV" "NH" "NJ"
    "NM" "NY" "NC" "ND" "OH" "OK" "OR" "PA" "RI" "SC"
    "SD" "TN" "TX" "UT" "VT" "VA" "WA" "WV" "WI" "WY"
)

DT_MODEL=("rf")                     # Downstream task model          
LEARNING_RATES=(0.1 0.01 0.006)     # Learning rates
ALPHAS=(0.3 0.5 0.7)                # The hyperparameter to balance mutual information
EPOCHS=100                          # Number of epochs
SEED=1                              # Random seed

# -----------------------------------------------------------------------------
# 3) RUN TRAINING LOCALLY FOR EACH HYPERPARAMETER COMBINATION
# -----------------------------------------------------------------------------
for state in "${STATE[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        for alpha in "${ALPHAS[@]}"; do

            echo "==== Starting job with state=${state}, method=${METHOD} ===="
            echo "attention_head=${ah}, learning_rate=${lr}, alpha=${alpha}"

            python train.py \
                --dataset        "${DATASET}" \
                --method         "${METHOD}" \
                --state          "${state}" \
                --lr             "${lr}" \
                --hgi_alpha      "${alpha}" \
                --epochs         "${EPOCHS}" \
                --seed           "${SEED}" \
                --dt_model       "${DT_MODEL}" \
                --eval >> ../checkpoints/logs/hgi/hp_pd_${state}_lr${lr}_alpha${alpha}.log 2> ../checkpoints/logs/hgi/hp_pd_${state}_lr${lr}_alpha${alpha}.err

            echo "==== Finished job ===="
        done
    done
done

echo "Hyperparam tuning finished."
