#!/bin/bash
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --partition=a100-8-gm320-c96-m1152
#SBATCH --time=24:00:00
#SBATCH --job-name=tune_placefm

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
METHOD="placefm"                        # Model name
DATASET="fsq"                       # Dataset name
STATE=(
    "AZ" "AR" "CA" "CO" "CT" "DE" "FL" "GA"
    "ID" "IL" "IN" "IA" "KS" "KY" "LA" "ME" "MD"
    "MA" "MI" "MN" "MS" "MO" "MT" "NE" "NV" "NH" "NJ"
    "NM" "NY" "NC" "ND" "OH" "OK" "OR" "PA" "RI" "SC"
    "SD" "TN" "TX" "UT" "VT" "VA" "WA" "WV" "WI" "WY"
)

GAMMAS=(0.0 0.5 1.0)                # The hyperparameter to balance spatial and semantic importance
ALPHAS=(0.0 0.5 1.0)                # The hyperparameter to balance spatial and semantic importance
BETAS=(0.0 0.5 1.0)                 # The hyperparameter to balance spatial and semantic importance
reduction_ratios=(0.1 0.05)                   # kmeans reduction ratio for calculating number of clusters
CLUSTERING_METHOD=("kmeans")

SEED=1                                        # Random seed
DT_MODEL=("rf")                               # Downstream task model  

# -----------------------------------------------------------------------------
# 3) RUN TRAINING LOCALLY FOR EACH HYPERPARAMETER COMBINATION
# -----------------------------------------------------------------------------
for state in "${STATE[@]}"; do
    mkdir -p ../checkpoints/logs/placefm/${state}
    for cm in "${CLUSTERING_METHOD[@]}"; do
        for gamma in "${GAMMAS[@]}"; do
            for alpha in "${ALPHAS[@]}"; do
                for beta in "${BETAS[@]}"; do
                    for rr in "${reduction_ratios[@]}"; do

                        echo "==== Starting job with state=${state}, method=${METHOD} ===="
                        echo "gamma=${gamma}, alpha=${alpha}, beta=${beta}, reduction_ratio=${rr}"

                        python train.py \
                            --dataset                           "${DATASET}" \
                            --method                            "${METHOD}" \
                            --clustering_method                 "${cm}" \
                            --state                             "${state}" \
                            --placefm_agg_gamma                 "${gamma}" \
                            --placefm_agg_alpha                 "${alpha}" \
                            --placefm_agg_beta                  "${beta}" \
                            --placefm_kmeans_reduction_ratio    "${rr}" \
                            --seed                              "${SEED}" \
                            --dt_model                          "${DT_MODEL}" \
                            --verbose \
                            --eval >> ../checkpoints/logs/placefm/${state}/hp_pd_${cm}_g${gamma}_a${alpha}_b${beta}_rr${rr}.log 2> ../checkpoints/logs/placefm/${state}/hp_pd_${cm}_g${gamma}_a${alpha}_b${beta}_rr${rr}.err

                        echo "==== Finished job ===="
                    done
                done
            done
        done
    done
done
done

echo "Hyperparam tuning finished."
