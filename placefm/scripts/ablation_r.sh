#!/bin/bash
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --partition=h200-8-gm1128-c192-m2048
#SBATCH --time=24:00:00
#SBATCH --job-name=ablation_r

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
    "CA"
)
alpha=0.5
gamma=0.0
beta=1.0
cm="kmeans"                                  # Clustering method

reduction_ratios=(0.01 0.05 0.1 0.2 0.5)                   # kmeans reduction ratio for calculating number of clusters

SEED=1                                        # Random seed
DT_MODEL=("rf")                               # Downstream task model  

# -----------------------------------------------------------------------------
# 3) RUN TRAINING LOCALLY FOR EACH HYPERPARAMETER COMBINATION
# -----------------------------------------------------------------------------
for state in "${STATE[@]}"; do
    mkdir -p ../checkpoints/logs/placefm/${state}
    for rr in "${reduction_ratios[@]}"; do

        echo "==== Starting job with state=${state}, method=${METHOD} ===="
        echo "gamma=${gamma}, alpha=${alpha}, beta=${beta}, reduction_ratio=${rr}"

        python train.py \
            --dataset                           "${DATASET}" \
            --method                            "${METHOD}" \
            --clustering_method                 kmeans \
            --state                             "${state}" \
            --placefm_agg_gamma                 "${gamma}" \
            --placefm_agg_alpha                 "${alpha}" \
            --placefm_agg_beta                  "${beta}" \
            --placefm_kmeans_reduction_ratio    "${rr}" \
            --seed                              "${SEED}" \
            --dt_model                          "${DT_MODEL}" \
            --verbose \
            --eval >> ../checkpoints/logs/placefm/${state}/ablation/hp_pd_${cm}_g${gamma}_a${alpha}_b${beta}_rr${rr}.log 2> ../checkpoints/logs/placefm/${state}/ablation/hp_pd_${cm}_g${gamma}_a${alpha}_b${beta}_rr${rr}.err

        echo "==== Finished job ===="
    done
done

echo "Ablation study finished."
