#!/bin/bash
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --partition=h200-8-gm1128-c192-m2048
#SBATCH --time=24:00:00
#SBATCH --job-name=plot

# -----------------------------------------------------------------------------
# 1) ACTIVATE YOUR CONDA ENV
# -----------------------------------------------------------------------------
conda init bash >/dev/null 2>&1
source ~/.bashrc
cd PlaceFM/placefm
conda activate fm

python train.py --method placefm --dataset fsq --state GA --dt_model rf --eval --verbose --clustering_method kmeans --placefm_agg_gamma 0.0 --placefm_agg_alpha 1.0 --placefm_agg_beta 0.5