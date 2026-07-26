#!/usr/bin/env bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100,gpu-l40s
#SBATCH --time=00:20:00 --mem-per-gpu=16G
#SBATCH --job-name=toy_field_demo
#SBATCH --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.out
#SBATCH --error=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.err
source ~/.bashrc; conda activate porygon
OUT="${1:?usage: sbatch toy_field_vs_gradient.sbatch <outdir>}"
python -u /storage/home/hcoda1/8/lwang831/workspace/imitation/scripts/toy_field_vs_gradient.py "${OUT}"
