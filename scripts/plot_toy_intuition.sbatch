#!/usr/bin/env bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1
#SBATCH -p gpu-v100,gpu-a100 --time=00:30:00 --mem-per-gpu=16G
#SBATCH --job-name=toy_intuition --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%j.out
#SBATCH --error=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%j.err
source ~/.bashrc
conda activate porygon
cd /storage/home/hcoda1/8/lwang831/workspace/imitation
python scripts/plot_toy_intuition.py /storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/toy_intuition
