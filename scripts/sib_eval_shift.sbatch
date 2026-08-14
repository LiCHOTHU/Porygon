#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-v100,gpu-a100,gpu-rtx6000,gpu-l40s
#SBATCH --time=02:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=sib_shift
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --array=0-4
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%A_%a.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%A_%a.err

# Success rate under held-out camera shift -- the number the SIB claim rests on.
#
# One arm per array element, not one job over all five: the cluster enters a
# 5-day maintenance reservation at 06:00 and a serial sweep (~32 min per cell,
# 5 arms) would not finish inside the window.  In parallel each element is ~1h.
#
# The headroom this is measuring against, from the vanilla probe:
#     clean             0.940   20/20 solved
#     cam_shift         0.330   12/20
#     cam_shift_medium  0.200    7/20
# A 61-point drop, so there is real room for a method to recover -- the ceiling
# worry raised by the 0.940 clean score does not apply to the shifted scenes.
#
# Rollout eval is sim-bound rather than GPU-bound, hence the older/cheaper
# partitions; the L40S is worth reserving for training.

set -o pipefail
ARMS=(vanilla sib mixup sam dropout)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

cd /storage/home/hcoda1/8/lwang831/workspace/imitation || exit 1
source ~/.bashrc
conda activate porygon
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MUJOCO_GL=egl
export EGL_DEVICE_ID=0
export HF_HOME=/storage/project/r-agarg35-0/lwang831/tmp/hf
export TRANSFORMERS_CACHE=$HF_HOME
export MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/mpl
export TORCH_HOME=/storage/project/r-agarg35-0/lwang831/tmp/torch

echo "=== ${ARM} on $(hostname) at $(date -Is) ==="
# cam_shift first: if the window runs out, the number that matters already exists.
python scripts/sib_eval_shift.py \
    --arms "${ARM}" \
    --shifts cam_shift clean \
    --rollouts 5 \
    --out "/storage/scratch1/8/lwang831/imitation/sib_shift_${ARM}.jsonl"
echo "=== done $(date -Is) ==="
