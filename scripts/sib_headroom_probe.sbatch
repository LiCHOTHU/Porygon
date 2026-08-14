#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-v100,gpu-a100,gpu-rtx6000
#SBATCH --time=02:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=sib_headroom
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err

# Does cam_shift create any headroom at all?
#
# vanilla scores 0.940 clean with 20/20 tasks solved -- essentially at ceiling.
# If the shifted scenes score about the same, then there is nothing for ANY
# regularizer to improve and the experiment cannot answer its question,
# regardless of what SIB does.  That would be a design failure, and with the
# cluster down for five days it is worth far more to learn it now than later.
# One arm, two conditions, on an existing checkpoint.

set -o pipefail
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

echo "=== headroom probe on $(hostname) at $(date -Is) ==="
python scripts/sib_eval_shift.py \
    --arms vanilla \
    --shifts clean cam_shift cam_shift_medium \
    --rollouts 5 \
    --out /storage/scratch1/8/lwang831/imitation/sib_headroom_probe.jsonl
echo "=== done $(date -Is) ==="
