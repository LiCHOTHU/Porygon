#!/usr/bin/env bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1
#SBATCH -p gpu-v100,gpu-a100 --time=00:30:00 --mem-per-gpu=32G
#SBATCH --job-name=smoke_psq --qos=inferno --account=gts-agarg35
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%j.out
#SBATCH --error=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%j.err
source ~/.bashrc
conda activate porygon
cd /storage/home/hcoda1/8/lwang831/workspace/imitation
export MUJOCO_GL=egl EGL_DEVICE_ID=0
CE=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch
H8=$CE/imitation/experiments_dice/libero/libero_90/field_hard8_ff_B_grad_s10000/dice_latest.pth
echo "=== per_state_q_scale SMOKE (task 32, hard-8 ckpt) ==="
DIAG_CK=$H8 DIAG_TASK=32 DIAG_PSQ=1 python scripts/diagnose_drift_field.py
echo "=== control (global scale) ==="
DIAG_CK=$H8 DIAG_TASK=8 DIAG_PSQ=1 python scripts/diagnose_drift_field.py
