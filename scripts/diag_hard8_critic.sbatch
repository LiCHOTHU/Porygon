#!/usr/bin/env bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1
#SBATCH -p gpu-v100,gpu-a100 --time=01:00:00 --mem-per-gpu=32G
#SBATCH --job-name=diag_hard8 --qos=inferno --account=gts-agarg35
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%j.out
#SBATCH --error=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%j.err
source ~/.bashrc
conda activate porygon
cd /storage/home/hcoda1/8/lwang831/workspace/imitation
export MUJOCO_GL=egl EGL_DEVICE_ID=0
CE=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch
H8=$CE/imitation/experiments_dice/libero/libero_90/field_hard8_ff_B_grad_s10000/dice_latest.pth
ST=$CE/imitation/experiments_dice/libero/libero_90/field_drift_t32_C_distrib_zeroth/dice_latest.pth
for T in 32 8 73; do
  echo "############ HARD-8 MULTITASK CRITIC (iter-100 B ckpt), TASK $T ############"
  DIAG_CK=$H8 DIAG_TASK=$T python scripts/diagnose_drift_field.py
done
echo "############ CONTRAST: SINGLE-TASK task-32 CRITIC ############"
DIAG_CK=$ST DIAG_TASK=32 python scripts/diagnose_drift_field.py
