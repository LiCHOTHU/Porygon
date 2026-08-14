#!/usr/bin/env bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1
#SBATCH -p gpu-v100,gpu-a100 --time=00:40:00 --mem-per-gpu=32G
#SBATCH --job-name=probe_st --qos=inferno --account=gts-agarg35
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%j.out
#SBATCH --error=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%j.err
source ~/.bashrc; conda activate porygon
cd /storage/home/hcoda1/8/lwang831/workspace/imitation
export MUJOCO_GL=egl EGL_DEVICE_ID=0
CE=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch
for T in 73 75 81 65; do
  D=$CE/imitation/experiments_dice/libero/libero_90/field_st_B_t${T}_inf_s10000
  CK=$D/dice_latest.pth; [ -f "$CK" ] || CK=$D/10000/dice_latest.pth
  echo "######## SINGLE-TASK CRITIC PROBE task $T ########"
  DIAG_CK=$CK DIAG_TASK=$T python scripts/diagnose_drift_field.py 2>/dev/null | grep -E "^\[[12]\]|grad_a|std\(Q\)|-> "
done
