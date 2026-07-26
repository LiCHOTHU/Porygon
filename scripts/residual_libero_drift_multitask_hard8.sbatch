#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=resid_drift_mt8
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%j.out
#SBATCH --error=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%j.err

# ============================================================
# MULTITASK residual RL (act_sim_ppo_v6_gtsim/rl_algorithm/residual): ONE shared
# residual finetuned jointly over the LIBERO hard-8 tasks, on the frozen
# DRIFTING multitask BC base. K=1 (drift). Single job.
#   sbatch scripts/residual_libero_drift_multitask_hard8.sbatch
# ============================================================
CONDA_ENV="porygon"
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/imitation"
SCR="/storage/scratch1/8/lwang831"
CEDAR="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch"
cd "${WORKDIR}" || exit 1
source ~/.bashrc
conda activate "${CONDA_ENV}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MUJOCO_GL=egl EGL_DEVICE_ID=0
nvidia-smi || true

TASKS="8,21,32,53,65,73,75,81"

CKPT="${SCR}/imitation/cold_start/libero/libero_90/drift_multitask_lib90/multitask_model_latest.pth"
OUT="${CEDAR}/imitation/experiments_residual/residual_mt8_drift"
RES="${CEDAR}/residual_mt8_drift_results.txt"
LOG="${CEDAR}/residual_mt8_drift.log"
mkdir -p "${OUT}"

echo "[$(date)] START residual MULTITASK DRIFT tasks=${TASKS}  ckpt=${CKPT}" | tee -a "${RES}"

python act_sim_ppo_v6_gtsim/integrations/libero_imitation/train.py \
  --checkpoint "${CKPT}" \
  --tasks "${TASKS}" \
  --out_dir "${OUT}" \
  --n_iters 200 --episodes_per_iter 8 --grad_steps_per_iter 64 \
  --warmup_episodes 8 --eval_every 10 --eval_episodes 10 --save_every 10 \
  --bc_loss_weight 10.0 --residual_scale 0.1 --actor_update_freq 2 \
  2>&1 | tee -a "${LOG}"

echo "[$(date)] DONE residual MULTITASK DRIFT" | tee -a "${RES}"
grep -E "\[eval\]|\[iter|\[done\]" "${LOG}" | tail -60 | tee -a "${RES}"
