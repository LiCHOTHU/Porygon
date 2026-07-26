#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=resid_fm_hard8
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --array=0-7
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A_%a.out
#SBATCH --error=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A_%a.err

# ============================================================
# Residual RL (act_sim_ppo_v6_gtsim/rl_algorithm/residual) finetuning OUR frozen
# FLOW-MATCHING multitask BC base, LIBERO hard-8 tasks. K=10 (FM). Job array: one
# task per array element. Same fixed config as the t32 pilot.
#   sbatch scripts/residual_libero_fm_hard8.sbatch
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

TASKS=(8 21 32 53 65 73 75 81)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

CKPT="${SCR}/imitation/cold_start/libero/libero_90/cold_multitask_lib90/multitask_model_latest.pth"
OUT="${CEDAR}/imitation/experiments_residual/residual_hard8_fm_t${TASK}"
RES="${CEDAR}/residual_hard8_fm_t${TASK}_results.txt"
LOG="${CEDAR}/residual_hard8_fm_t${TASK}.log"
mkdir -p "${OUT}"

echo "[$(date)] START residual hard8 FM task=${TASK}  ckpt=${CKPT}" | tee -a "${RES}"

python act_sim_ppo_v6_gtsim/integrations/libero_imitation/train.py \
  --checkpoint "${CKPT}" \
  --task ${TASK} \
  --out_dir "${OUT}" \
  --n_iters 200 --episodes_per_iter 4 --grad_steps_per_iter 64 \
  --warmup_episodes 20 --eval_every 5 --eval_episodes 20 --save_every 10 \
  --bc_loss_weight 10.0 --residual_scale 0.1 --actor_update_freq 2 \
  2>&1 | tee -a "${LOG}"

echo "[$(date)] DONE residual hard8 FM task=${TASK}" | tee -a "${RES}"
grep -E "\[eval\]|\[iter|\[done\]" "${LOG}" | tail -50 | tee -a "${RES}"
