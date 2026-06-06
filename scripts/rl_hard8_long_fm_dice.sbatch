#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=rl_hard8_long_fm_dice
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err

# ============================================================
# LONG-RUN DICE-RL ("From Prior to Pro") on the FM policy, hard-8 tasks.
# Scaled up from the 2026-06-04 short run (n_iters=12) to approach the
# DICE paper's regime: more warmup, more outer iters, more total env
# steps so the ensemble critic can actually converge before we cut.
# Auto-resumes from dice_latest.pth on requeue (preemption-safe).
# Hard-task indices: 8, 21, 32, 53, 65, 73, 75, 81
# Storage:  /storage/scratch1/8/lwang831 only
# ============================================================
CONDA_ENV="porygon"
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/imitation"
SCR_ROOT="/storage/scratch1/8/lwang831"
cd "${WORKDIR}" || exit 1

echo "================ Job Info ================"
echo "JobID:  ${SLURM_JOB_ID}"
echo "Node:   ${SLURM_NODELIST}"
echo "Start:  $(date)"
echo "========================================="

source ~/.bashrc
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MUJOCO_GL=egl
export EGL_DEVICE_ID=0

nvidia-smi || true
python -V

EXP_NAME_BC="cold_multitask_lib90"
RL_NAME="rl_hard8_long_fm_dice"
CKPT="${SCR_ROOT}/imitation/cold_start/libero/libero_90/${EXP_NAME_BC}/multitask_model_latest.pth"
LOG="${SCR_ROOT}/${RL_NAME}.log"
RES="${SCR_ROOT}/${RL_NAME}_results.txt"
RESUME_CKPT="${SCR_ROOT}/imitation/experiments_dice/libero/libero_90/${RL_NAME}/dice_latest.pth"
mkdir -p "${SCR_ROOT}/imitation/logs"

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: FM multi-task BC ckpt not found at ${CKPT}" | tee -a "${RES}"
  exit 1
fi

N_ITERS=40           # was 12 -> ~3.3x more iters (paper does 100-500)
EVAL_EVERY=5         # was 3 -> less eval overhead
EPISODES_PER_ITER=8
WARMUP_EPISODES=64   # was 24 -> seed replay better for ensemble critic
GRAD_STEPS=200
SAVE_EVERY=5         # checkpoint every 5 iters for resume safety
EXPECTED=9           # 1 baseline + 8 evals (iter 5,10,...,40)

if [ -f "${LOG}" ]; then
  NDONE=$(grep -c "eval success rate" "${LOG}" || true)
  if [ "${NDONE}" -ge "${EXPECTED}" ]; then
    EVALS=$(grep "eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
    echo "[$(date)] CACHED ${RL_NAME} evals=[${EVALS}]" | tee -a "${RES}"
    exit 0
  fi
fi

RESUME_FLAG=""
if [ -f "${RESUME_CKPT}" ]; then
  RESUME_FLAG="+dice_resume_checkpoint=${RESUME_CKPT}"
  echo "[$(date)] RESUME from ${RESUME_CKPT}" | tee -a "${RES}"
fi

echo "[$(date)] START ${RL_NAME} ckpt=${CKPT} n_iters=${N_ITERS}" | tee -a "${RES}"

python dice_train.py \
  cold_start_checkpoint="${CKPT}" \
  ${RESUME_FLAG} \
  'task_indices=[8,21,32,53,65,73,75,81]' \
  dice.n_iters=${N_ITERS} \
  dice.episodes_per_iter=${EPISODES_PER_ITER} \
  dice.warmup_episodes=${WARMUP_EPISODES} \
  dice.gradient_steps=${GRAD_STEPS} \
  dice.eval_interval=${EVAL_EVERY} \
  dice.save_interval=${SAVE_EVERY} \
  logging.mode=disabled \
  exp_name=${RL_NAME} \
  hydra.run.dir="${SCR_ROOT}/imitation/hydra/${RL_NAME}_${SLURM_JOB_ID}" \
  >> "${LOG}" 2>&1 || true

EVALS=$(grep "eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
echo "[$(date)] DONE ${RL_NAME} evals=[${EVALS}]" | tee -a "${RES}"
LAST_PT=$(grep "per-task:" "${LOG}" | tail -1)
echo "final per-task: ${LAST_PT}" | tee -a "${RES}"

echo "Finished at $(date)"
