#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=fmrl_demos_sweep_t16
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err

# ============================================================
# FM-RL (GRPO + per-step PPO clip) demos sweep on LIBERO task 16.
# For each cold-start checkpoint trained at d=5/10/20/40 demos:
#   fine-tune for N_ITERS RL iterations and record deterministic eval
#   success rates. Goal: fill the BC -> BC+RL gain table.
# Resumable: if an RL exp dir already contains a finished log, the cell is
#   skipped (re-submit with --requeue safe).
# Storage: everything under /storage/scratch1/8/lwang831 (never $HOME).
# ============================================================
CONDA_ENV="porygon"
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/imitation"
# NOTE: do NOT use $SCRATCH — PACE's ~/.bashrc exports it to /storage/project/...
# and would silently override us. Use SCR_ROOT.
SCR_ROOT="/storage/scratch1/8/lwang831"
cd "${WORKDIR}" || exit 1

echo "================ Job Info ================"
echo "JobID:  ${SLURM_JOB_ID}"
echo "Node:   ${SLURM_NODELIST}"
echo "CWD:    $(pwd)"
echo "Start:  $(date)"
echo "========================================="

# ------------------------------------------------------------
# Environment (porygon's activate.d remaps cluster's RO cache vars)
# ------------------------------------------------------------
source ~/.bashrc
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Headless rendering for robosuite/mujoco offscreen depth+rgb
export MUJOCO_GL=egl
export EGL_DEVICE_ID=0

nvidia-smi || true
python -V
which python

# ------------------------------------------------------------
# RL sweep config (v2-tuned stable settings)
# ------------------------------------------------------------
TASK=16
DEMOS_LIST=(5 10 20 40)
N_ITERS=15
GROUP_SIZE=8
INITS_PER_ITER=6
PPO_EPOCHS=4
LR=1.0e-6
TARGET_KL=0.05
EVAL_EVERY=5
EVAL_ROLLOUTS=20

RES="${SCR_ROOT}/rl_demos_sweep_t16_results.txt"
mkdir -p "${SCR_ROOT}/imitation/logs"
echo "[$(date)] sweep job=${SLURM_JOB_ID} N_ITERS=${N_ITERS} groups=${INITS_PER_ITER} G=${GROUP_SIZE}" | tee -a "${RES}"

run_one() {
  local D=$1
  local COLD_NAME=cold_t${TASK}_d${D}
  local RL_NAME=fm_rl_t${TASK}_d${D}
  local CKPT="${SCR_ROOT}/imitation/cold_start/libero/libero_90/${COLD_NAME}/multitask_model_latest.pth"
  local EXP_DIR="${SCR_ROOT}/imitation/experiments_rl/libero/libero_90/${RL_NAME}"
  local LOG="${SCR_ROOT}/${RL_NAME}.log"

  if [ ! -f "${CKPT}" ]; then
    echo "[$(date)] SKIP d=${D}: no cold-start checkpoint at ${CKPT}" | tee -a "${RES}"
    return
  fi

  # Resume safety: skip if a prior run already finished N_ITERS deterministic evals
  if [ -f "${LOG}" ]; then
    local NDONE=$(grep -c "deterministic eval success rate" "${LOG}" || true)
    local EXPECTED=$(( N_ITERS / EVAL_EVERY + 1 ))
    if [ "${NDONE}" -ge "${EXPECTED}" ]; then
      local EVALS=$(grep "deterministic eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
      echo "[$(date)] CACHED d=${D} RL evals=[${EVALS}]" | tee -a "${RES}"
      return
    fi
  fi

  echo "[$(date)] START RL d=${D} ckpt=${CKPT}" | tee -a "${RES}"

  python rl_train.py \
    cold_start_checkpoint="${CKPT}" task_index=${TASK} \
    rl.n_iters=${N_ITERS} \
    rl.group_size=${GROUP_SIZE} \
    rl.inits_per_iter=${INITS_PER_ITER} \
    rl.ppo_epochs=${PPO_EPOCHS} \
    rl.lr=${LR} \
    rl.target_kl=${TARGET_KL} \
    rl.eval_interval=${EVAL_EVERY} \
    rl.eval_rollouts_per_env=${EVAL_ROLLOUTS} \
    logging.mode=disabled \
    exp_name=${RL_NAME} \
    hydra.run.dir="${SCR_ROOT}/imitation/hydra/${RL_NAME}_${SLURM_JOB_ID}" \
    > "${LOG}" 2>&1 || true

  local EVALS=$(grep "deterministic eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
  echo "[$(date)] DONE d=${D} RL evals=[${EVALS}]" | tee -a "${RES}"
}

for D in "${DEMOS_LIST[@]}"; do
  run_one "${D}"
done

echo "[$(date)] RL SWEEP DONE job=${SLURM_JOB_ID}" | tee -a "${RES}"
echo "Finished at $(date)"
