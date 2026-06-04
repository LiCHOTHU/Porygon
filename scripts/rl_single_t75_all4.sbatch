#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100,gpu-v100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=rl_single_t75_all4
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err

# ============================================================
# Single-task RL finetuning + eval on LIBERO-90 task 75
# (STUDY_SCENE1 pick up the book and place it in the right
# compartment of the caddy). Cold-start success: FM~0.50-0.73,
# drift~0.67. Runs all 4 algos sequentially in one job:
#   1) fm_dice          (DICE-RL on FM, K=10)
#   2) drift_dice       (DICE-RL on drift, K=1)
#   3) fm_simplevla     (SimpleVLA-RL on FM)
#   4) drift_simplevla  (SimpleVLA-RL on drift)
# Each algo gets ~1.5-2h. Resume-safe (skips if log already has
# enough eval lines from a previous run).
# Storage: /storage/scratch1/8/lwang831 only.
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

FM_CKPT="${SCR_ROOT}/imitation/cold_start/libero/libero_90/cold_multitask_lib90/multitask_model_latest.pth"
DRIFT_CKPT="${SCR_ROOT}/imitation/cold_start/libero/libero_90/drift_multitask_lib90/multitask_model_latest.pth"
RES="${SCR_ROOT}/rl_single_t75_all4_results.txt"
mkdir -p "${SCR_ROOT}/imitation/logs"

if [ ! -f "${FM_CKPT}" ] || [ ! -f "${DRIFT_CKPT}" ]; then
  echo "ERROR: cold-start ckpts missing" | tee -a "${RES}"
  exit 1
fi

echo "[$(date)] ===== START single-task t75 (all 4 algos) =====" | tee -a "${RES}"

run_dice () {
  local RL_NAME=$1
  local CKPT=$2
  local EXTRA=$3
  local LOG="${SCR_ROOT}/${RL_NAME}.log"
  local EXPECTED=7   # 1 baseline + 6 evals (every 2 iters of 12)

  if [ -f "${LOG}" ]; then
    local NDONE=$(grep -c "eval success rate" "${LOG}" || true)
    if [ "${NDONE}" -ge "${EXPECTED}" ]; then
      local EVALS=$(grep "eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
      echo "[$(date)] CACHED ${RL_NAME} evals=[${EVALS}]" | tee -a "${RES}"
      return 0
    fi
  fi

  echo "[$(date)] >>> START ${RL_NAME}" | tee -a "${RES}"
  python dice_train.py ${EXTRA} \
    cold_start_checkpoint="${CKPT}" \
    'task_indices=[75]' \
    dice.n_iters=12 \
    dice.episodes_per_iter=8 \
    dice.warmup_episodes=16 \
    dice.gradient_steps=200 \
    dice.eval_interval=2 \
    dice.save_interval=4 \
    logging.mode=disabled \
    exp_name=${RL_NAME} \
    hydra.run.dir="${SCR_ROOT}/imitation/hydra/${RL_NAME}_${SLURM_JOB_ID}" \
    > "${LOG}" 2>&1 || true

  local EVALS=$(grep "eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
  echo "[$(date)] <<< DONE ${RL_NAME} evals=[${EVALS}]" | tee -a "${RES}"
  local LAST_PT=$(grep "per-task:" "${LOG}" | tail -1)
  echo "  final per-task: ${LAST_PT}" | tee -a "${RES}"
}

run_simplevla () {
  local RL_NAME=$1
  local CKPT=$2
  local EXTRA=$3
  local LOG="${SCR_ROOT}/${RL_NAME}.log"
  local EXPECTED=5   # 1 baseline (iter 0) + 4 evals (every 2 iters of 8)

  if [ -f "${LOG}" ]; then
    local NDONE=$(grep -c "deterministic eval success rate" "${LOG}" || true)
    if [ "${NDONE}" -ge "${EXPECTED}" ]; then
      local EVALS=$(grep "deterministic eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
      echo "[$(date)] CACHED ${RL_NAME} evals=[${EVALS}]" | tee -a "${RES}"
      return 0
    fi
  fi

  echo "[$(date)] >>> START ${RL_NAME}" | tee -a "${RES}"
  python rl_train.py ${EXTRA} \
    cold_start_checkpoint="${CKPT}" \
    'task_indices=[75]' \
    rl.n_iters=8 \
    rl.group_size=16 \
    rl.inits_per_iter=4 \
    rl.ppo_epochs=2 \
    rl.lr=1.0e-6 \
    rl.target_kl=0.05 \
    rl.filter_accuracy=true \
    rl.filter_low=0.1 \
    rl.filter_high=0.9 \
    rl.eval_interval=2 \
    rl.eval_rollouts_per_env=15 \
    logging.mode=disabled \
    exp_name=${RL_NAME} \
    hydra.run.dir="${SCR_ROOT}/imitation/hydra/${RL_NAME}_${SLURM_JOB_ID}" \
    > "${LOG}" 2>&1 || true

  local EVALS=$(grep "deterministic eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
  echo "[$(date)] <<< DONE ${RL_NAME} evals=[${EVALS}]" | tee -a "${RES}"
  local LAST_PT=$(grep "per-task:" "${LOG}" | tail -1)
  echo "  final per-task: ${LAST_PT}" | tee -a "${RES}"
  local FILTER_N=$(grep -c "] filter \[" "${LOG}" || true)
  echo "  filter activations: ${FILTER_N}" | tee -a "${RES}"
}

# 1) DICE-RL on FM
run_dice "rl_single_t75_fm_dice" "${FM_CKPT}" ""

# 2) DICE-RL on drift (K=1)
run_dice "rl_single_t75_drift_dice" "${DRIFT_CKPT}" "algo.num_inference_steps=1"

# 3) SimpleVLA-RL on FM
run_simplevla "rl_single_t75_fm_simplevla" "${FM_CKPT}" ""

# 4) SimpleVLA-RL on drift (K=1)
run_simplevla "rl_single_t75_drift_simplevla" "${DRIFT_CKPT}" "algo.num_inference_steps=1"

echo "[$(date)] ===== ALL 4 DONE =====" | tee -a "${RES}"
