#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100,gpu-v100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=rl_hard8_drift_dice
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err

# ============================================================
# DICE-RL ("From Prior to Pro", arXiv 2603.10263) on the DRIFT policy.
# Residual RL: a_total = a_teacher(drift, K=1) + actor(state, noise).
# Implementation note: the drift state-dict loads cleanly into the FM
# architecture; setting num_inference_steps=1 collapses the K-step Euler
# sampler used by FMTeacher to drift's deployed 1-step rule
# (verified by scripts/smoke_dice_residual.py test [3]: exact bit-equality).
# So the algorithm is identical to fm_dice; only the teacher action changes.
# 8 hardest LIBERO-90 tasks: 8, 21, 32, 53, 65, 73, 75, 81
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

EXP_NAME_BC="drift_multitask_lib90"
RL_NAME="rl_hard8_drift_dice"
CKPT="${SCR_ROOT}/imitation/cold_start/libero/libero_90/${EXP_NAME_BC}/multitask_model_latest.pth"
LOG="${SCR_ROOT}/${RL_NAME}.log"
RES="${SCR_ROOT}/${RL_NAME}_results.txt"
mkdir -p "${SCR_ROOT}/imitation/logs"

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: drift multi-task BC ckpt not found at ${CKPT}" | tee -a "${RES}"
  exit 1
fi

N_ITERS=12
EVAL_EVERY=3
EPISODES_PER_ITER=8
WARMUP_EPISODES=24
GRAD_STEPS=200
EXPECTED=5

if [ -f "${LOG}" ]; then
  NDONE=$(grep -c "eval success rate" "${LOG}" || true)
  if [ "${NDONE}" -ge "${EXPECTED}" ]; then
    EVALS=$(grep "eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
    echo "[$(date)] CACHED ${RL_NAME} evals=[${EVALS}]" | tee -a "${RES}"
    exit 0
  fi
fi

echo "[$(date)] START ${RL_NAME} ckpt=${CKPT}" | tee -a "${RES}"

python dice_train.py \
  algo.num_inference_steps=1 \
  cold_start_checkpoint="${CKPT}" \
  'task_indices=[8,21,32,53,65,73,75,81]' \
  dice.n_iters=${N_ITERS} \
  dice.episodes_per_iter=${EPISODES_PER_ITER} \
  dice.warmup_episodes=${WARMUP_EPISODES} \
  dice.gradient_steps=${GRAD_STEPS} \
  dice.eval_interval=${EVAL_EVERY} \
  logging.mode=disabled \
  exp_name=${RL_NAME} \
  hydra.run.dir="${SCR_ROOT}/imitation/hydra/${RL_NAME}_${SLURM_JOB_ID}" \
  > "${LOG}" 2>&1 || true

EVALS=$(grep "eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
echo "[$(date)] DONE ${RL_NAME} evals=[${EVALS}]" | tee -a "${RES}"
LAST_PT=$(grep "per-task:" "${LOG}" | tail -1)
echo "final per-task: ${LAST_PT}" | tee -a "${RES}"

QFILT_N=$(grep -c "qfilt_keep" "${LOG}" || true)
echo "Q-filter activations logged: ${QFILT_N}" | tee -a "${RES}"

echo "Finished at $(date)"
