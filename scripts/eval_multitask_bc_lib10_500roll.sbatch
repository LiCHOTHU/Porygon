#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100
#SBATCH --time=02:30:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=fm_eval_multitask_lib10_500
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err

# ============================================================
# High-precision multi-task deterministic eval of the FM lib10 BC ckpt.
# 50 rollouts/task across 10 tasks = 500 rolls per policy.
# Per-task SE ~7pp (50 rolls); overall-avg SE ~2.2pp.
# Storage: /storage/scratch1/8/lwang831 only.
# ============================================================
CONDA_ENV="porygon"
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/imitation"
SCR_ROOT="/storage/scratch1/8/lwang831"
cd "${WORKDIR}" || exit 1

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

EXP_NAME_BC="cold_multitask_lib10"
RUN_NAME="fm_eval_multitask_lib10_500roll"
CKPT="${SCR_ROOT}/imitation/cold_start/libero/libero_10/${EXP_NAME_BC}/multitask_model_latest.pth"
LOG="${SCR_ROOT}/${RUN_NAME}.log"
RES="${SCR_ROOT}/${RUN_NAME}_results.txt"
mkdir -p "${SCR_ROOT}/imitation/logs"

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: lib10 FM BC ckpt not found at ${CKPT}" | tee -a "${RES}"
  exit 1
fi

# Resume safety: if log already has an eval line, skip.
if [ -f "${LOG}" ] && grep -q "deterministic eval success rate" "${LOG}"; then
  EVAL=$(grep "deterministic eval success rate" "${LOG}" | tail -1)
  echo "[$(date)] CACHED ${RUN_NAME} ${EVAL}" | tee -a "${RES}"
  LAST_PT=$(grep "per-task:" "${LOG}" | tail -1)
  echo "per-task: ${LAST_PT}" | tee -a "${RES}"
  exit 0
fi

echo "[$(date)] START ${RUN_NAME} ckpt=${CKPT} (50 rolls/task)" | tee -a "${RES}"

python rl_train.py \
  cold_start_checkpoint="${CKPT}" \
  task.benchmark_name=libero_10 \
  'task_indices=[0,1,2,3,4,5,6,7,8,9]' \
  rl.n_iters=0 \
  rl.eval_rollouts_per_env=50 \
  logging.mode=disabled \
  exp_name=${RUN_NAME} \
  hydra.run.dir="${SCR_ROOT}/imitation/hydra/${RUN_NAME}_${SLURM_JOB_ID}" \
  >> "${LOG}" 2>&1 || true

EVAL=$(grep "deterministic eval success rate" "${LOG}" | tail -1)
echo "[$(date)] DONE ${RUN_NAME} ${EVAL}" | tee -a "${RES}"
LAST_PT=$(grep "per-task:" "${LOG}" | tail -1)
echo "per-task: ${LAST_PT}" | tee -a "${RES}"

echo "Finished at $(date)"
