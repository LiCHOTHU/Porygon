#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100
#SBATCH --time=03:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=fm_eval_multitask_lib90
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err

# ============================================================
# Full LIBERO-90 deterministic eval of the multi-task BC ckpt
# (cold_multitask_lib90/multitask_model_latest.pth). Iterates
# task_index 0..89, 5 deterministic rollouts each (initial+final
# at n_iters=0 -> 10 rolls/task = 900 rolls total). Writes one
# results file with per-task success rates for later analysis.
# Skips a task if its log already contains the eval line.
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

EXP_NAME_BC="cold_multitask_lib90"
CKPT="${SCR_ROOT}/imitation/cold_start/libero/libero_90/${EXP_NAME_BC}/multitask_model_latest.pth"
RES="${SCR_ROOT}/eval_multitask_lib90_per_task.txt"
LOGDIR="${SCR_ROOT}/imitation/eval_logs/${EXP_NAME_BC}"
mkdir -p "${LOGDIR}" "${SCR_ROOT}/imitation/logs"

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: cold-start ckpt not found at ${CKPT}" | tee -a "${RES}"
  exit 1
fi

echo "[$(date)] eval start ckpt=${CKPT} job=${SLURM_JOB_ID}" | tee -a "${RES}"

for T in $(seq 0 89); do
  LOG="${LOGDIR}/task_${T}.log"
  # Resume safety: skip if eval line already present
  if [ -f "${LOG}" ] && grep -q "deterministic eval success rate" "${LOG}"; then
    EVALS=$(grep "deterministic eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
    echo "[$(date)] CACHED task=${T} evals=[${EVALS}]" | tee -a "${RES}"
    continue
  fi

  python rl_train.py \
    cold_start_checkpoint="${CKPT}" task_index=${T} \
    rl.n_iters=0 \
    rl.eval_rollouts_per_env=5 \
    logging.mode=disabled \
    exp_name="eval_multitask_bc_t${T}" \
    hydra.run.dir="${SCR_ROOT}/imitation/hydra/eval_multitask_bc_t${T}_${SLURM_JOB_ID}" \
    > "${LOG}" 2>&1 || true

  EVALS=$(grep "deterministic eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
  echo "[$(date)] DONE task=${T} evals=[${EVALS}]" | tee -a "${RES}"
done

echo "[$(date)] EVAL DONE job=${SLURM_JOB_ID}" | tee -a "${RES}"
echo "Finished at $(date)"
