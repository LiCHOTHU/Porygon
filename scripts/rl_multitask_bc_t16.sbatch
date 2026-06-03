#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100
#SBATCH --time=06:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=fm_rl_multitask_bc_t16
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err

# ============================================================
# FM-RL on task 16 starting from the multi-task BC ckpt
# (cold_multitask_lib90/multitask_model_latest.pth). Same v2-tuned
# hyperparams as the single-task d=N RL runs (15 iters, 6 groups,
# G=8, ppo_epochs=4, lr=1e-6, target_kl=0.05). Goal: compare
# multi-task BC -> RL vs single-task BC -> RL at the same task.
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

TASK=16
EXP_NAME_BC="cold_multitask_lib90"
RL_NAME="fm_rl_multitask_bc_t${TASK}"
CKPT="${SCR_ROOT}/imitation/cold_start/libero/libero_90/${EXP_NAME_BC}/multitask_model_latest.pth"
LOG="${SCR_ROOT}/${RL_NAME}.log"
RES="${SCR_ROOT}/rl_multitask_bc_t${TASK}_results.txt"
mkdir -p "${SCR_ROOT}/imitation/logs"

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: multi-task BC ckpt not found at ${CKPT}" | tee -a "${RES}"
  exit 1
fi

N_ITERS=15
EVAL_EVERY=5
EXPECTED=$(( N_ITERS / EVAL_EVERY + 1 ))

# Resume safety: skip if a prior run already finished all evals
if [ -f "${LOG}" ]; then
  NDONE=$(grep -c "deterministic eval success rate" "${LOG}" || true)
  if [ "${NDONE}" -ge "${EXPECTED}" ]; then
    EVALS=$(grep "deterministic eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
    echo "[$(date)] CACHED ${RL_NAME} evals=[${EVALS}]" | tee -a "${RES}"
    exit 0
  fi
fi

echo "[$(date)] START ${RL_NAME} ckpt=${CKPT}" | tee -a "${RES}"

python rl_train.py \
  cold_start_checkpoint="${CKPT}" task_index=${TASK} \
  rl.n_iters=${N_ITERS} \
  rl.group_size=8 \
  rl.inits_per_iter=6 \
  rl.ppo_epochs=4 \
  rl.lr=1.0e-6 \
  rl.target_kl=0.05 \
  rl.eval_interval=${EVAL_EVERY} \
  rl.eval_rollouts_per_env=20 \
  logging.mode=disabled \
  exp_name=${RL_NAME} \
  hydra.run.dir="${SCR_ROOT}/imitation/hydra/${RL_NAME}_${SLURM_JOB_ID}" \
  > "${LOG}" 2>&1 || true

EVALS=$(grep "deterministic eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
echo "[$(date)] DONE ${RL_NAME} evals=[${EVALS}]" | tee -a "${RES}"
echo "Finished at $(date)"
