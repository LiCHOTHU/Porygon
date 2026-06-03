#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=fm_rl_multitask_20tasks
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err

# ============================================================
# Multi-task FM-RL on 20 picked LIBERO-90 tasks (SimpleVLA-RL
# style: one policy RL-trained on the suite jointly, sparse 0/1
# reward, GRPO advantage with within-(task,init) baseline).
# Cold-start: cold_multitask_lib90/multitask_model_latest.pth.
# Hyperparams kept from the single-task v2-tuned sweep, with
# inits_per_iter interpreted as inits *per task* per iter:
#   n_iters=10, eval_interval=5 -> evals at 0/5/10
#   group_size=8, inits_per_iter=1 (per task -> 20 inits/iter)
#   ppo_epochs=4, lr=1e-6, target_kl=0.05
#   eval_rollouts_per_env=20 -> 400 eval rollouts per eval
# Wall-clock est on gpu-a100: ~4-5h (1600 collect + 1200 eval rolls).
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
RL_NAME="fm_rl_multitask_20tasks"
CKPT="${SCR_ROOT}/imitation/cold_start/libero/libero_90/${EXP_NAME_BC}/multitask_model_latest.pth"
LOG="${SCR_ROOT}/${RL_NAME}.log"
RES="${SCR_ROOT}/rl_multitask_20tasks_results.txt"
mkdir -p "${SCR_ROOT}/imitation/logs"

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: multi-task BC ckpt not found at ${CKPT}" | tee -a "${RES}"
  exit 1
fi

N_ITERS=10
EVAL_EVERY=5
EXPECTED=$(( N_ITERS / EVAL_EVERY + 1 ))

# Resume safety: if all expected evals are already in the log, skip.
if [ -f "${LOG}" ]; then
  NDONE=$(grep -c "deterministic eval success rate" "${LOG}" || true)
  if [ "${NDONE}" -ge "${EXPECTED}" ]; then
    EVALS=$(grep "deterministic eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
    echo "[$(date)] CACHED ${RL_NAME} evals=[${EVALS}]" | tee -a "${RES}"
    exit 0
  fi
fi

echo "[$(date)] START ${RL_NAME} ckpt=${CKPT}" | tee -a "${RES}"

# task_indices = bottom-20 by per-task BC success on full LIBERO-90 eval.
python rl_train.py \
  cold_start_checkpoint="${CKPT}" \
  'task_indices=[6,7,8,9,13,14,21,23,32,36,43,45,51,52,53,69,73,75,78,82]' \
  rl.n_iters=${N_ITERS} \
  rl.group_size=8 \
  rl.inits_per_iter=1 \
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

# Pull final per-task line for the last eval iter.
LAST_PT=$(grep "per-task:" "${LOG}" | tail -1)
echo "final per-task: ${LAST_PT}" | tee -a "${RES}"

echo "Finished at $(date)"
