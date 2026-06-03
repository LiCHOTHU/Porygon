#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=fm_rl_multitask_20tasks_filtered
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err

# ============================================================
# Multi-task FM-RL on the 20 picked LIBERO-90 tasks, REVISION 2.
# Adds SimpleVLA-RL filter_accuracy [0.1, 0.9]: drop saturated
# all-1 and impossible all-0 groups *before* the PPO update so
# the gradient is driven by groups with real within-group reward
# variance. Doubles inits_per_iter (1 -> 2) so post-filter we
# still have rows to learn from.
#
# Rationale: revision 1 (job 9362908) RL'd a BC checkpoint already
# at 80.2% on these 20 tasks. Most per-(task,init) groups landed
# at 7/8 or 8/8 -- where GRPO's within-group advantage is near-zero
# -- so the curve was flat (BC 0.802 -> RL 0.760). SimpleVLA-RL's
# 17 -> 92 on libero_10 used the same single-policy, sparse-reward,
# uniform-shuffle setup BUT (a) started at 17% (groups land in the
# usable band naturally), (b) ran filter_accuracy, (c) 3x our
# per-iter batch. Filter is the cheapest of those three to copy.
#
# Budget (fits embers 8h):
#   n_iters=6, eval_interval=3 -> evals at 0/3/6 + end-of-loop=6
#   group_size=8, inits_per_iter=2 -> 320 collect rollouts/iter
#   filter [0.1, 0.9]
#   ppo_epochs=4, lr=1e-6, target_kl=0.05
#   eval_rollouts_per_env=15 -> 300 eval rollouts/eval
# Wall-clock est on gpu-a100: 1920 collect + 1200 eval = 3120 rolls,
#   ~7.0h at ~440 rolls/hr (matches revision 1's observed throughput).
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
RL_NAME="fm_rl_multitask_20tasks_filtered"
CKPT="${SCR_ROOT}/imitation/cold_start/libero/libero_90/${EXP_NAME_BC}/multitask_model_latest.pth"
LOG="${SCR_ROOT}/${RL_NAME}.log"
RES="${SCR_ROOT}/rl_multitask_20tasks_filtered_results.txt"
mkdir -p "${SCR_ROOT}/imitation/logs"

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: multi-task BC ckpt not found at ${CKPT}" | tee -a "${RES}"
  exit 1
fi

N_ITERS=6
EVAL_EVERY=3
# evals at 0, 3, 6 in-loop + 1 final post-loop = 4 expected
EXPECTED=4

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

# task_indices = same bottom-20 by BC success used in revision 1.
python rl_train.py \
  cold_start_checkpoint="${CKPT}" \
  'task_indices=[6,7,8,9,13,14,21,23,32,36,43,45,51,52,53,69,73,75,78,82]' \
  rl.n_iters=${N_ITERS} \
  rl.group_size=8 \
  rl.inits_per_iter=2 \
  rl.ppo_epochs=4 \
  rl.lr=1.0e-6 \
  rl.target_kl=0.05 \
  rl.filter_accuracy=true \
  rl.filter_low=0.1 \
  rl.filter_high=0.9 \
  rl.eval_interval=${EVAL_EVERY} \
  rl.eval_rollouts_per_env=15 \
  logging.mode=disabled \
  exp_name=${RL_NAME} \
  hydra.run.dir="${SCR_ROOT}/imitation/hydra/${RL_NAME}_${SLURM_JOB_ID}" \
  > "${LOG}" 2>&1 || true

EVALS=$(grep "deterministic eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
echo "[$(date)] DONE ${RL_NAME} evals=[${EVALS}]" | tee -a "${RES}"

# Per-task at the final eval iter.
LAST_PT=$(grep "per-task:" "${LOG}" | tail -1)
echo "final per-task: ${LAST_PT}" | tee -a "${RES}"

# Filter activation stats.
FILTER_N=$(grep -c "] filter \[" "${LOG}" || true)
echo "filter activations: ${FILTER_N}" | tee -a "${RES}"
echo "filter (last 3 iters):" | tee -a "${RES}"
grep "] filter \[" "${LOG}" | tail -3 | tee -a "${RES}"

echo "Finished at $(date)"
