#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100,gpu-v100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=rl_hard8_drift_vanilla
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err

# ============================================================
# "From Prior to Pro" RL on the *drift* policy.
# Implementation note: drift and FM share the same DiTNoiseNet architecture and
# rl_train.py is policy-generic. With num_inference_steps=1, FM-RL's stochastic
# Euler step reduces to drift's 1-step rule exactly (mu = x + dt*v with dt=1,
# t=0 == a = x_0 + v(x_0, drift_t=0, cond)). We therefore "swap the drifting
# model in" by:
#   - using algo=fm_policy_rl_S (instantiates FlowMatchingPolicyRL)
#   - overriding algo.num_inference_steps=1
#   - pointing cold_start_checkpoint at the drift BC checkpoint
# Smoke test (scripts/smoke_drift_rl_swap.py) confirmed:
#   - state-dict loads cleanly (0 missing, 0 unexpected keys)
#   - logp_new == logp_old at unchanged params (exact)
#   - ratio=1.0, approx_kl=0, clipfrac=0 (exact)
#
# 8 hardest LIBERO-90 tasks (drop t51 benchmark-broken):
#   indices: 8, 21, 32, 53, 65, 73, 75, 81
# Budget mirrors the FM run (rl_hard8_fm_vanilla.sbatch); fits embers 8h.
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

EXP_NAME_BC="drift_multitask_lib90"
RL_NAME="rl_hard8_drift_vanilla"
CKPT="${SCR_ROOT}/imitation/cold_start/libero/libero_90/${EXP_NAME_BC}/multitask_model_latest.pth"
LOG="${SCR_ROOT}/${RL_NAME}.log"
RES="${SCR_ROOT}/${RL_NAME}_results.txt"
mkdir -p "${SCR_ROOT}/imitation/logs"

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: drift multi-task BC ckpt not found at ${CKPT}" | tee -a "${RES}"
  exit 1
fi

N_ITERS=6
EVAL_EVERY=3
EXPECTED=4  # evals at 0, 3, 6 + final

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
  algo.num_inference_steps=1 \
  cold_start_checkpoint="${CKPT}" \
  'task_indices=[8,21,32,53,65,73,75,81]' \
  rl.n_iters=${N_ITERS} \
  rl.group_size=8 \
  rl.inits_per_iter=2 \
  rl.ppo_epochs=4 \
  rl.lr=1.0e-6 \
  rl.target_kl=0.05 \
  rl.filter_accuracy=false \
  rl.eval_interval=${EVAL_EVERY} \
  rl.eval_rollouts_per_env=15 \
  logging.mode=disabled \
  exp_name=${RL_NAME} \
  hydra.run.dir="${SCR_ROOT}/imitation/hydra/${RL_NAME}_${SLURM_JOB_ID}" \
  > "${LOG}" 2>&1 || true

EVALS=$(grep "deterministic eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
echo "[$(date)] DONE ${RL_NAME} evals=[${EVALS}]" | tee -a "${RES}"
LAST_PT=$(grep "per-task:" "${LOG}" | tail -1)
echo "final per-task: ${LAST_PT}" | tee -a "${RES}"

echo "Finished at $(date)"
