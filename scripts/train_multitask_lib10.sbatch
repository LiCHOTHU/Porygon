#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=fm_multitask_lib10
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err

# ============================================================
# Multi-task BC pretrain: FM policy on full LIBERO-Long / libero_10
# (10 long-horizon tasks, 50 demos/task = 500 demos). chunk=16/horizon=8
# (BC-optimal from the 2026-05-30 sweep). No per-iter rollout eval (too
# expensive); evaluate separately afterwards. Resumable via
# training.resume=true (default) + sbatch --requeue.
# Mirror of train_multitask_lib90.sbatch, only task.benchmark_name +
# exp_name + job-name changed.
# Storage: everything under /storage/scratch1/8/lwang831 (never $HOME).
# ============================================================
CONDA_ENV="porygon"
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/imitation"
SCR_ROOT="/storage/scratch1/8/lwang831"
cd "${WORKDIR}" || exit 1

echo "================ Job Info ================"
echo "JobID:  ${SLURM_JOB_ID}"
echo "Node:   ${SLURM_NODELIST}"
echo "CWD:    $(pwd)"
echo "Start:  $(date)"
echo "========================================="

# ------------------------------------------------------------
# Environment (porygon's activate.d remaps cluster RO cache vars)
# ------------------------------------------------------------
source ~/.bashrc
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Headless rendering (in case any LIBERO env touches it during init)
export MUJOCO_GL=egl
export EGL_DEVICE_ID=0

nvidia-smi || true
python -V
which python

# ------------------------------------------------------------
# Training config
# ------------------------------------------------------------
EXP_NAME="cold_multitask_lib10"
N_EPOCHS=30
CHUNK=16
HORIZON=8

mkdir -p "${SCR_ROOT}/imitation/logs"

# NOTE: train.py has no default config_name — must pass --config-name explicitly.
python train.py --config-name=train \
    task=libero task.benchmark_name=libero_10 algo=fm_policy_S \
    algo.chunk_size=${CHUNK} algo.action_horizon=${HORIZON} \
    training.n_epochs=${N_EPOCHS} \
    rollout.enabled=false \
    logging.mode=disabled \
    exp_name=${EXP_NAME} \
    data_prefix="${SCR_ROOT}/imitation/data" \
    output_prefix="${SCR_ROOT}/imitation/cold_start" \
    hydra.run.dir="${SCR_ROOT}/imitation/hydra/${EXP_NAME}_${SLURM_JOB_ID}"

echo "Finished at $(date)"
