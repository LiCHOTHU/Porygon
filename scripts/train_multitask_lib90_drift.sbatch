#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=drift_multitask_lib90
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err

# ============================================================
# Multi-task BC pretrain: PolicyDrifting on full LIBERO-90 (all 90 tasks,
# 50 demos/task = 4500 demos). Same model architecture as fm_policy_S
# (18.27M params, identical), same chunk=16/horizon=8 BC-optimal setting
# from the 2026-05-30 sweep. ONLY the training algorithm differs:
#   * Loss: per-particle attractive/repulsive drift force in flattened
#     action space (positives = own demo, negatives = other-batch demos),
#     multi-temperature symmetric-softmax affinity, stop-grad goal,
#     MSE regression -- following "Generative Modeling via Drifting"
#     (arXiv:2602.04770).
#   * Inference: 1-step Euler from noise, a = clamp(x_0 + v(x_0, 0, cond), -1, 1).
# Drop-in apples-to-apples vs cold_multitask_lib90 (FM): same data, same
# arch, same compute -- only the algorithm changes.
# Resumable via training.resume=true (default) + sbatch --requeue.
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
EXP_NAME="drift_multitask_lib90"
N_EPOCHS=30
CHUNK=16
HORIZON=8

mkdir -p "${SCR_ROOT}/imitation/logs"

# NOTE: train.py has no default config_name -- must pass --config-name explicitly.
python train.py --config-name=train \
    task=libero algo=policy_drifting_S \
    algo.chunk_size=${CHUNK} algo.action_horizon=${HORIZON} \
    training.n_epochs=${N_EPOCHS} \
    rollout.enabled=false \
    logging.mode=disabled \
    exp_name=${EXP_NAME} \
    data_prefix="${SCR_ROOT}/imitation/data" \
    output_prefix="${SCR_ROOT}/imitation/cold_start" \
    hydra.run.dir="${SCR_ROOT}/imitation/hydra/${EXP_NAME}_${SLURM_JOB_ID}"

echo "Finished at $(date)"
