#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-v100,gpu-a100,gpu-h100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=drift_eval_multitask_lib90
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err

# ============================================================
# Multi-task deterministic eval of drift_multitask_lib90 ckpt
# on all 90 LIBERO-90 tasks. 10 rollouts/task, 5 parallel envs
# -> 900 total rollouts. Per-task resumable via results file.
# Storage: /storage/scratch1/8/lwang831 only.
# Not on L40S -- preprocessing/rollouts are sim-bound (mujoco
# offscreen render), so V100/A100 is fine and conserves L40S.
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

EXP_DIR="${SCR_ROOT}/imitation/cold_start/libero/libero_90/drift_multitask_lib90"
CKPT_NAME="multitask_model_latest.pth"
RESULTS="${SCR_ROOT}/imitation/eval_results/drift_multitask_lib90_per_task.tsv"
mkdir -p "$(dirname ${RESULTS})" "${SCR_ROOT}/imitation/logs"

python scripts/eval_drift_multitask_lib90.py \
    --exp_dir "${EXP_DIR}" \
    --ckpt_name "${CKPT_NAME}" \
    --results_file "${RESULTS}" \
    --rollouts_per_env 10 \
    --num_parallel_envs 5 \
    --start_task 0 \
    --end_task 90

echo "Finished at $(date)"
