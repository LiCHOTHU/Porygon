#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-v100,gpu-a100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=libero10_proc
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err

# ============================================================
# Preprocess LIBERO-Long (libero_10) raw demos -> repo format (depth,
# abs_actions, calibration, hand matrices) by replaying in sim.
# Resumable: already-processed tasks are skipped.
# Input  (symlink): /storage/scratch1/8/lwang831/imitation/data/libero/libero_10_unprocessed/
#                   -> /storage/cedar/.../LIBERO-datasets/libero_10/
# Output:           /storage/scratch1/8/lwang831/imitation/data/libero/libero_10/
# ~2.5 min/task * 10 tasks ~= 25 min on L40S.
# ============================================================
CONDA_ENV="porygon"
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/imitation"
SCRIPT="${WORKDIR}/scripts/process_libero_data.py"
SCR_ROOT="/storage/scratch1/8/lwang831"

cd "${WORKDIR}" || exit 1

echo "================ Job Info ================"
echo "JobID:  ${SLURM_JOB_ID}"
echo "Node:   ${SLURM_NODELIST}"
echo "CWD:    $(pwd)"
echo "Start:  $(date)"
echo "========================================="

source ~/.bashrc
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Headless GPU rendering for mujoco/robosuite offscreen depth+rgb
export MUJOCO_GL=egl
export EGL_DEVICE_ID=0

nvidia-smi || true
python -V
which python

# Override benchmark_name to libero_10 (default is libero_90 via config/task/libero.yaml).
# data_prefix is overridden to scratch1 so the symlink + outputs resolve correctly.
python "${SCRIPT}" \
    task.benchmark_name=libero_10 \
    data_prefix="${SCR_ROOT}/imitation/data" \
    hydra.run.dir="${SCR_ROOT}/imitation/hydra/libero10_proc_${SLURM_JOB_ID}"

echo "Finished at $(date)"
