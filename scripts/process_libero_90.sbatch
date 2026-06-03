#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=libero90_proc
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err

# ============================================================
# Preprocess LIBERO-90 raw demos -> repo format (depth, abs_actions,
# calibration) by replaying in sim. Resumable: already-processed tasks
# are skipped, so requeue/preemption is safe.
# Output: data/libero/libero_90 (symlinked to /storage/scratch1/8/lwang831/...)
# ============================================================
CONDA_ENV="porygon"
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/imitation"
SCRIPT="${WORKDIR}/scripts/process_libero_data.py"

cd "${WORKDIR}" || exit 1

echo "================ Job Info ================"
echo "JobID:  ${SLURM_JOB_ID}"
echo "Node:   ${SLURM_NODELIST}"
echo "CWD:    $(pwd)"
echo "Start:  $(date)"
echo "========================================="

# ============================================================
# Environment
# (conda activate runs porygon's activate.d, which redirects the cluster's
#  non-writable cache vars TORCH_HOME/HF_HOME/etc. to ~/.cache — must come
#  AFTER `source ~/.bashrc` so it wins.)
# ============================================================
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

# ============================================================
# Process ALL 90 tasks (no task_nums => full benchmark; resumes/skips done).
# Redirect hydra's run dir to scratch so nothing is written under home.
# ============================================================
python "${SCRIPT}" \
    hydra.run.dir="/storage/scratch1/8/lwang831/imitation/hydra/libero90_proc_${SLURM_JOB_ID}"

echo "Finished at $(date)"
