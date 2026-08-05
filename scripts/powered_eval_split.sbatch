#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100,gpu-v100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=powered_eval_one
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%j.out
#SBATCH --error=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%j.err

# Powered eval of a SINGLE checkpoint (one target) so it fits in one 8h segment.
# 100 rollouts/env x 3 seeds (300/task) over hard-8 = 2400 rollouts (~7h).
# Pass via --export:  CELL, LABEL, CKPT
# Gap vs BC is computed offline against the already-stored powered_eval_2x2_${CELL}.json BC row.

CONDA_ENV="porygon"
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/imitation"
CEDAR="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch"
cd "${WORKDIR}" || exit 1
source ~/.bashrc
conda activate "${CONDA_ENV}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MUJOCO_GL=egl EGL_DEVICE_ID=0
nvidia-smi || true

LOG="${CEDAR}/powered_eval_split_${CELL}_${LABEL}_s${BSEED}.log"
JSON="${CEDAR}/powered_eval_split_${CELL}_${LABEL}_s${BSEED}.json"

echo "[$(date)] START powered_eval_one CELL=${CELL} LABEL=${LABEL} CKPT=${CKPT}"

python scripts/powered_eval.py \
  ++targets="[['${LABEL}','${CKPT}']]" \
  ++task_indices=[8,21,32,53,65,73,75,81] \
  ++n_seeds=1 \
  ++base_seed=${BSEED} \
  ++rollouts_per_env=100 \
  ++out_json="${JSON}" \
  hydra.run.dir="${CEDAR}/imitation/hydra/powered_eval_split_${CELL}_${LABEL}_${SLURM_JOB_ID}" \
  >> "${LOG}" 2>&1 || true

echo "[$(date)] DONE powered_eval_one CELL=${CELL} LABEL=${LABEL}"
