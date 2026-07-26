#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100,gpu-v100
#SBATCH --time=04:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=ftmetrics
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.out
#SBATCH --error=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.err

# ============================================================
# GOODCOV/BADENT finetunability metrics (paper Sec 5.2) for one DICE ckpt.
#   sbatch scripts/finetunability_metrics.sbatch <LABEL> <CKPT> [extra overrides...]
# e.g. aln0 ckpts need: ++dice.actor_layernorm=false
# ============================================================
LABEL="${1:?usage: sbatch finetunability_metrics.sbatch <LABEL> <CKPT> [overrides]}"
CKPT="${2:?usage: sbatch finetunability_metrics.sbatch <LABEL> <CKPT> [overrides]}"
EXTRA=("${@:3}")
CONDA_ENV="porygon"; WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/imitation"
CEDAR="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch"
cd "${WORKDIR}" || exit 1; source ~/.bashrc; conda activate "${CONDA_ENV}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MUJOCO_GL=egl EGL_DEVICE_ID=0 HYDRA_FULL_ERROR=1
[ -f "${CKPT}" ] || { echo "ERROR: ckpt missing ${CKPT}"; exit 1; }

JSON="${CEDAR}/ftmetrics_${LABEL}.json"
echo "[$(date)] START ftmetrics ${LABEL} ckpt=${CKPT}"
python scripts/finetunability_metrics.py \
  ++targets="[['${LABEL}','${CKPT}']]" \
  ++task_indices=[8,21,32,53,65,73,75,81] \
  ++episodes_per_task=25 \
  ++num_z=16 \
  ++alphas=[0.5,0.75,0.9] \
  ++nbins=50 \
  ++out_json="${JSON}" \
  hydra.run.dir="${CEDAR}/imitation/hydra/ftmetrics_${LABEL}_${SLURM_JOB_ID}" \
  "${EXTRA[@]}"
echo "[$(date)] DONE ${LABEL} -> ${JSON}"
