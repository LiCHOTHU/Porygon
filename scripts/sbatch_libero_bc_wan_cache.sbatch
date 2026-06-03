#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=06:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=libero_wan_cache
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/dialga_outputs/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/dialga_outputs/imitation/logs/%x_%j.err

# ============================================================
# One-shot Wan-VAE pre-encode for LIBERO-90 BC training.
# At ~10 batches/sec with batch=16 over ~675k (demo, t) windows
# the full pass takes roughly 70 minutes on a single L40S.
# Resume-safe: skips any task whose .h5 already has done=True.
# ============================================================
set -eo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/imitation"
SCR_ROOT="/storage/scratch1/8/lwang831/dialga_outputs/imitation"
PROJ_TMP="/storage/project/r-agarg35-0/lwang831/tmp"

cd "${WORKDIR}" || exit 1
mkdir -p "${SCR_ROOT}/logs" "${PROJ_TMP}/matplotlib"

source ~/.bashrc
conda activate river

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TMPDIR="${PROJ_TMP}"
export MPLCONFIGDIR="${PROJ_TMP}/matplotlib"
export TORCH_HOME="${PROJ_TMP}/torch_hub"
export XDG_CACHE_HOME="${PROJ_TMP}/xdg_cache"
mkdir -p "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${HF_HOME}"

echo "[$(date)] JobID=${SLURM_JOB_ID}  Node=${SLURM_NODELIST}"
nvidia-smi || true

python -u scripts/cache_libero_bc_wan.py \
    --data_dir "${WORKDIR}/data/libero/libero_90" \
    --cache_dir "${SCR_ROOT}/libero_bc_wan_cache" \
    --camera_key agentview_image \
    --batch_size 16 \
    --n_demos 50

echo "[$(date)] done."
