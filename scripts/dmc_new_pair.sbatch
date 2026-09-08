#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100,gpu-v100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=32G
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.out
#SBATCH --error=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.err
# Build a new DMC dense/sparse PAIR for the scope test:
#   1. SAC on the DENSE task, harvesting demos from several checkpoints
#   2. re-roll those same checkpoints in the SPARSE task for exact sparse labels
# usage: sbatch dmc_new_pair.sbatch <domain> <dense_task> <sparse_task>
set -x
DOMAIN="${1:?domain}"; DENSE="${2:?dense task}"; SPARSE="${3:?sparse task}"; STEPS="${4:-60000}"
source ~/.bashrc
conda activate /storage/project/r-agarg35-0/lwang831/conda/envs/dice-rl
export MUJOCO_GL=egl EGL_DEVICE_ID=0
export XDG_CACHE_HOME=/storage/scratch1/8/lwang831/.cache
export NUMBA_CACHE_DIR=$XDG_CACHE_HOME/numba
OUT=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/dmc_base
mkdir -p "$OUT"
cd /storage/home/hcoda1/8/lwang831/workspace/imitation || exit 1
nvidia-smi || true

echo "[$(date)] STAGE 1: SAC + demo harvest on ${DOMAIN} ${DENSE}"
python scripts/dmc_make_base_data.py "${DOMAIN}" "${DENSE}" "${OUT}" "${STEPS}" || exit 1

echo "[$(date)] STAGE 2: sparse re-harvest on ${DOMAIN} ${SPARSE}"
python scripts/dmc_relabel_sparse.py "${DOMAIN}" "${DENSE}" "${SPARSE}" "${OUT}" || exit 1

echo "[$(date)] DONE pair ${DOMAIN}/${DENSE}+${SPARSE}"
