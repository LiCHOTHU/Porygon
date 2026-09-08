#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100,gpu-v100,gpu-l40s
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=32G
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.out
#SBATCH --error=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.err
# Build a DENSE-ONLY DMC task base: SAC teacher + demo harvest, no sparse twin.
# For the harder robots (humanoid, quadruped, manipulator) that have no sparse
# variant in the suite; env_steps scales with difficulty.
# usage: sbatch dmc_new_task.sbatch <domain> <task> <env_steps>
set -x
DOMAIN="${1:?domain}"; TASK="${2:?task}"; STEPS="${3:-400000}"
source ~/.bashrc
conda activate /storage/project/r-agarg35-0/lwang831/conda/envs/dice-rl
export MUJOCO_GL=egl EGL_DEVICE_ID=0
export XDG_CACHE_HOME=/storage/scratch1/8/lwang831/.cache
export NUMBA_CACHE_DIR=$XDG_CACHE_HOME/numba
OUT=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/dmc_base
mkdir -p "$OUT"
cd /storage/home/hcoda1/8/lwang831/workspace/imitation || exit 1
nvidia-smi || true
python scripts/dmc_make_base_data.py "${DOMAIN}" "${TASK}" "${OUT}" "${STEPS}" || exit 1
echo "[$(date)] DONE ${DOMAIN}/${TASK}"
