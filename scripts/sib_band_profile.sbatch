#!/usr/bin/env bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1
#SBATCH -p gpu-a100,gpu-l40s,gpu-rtx6000,gpu-v100
#SBATCH --time=00:40:00 --mem-per-gpu=48G
#SBATCH --job-name=sib_bands --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err
set -o pipefail
cd /storage/home/hcoda1/8/lwang831/workspace/imitation || exit 1
source ~/.bashrc
conda activate porygon
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MUJOCO_GL=egl EGL_DEVICE_ID=0
export HF_HOME=/storage/project/r-agarg35-0/lwang831/tmp/hf
export TRANSFORMERS_CACHE=$HF_HOME
export MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/mpl
export TORCH_HOME=/storage/project/r-agarg35-0/lwang831/tmp/torch
if ! python -c "import torch; torch.zeros(64,64,device='cuda').matmul(torch.zeros(64,64,device='cuda')); torch.cuda.synchronize()" 2>/dev/null; then
    echo "!!! GPU unhealthy on $(hostname) -- requeueing"; scontrol requeue "${SLURM_JOB_ID}"; sleep 60; exit 1
fi
python scripts/sib_band_profile.py --batches 8
echo "=== done $(date -Is) ==="
