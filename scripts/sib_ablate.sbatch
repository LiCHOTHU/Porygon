#!/usr/bin/env bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1
#SBATCH -p gpu-l40s,gpu-a100,gpu-h100
#SBATCH --time=04:00:00 --mem-per-gpu=48G
#SBATCH --job-name=sib_noV --qos=embers --account=gts-agarg35-ideas_l40s --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err

# SIB with the bottleneck factor V removed -- penalty becomes sqrt(S_k) alone.
# Decides whether the information-bottleneck framing is load-bearing or whether
# the method is really a reliance-concentration penalty wearing that name.
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
python train.py --config-name=train task=libero algo=fm_sib_noV_S \
    output_prefix=/storage/scratch1/8/lwang831/imitation/experiments \
    'task.task_subset=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]' \
    task.demos_per_env=50 training.n_epochs=15 \
    rollout.enabled=true rollout.interval=15 rollout.rollouts_per_env=5 \
    logging.mode=disabled exp_name=sib_libero_noV
rc=$?; echo "=== done rc=${rc} $(date -Is) ==="; exit $rc
