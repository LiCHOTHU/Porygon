#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s,gpu-a100,gpu-h100
#SBATCH --time=05:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=sib_libero
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --requeue
#SBATCH --array=0-4
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%A_%a.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%A_%a.err

# Does the spectral penalty help a visuomotor policy?  (DEVLOG Entry 15)
#
# The metric is ROLLOUT SUCCESS RATE in the simulator, not training loss.  A
# policy's training loss says nothing about whether it completes the task, and
# the whole question here is generalization -- so rollouts run during training
# (clean scenes, the control) and scripts/sib_eval_shift.py measures the
# held-out camera shift afterwards, which is where the claim lives.
#
#   0 vanilla   1 sib   2 mixup   3 sam   4 dropout
#
# output_prefix MUST point at scratch.  config/train.yaml defaults it to the
# relative './experiments', which lands inside the repo -- i.e. inside the 20 GB
# home quota -- and a 5-arm array at ~430 MB per checkpoint fills it and then
# fails mid-write.  Nothing about training belongs in home.
#
# 15 epochs, not 30: vanilla already reached 0.940 clean success with 20/20 tasks
# solved at epoch 15, so further epochs buy nothing measurable, and SIB cannot
# afford them -- it costs ~23x vanilla per epoch (33.6 vs 1.44 min, double
# backward plus the math SDPA fallback), which is what blew the first attempt
# past the 8h wall.  Equal epochs across arms is the point: a fair comparison
# cannot hand vanilla 30 epochs and SIB 10.
#
# Rollout every 5 epochs rather than only at the end: embers is preemptible and
# these arms were preempted at 2.8-4.8h last time, so a success rate has to exist
# at an intermediate epoch or preemption costs the entire measurement.
# `resume: true` plus the per-epoch multitask_model_latest.pth means a requeue
# picks up where it left off.
#
# The rollout is scoped to the 20 trained tasks.  task_subset used to be read only
# by the dataset, so the runner evaluated all 90 tasks of libero_90 -- 70 never
# trained on -- which is both the wrong number and ~4.5x the rollout budget.
# LiberoRunner now takes task_subset (runner.py).
#
# NB the account: --account=gts-agarg35 silently falls back to V100, ~3.4x slower
# for libero-scale multitask training (DEVLOG 2026-06-03).

# NB: no `set -u` -- /etc/bashrc references BASHRCSOURCED unbound, so `source ~/.bashrc`
# dies instantly under it.  This cost one submission.
set -o pipefail
ARMS=(fm_policy_S fm_sib_S fm_mixup_S fm_sam_S fm_dropout_S)
NAMES=(vanilla sib mixup sam dropout)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}

CONDA_ENV="porygon"
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/imitation"
SCR_ROOT="/storage/scratch1/8/lwang831/imitation"
cd "${WORKDIR}" || exit 1
source ~/.bashrc
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MUJOCO_GL=egl
export EGL_DEVICE_ID=0
# these default to unwritable roots on the compute nodes and break the encoder
export HF_HOME=/storage/project/r-agarg35-0/lwang831/tmp/hf
export TRANSFORMERS_CACHE=$HF_HOME
export MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/mpl
export TORCH_HOME=/storage/project/r-agarg35-0/lwang831/tmp/torch

echo "=== ${NAME} (${ARM}) on $(hostname) at $(date -Is) ==="
nvidia-smi --query-gpu=name --format=csv,noheader

# 20 tasks of libero_90: enough diversity for a multitask policy, small enough
# that five arms finish inside the 8h embers wall.  libero_10 is NOT usable --
# its processed directory is empty.
python train.py --config-name=train task=libero algo="${ARM}" \
    output_prefix="${SCR_ROOT}/experiments" \
    'task.task_subset=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]' \
    task.demos_per_env=50 \
    training.n_epochs=15 \
    rollout.enabled=true \
    rollout.interval=5 \
    rollout.rollouts_per_env=5 \
    logging.mode=disabled \
    exp_name="sib_libero_${NAME}"

echo "=== done $(date -Is) ==="
