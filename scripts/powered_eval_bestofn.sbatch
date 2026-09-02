#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100,gpu-v100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.out
#SBATCH --error=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.err
#
# Powered evaluation with BEST-OF-N critic selection (dice_eval_strategy=max_q_min),
# the protocol DICE-RL's own LIBERO evaluation uses. Our tables use a single noise
# draw, under which a near-zero residual reproduces the base -- so a fine-tuned
# number can look identical to its base even when the residual learned something.
# This re-scores existing checkpoints under the selection protocol to test whether
# the multitask "nothing works" result is real or an artefact of how we evaluate.
#
# Pass via --export:  LABEL, CKPT
set -x
cd /storage/home/hcoda1/8/lwang831/workspace/imitation || exit 1
source ~/.bashrc
conda activate porygon
export XDG_CACHE_HOME=/storage/scratch1/8/lwang831/.cache
export NUMBA_CACHE_DIR=$XDG_CACHE_HOME/numba
export MUJOCO_GL=egl EGL_DEVICE_ID=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
CEDAR=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch
nvidia-smi || true

python scripts/powered_eval.py \
  ++targets="[['${LABEL}','${CKPT}']]" \
  ++task_indices=[8,21,32,53,65,73,75,81] \
  ++n_seeds=3 ++base_seed=1000 ++rollouts_per_env=100 \
  ++dice_eval_strategy=max_q_min ++dice_eval_samples=16 \
  ++out_json="${CEDAR}/powered_eval_bon_${LABEL}.json" \
  hydra.run.dir="${CEDAR}/imitation/hydra/pe_bon_${LABEL}_${SLURM_JOB_ID}"
echo "DONE ${LABEL}"
