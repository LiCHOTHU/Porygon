#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100,gpu-v100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=eval_hard8_t1
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.out
#SBATCH --error=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.err

# ============================================================
# Powered eval of ONE checkpoint on hard-8: 100 rollouts x 3 seeds per task.
#   sbatch scripts/eval_hard8_tier1.sbatch <LABEL> <CKPT> [DICE_EVAL_STRATEGY]
# DICE ckpts default to the faithful protocol (max_q_min over 10 latents,
# matching the training-loop eval and the official DICE-RL eval); pass
# "single" to eval the raw one-draw policy instead. Non-DICE ckpts ignore it.
# ============================================================
LABEL="${1:?usage: sbatch eval_hard8_tier1.sbatch <LABEL> <CKPT> [STRATEGY]}"
CKPT="${2:?usage: sbatch eval_hard8_tier1.sbatch <LABEL> <CKPT> [STRATEGY]}"
STRAT="${3:-max_q_min}"
EXTRA=("${@:4}")   # optional extra hydra overrides, e.g. ++dice.actor_layernorm=false
CONDA_ENV="porygon"; WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/imitation"
CEDAR="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch"
cd "${WORKDIR}" || exit 1; source ~/.bashrc; conda activate "${CONDA_ENV}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MUJOCO_GL=egl EGL_DEVICE_ID=0 HYDRA_FULL_ERROR=1
nvidia-smi || true
[ -f "${CKPT}" ] || { echo "ERROR: ckpt missing ${CKPT}"; exit 1; }

JSON="${CEDAR}/eval_hard8_t1_${LABEL}.json"
echo "[$(date)] START powered eval ${LABEL} strat=${STRAT} ckpt=${CKPT}"
python scripts/powered_eval.py \
  ++targets="[['${LABEL}','${CKPT}']]" \
  ++task_indices=[8,21,32,53,65,73,75,81] \
  ++n_seeds=3 \
  ++base_seed=1000 \
  ++rollouts_per_env=100 \
  ++dice_eval_strategy=${STRAT} \
  ++dice_eval_samples=10 \
  ++out_json="${JSON}" \
  hydra.run.dir="${CEDAR}/imitation/hydra/eval_hard8_t1_${LABEL}_${SLURM_JOB_ID}" \
  "${EXTRA[@]}"
echo "[$(date)] DONE ${LABEL} -> ${JSON}"
