#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=fm_dice_single_task
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err

# ============================================================
# DICE-RL-Robot-style fine-tuning on a SINGLE LIBERO-90 task.
# First ladder: validate the actor/critic/teacher/replay spine on one task
# before scaling to multi-task. Spec mirrors their README defaults where useful:
#   bc_loss_weight=100, ensemble_size=2, conservative=min, gamma=0.99, tau=0.005,
#   batch_size=256, gradient_steps per iter=200.
#
# Pick TASK_INDEX to match a task where BC is in the partial-success regime so
# RL has signal (avoid 0% and avoid saturated 100%). Defaults to task 0 -- override
# on submission: sbatch --export=TASK_INDEX=43 scripts/dice_train_single_task.sbatch
# ============================================================

CONDA_ENV="porygon"
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/imitation"
SCR_ROOT="/storage/scratch1/8/lwang831"
cd "${WORKDIR}" || exit 1

echo "================ Job Info ================"
echo "JobID:  ${SLURM_JOB_ID}"
echo "Node:   ${SLURM_NODELIST}"
echo "Start:  $(date)"
echo "========================================="

source ~/.bashrc
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MUJOCO_GL=egl
export EGL_DEVICE_ID=0

nvidia-smi || true
python -V

EXP_NAME_BC="${EXP_NAME_BC:-cold_multitask_lib90}"   # override for single-task cold-starts
TASK_INDEX=${TASK_INDEX:-43}     # default: KITCHEN_SCENE5_put_the_ketchup_... (mid-BC)
# Seed precedence: SLURM_ARRAY_TASK_ID > env SEED > 10000 (existing first-run seed).
SEED="${SLURM_ARRAY_TASK_ID:-${SEED:-10000}}"
RUN_NAME="fm_dice_${EXP_NAME_BC#cold_}_t${TASK_INDEX}_s${SEED}"
CKPT="${SCR_ROOT}/imitation/cold_start/libero/libero_90/${EXP_NAME_BC}/multitask_model_latest.pth"
LOG="${SCR_ROOT}/${RUN_NAME}.log"
RES="${SCR_ROOT}/dice_results.txt"
mkdir -p "${SCR_ROOT}/imitation/logs"

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: BC cold-start ckpt not found at ${CKPT}" | tee -a "${RES}"
  exit 1
fi

echo "[$(date)] START ${RUN_NAME} task=${TASK_INDEX} seed=${SEED} ckpt=${CKPT}" | tee -a "${RES}"

python dice_train.py \
  cold_start_checkpoint="${CKPT}" \
  task_index=${TASK_INDEX} \
  seed=${SEED} \
  dice.n_iters=50 \
  dice.episodes_per_iter=4 \
  dice.warmup_episodes=20 \
  dice.gradient_steps=200 \
  dice.eval_interval=5 \
  dice.save_interval=10 \
  dice.batch_size=256 \
  dice.replay_size=50000 \
  dice.bc_loss_weight=100.0 \
  dice.actor_lr=3.0e-5 \
  dice.critic_lr=3.0e-4 \
  dice.gamma=0.99 \
  dice.tau=0.005 \
  rollout.rollouts_per_env=20 \
  logging.mode=disabled \
  exp_name=${RUN_NAME} \
  hydra.run.dir="${SCR_ROOT}/imitation/hydra/${RUN_NAME}_${SLURM_JOB_ID}" \
  > "${LOG}" 2>&1 || true

EVALS=$(grep "eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
BC_BASE=$(grep "BC baseline success rate" "${LOG}" | grep -oE "[0-9.]+$" | head -1)
echo "[$(date)] DONE ${RUN_NAME} seed=${SEED} bc=${BC_BASE} dice_evals=[${EVALS}]" | tee -a "${RES}"

echo "Finished at $(date)"
