#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100,gpu-v100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=rl_hard8_fm_vanilla
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err

# ============================================================
# "From Prior to Pro" RL (vanilla FM-RL, no SimpleVLA filter) on
# the 8 hardest LIBERO-90 tasks (per the 2026-06-03 head-to-head
# eval: min(drift,FM)<=0.7 excluding t51 which is benchmark-broken).
# Initializes from cold_multitask_lib90 BC checkpoint.
#
# Hard-task indices: 8, 21, 32, 53, 65, 73, 75, 81
#   t8  KITCHEN_SCENE1_open_top_drawer_and_put_...   (FM 0.60)
#   t21 KITCHEN_SCENE3_turn_on_stove_and_put_pan     (FM 0.50)
#   t32 KITCHEN_SCENE5_put_ketchup_in_top_drawer     (FM 0.70)
#   t53 LIVING_ROOM_SCENE2_pick_up_oj_in_basket      (FM 0.60)
#   t65 LIVING_ROOM_SCENE5_put_red_mug_on_left_plate (FM 0.90)
#   t73 STUDY_SCENE1_book_in_front_compartment       (FM 0.60)
#   t75 STUDY_SCENE1_book_in_right_compartment       (FM 0.50)
#   t81 STUDY_SCENE3_book_in_front_compartment       (FM 0.90)
#
# Budget (fits embers 8h):
#   n_iters=6, eval_interval=3 -> evals at 0/3/6 + final = 4
#   group_size=8, inits_per_iter=2 -> 128 collect rolls/iter
#   ppo_epochs=4, lr=1.0e-6, target_kl=0.05
#   eval_rollouts_per_env=15 -> 120 eval rolls/eval
# Total: 768 collect + 480 eval = ~1250 rolls (~3-4h on A100).
# Storage: /storage/scratch1/8/lwang831 only.
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

EXP_NAME_BC="cold_multitask_lib90"
RL_NAME="rl_hard8_fm_vanilla"
CKPT="${SCR_ROOT}/imitation/cold_start/libero/libero_90/${EXP_NAME_BC}/multitask_model_latest.pth"
LOG="${SCR_ROOT}/${RL_NAME}.log"
RES="${SCR_ROOT}/${RL_NAME}_results.txt"
mkdir -p "${SCR_ROOT}/imitation/logs"

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: multi-task BC ckpt not found at ${CKPT}" | tee -a "${RES}"
  exit 1
fi

N_ITERS=6
EVAL_EVERY=3
EXPECTED=4  # evals at 0, 3, 6 + final post-loop

if [ -f "${LOG}" ]; then
  NDONE=$(grep -c "deterministic eval success rate" "${LOG}" || true)
  if [ "${NDONE}" -ge "${EXPECTED}" ]; then
    EVALS=$(grep "deterministic eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
    echo "[$(date)] CACHED ${RL_NAME} evals=[${EVALS}]" | tee -a "${RES}"
    exit 0
  fi
fi

echo "[$(date)] START ${RL_NAME} ckpt=${CKPT}" | tee -a "${RES}"

python rl_train.py \
  cold_start_checkpoint="${CKPT}" \
  'task_indices=[8,21,32,53,65,73,75,81]' \
  rl.n_iters=${N_ITERS} \
  rl.group_size=8 \
  rl.inits_per_iter=2 \
  rl.ppo_epochs=4 \
  rl.lr=1.0e-6 \
  rl.target_kl=0.05 \
  rl.filter_accuracy=false \
  rl.eval_interval=${EVAL_EVERY} \
  rl.eval_rollouts_per_env=15 \
  logging.mode=disabled \
  exp_name=${RL_NAME} \
  hydra.run.dir="${SCR_ROOT}/imitation/hydra/${RL_NAME}_${SLURM_JOB_ID}" \
  > "${LOG}" 2>&1 || true

EVALS=$(grep "deterministic eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
echo "[$(date)] DONE ${RL_NAME} evals=[${EVALS}]" | tee -a "${RES}"
LAST_PT=$(grep "per-task:" "${LOG}" | tail -1)
echo "final per-task: ${LAST_PT}" | tee -a "${RES}"

echo "Finished at $(date)"
