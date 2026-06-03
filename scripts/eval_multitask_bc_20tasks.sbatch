#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100
#SBATCH --time=04:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=fm_eval_bc_20tasks
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err

# ============================================================
# BC eval of the multi-task LIBERO-90 checkpoint
# (cold_multitask_lib90/multitask_model_latest.pth) on the 20
# tasks selected for the multi-task RL run. 20 deterministic
# rollouts per task = 400 rolls. Per-task and overall success
# rate written to ${SCR_ROOT}/eval_bc_20tasks.txt.
# These 20 tasks are the bottom-20 by BC success on the full
# LIBERO-90 eval (sweet spot for RL headroom).
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
CKPT="${SCR_ROOT}/imitation/cold_start/libero/libero_90/${EXP_NAME_BC}/multitask_model_latest.pth"
RES="${SCR_ROOT}/eval_bc_20tasks.txt"
LOGDIR="${SCR_ROOT}/imitation/eval_logs/bc_20tasks"
mkdir -p "${LOGDIR}" "${SCR_ROOT}/imitation/logs"

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: cold-start ckpt not found at ${CKPT}" | tee -a "${RES}"
  exit 1
fi

# Bottom-20 by per-task BC success (from /storage/scratch1/8/lwang831/eval_multitask_lib90_per_task.txt).
TASKS=(6 7 8 9 13 14 21 23 32 36 43 45 51 52 53 69 73 75 78 82)

echo "[$(date)] eval start ckpt=${CKPT} job=${SLURM_JOB_ID} tasks=(${TASKS[*]})" | tee -a "${RES}"

for T in "${TASKS[@]}"; do
  LOG="${LOGDIR}/task_${T}.log"
  if [ -f "${LOG}" ] && grep -q "deterministic eval success rate" "${LOG}"; then
    EVALS=$(grep "deterministic eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
    echo "[$(date)] CACHED task=${T} evals=[${EVALS}]" | tee -a "${RES}"
    continue
  fi

  python rl_train.py \
    cold_start_checkpoint="${CKPT}" task_index=${T} \
    rl.n_iters=0 \
    rl.eval_rollouts_per_env=20 \
    logging.mode=disabled \
    exp_name="eval_bc_20tasks_t${T}" \
    hydra.run.dir="${SCR_ROOT}/imitation/hydra/eval_bc_20tasks_t${T}_${SLURM_JOB_ID}" \
    > "${LOG}" 2>&1 || true

  EVALS=$(grep "deterministic eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
  echo "[$(date)] DONE task=${T} evals=[${EVALS}]" | tee -a "${RES}"
done

echo "[$(date)] EVAL DONE job=${SLURM_JOB_ID}" | tee -a "${RES}"

# Compute overall mean (last eval per task is the final-iter eval; n_iters=0 so iter0==final).
export SCR_ROOT
python - <<'PY' | tee -a "${RES}"
import os, re
logdir = os.path.join(os.environ["SCR_ROOT"], "imitation/eval_logs/bc_20tasks")
vals = []
for fn in sorted(os.listdir(logdir)):
    p = os.path.join(logdir, fn)
    with open(p) as f:
        nums = [float(x) for x in re.findall(r"deterministic eval success rate: ([0-9.]+)", f.read())]
    if nums:
        vals.append((fn, nums[-1]))
print("---- per-task BC success ----")
for fn, v in vals:
    print(f"{fn}: {v:.3f}")
if vals:
    mean = sum(v for _, v in vals) / len(vals)
    print(f"---- mean BC success (20 tasks, 20 rolls each): {mean:.3f}")
PY

echo "Finished at $(date)"
