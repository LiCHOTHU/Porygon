#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100,gpu-v100
#SBATCH --time=00:45:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=rl_smoke_drift_simplevla
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err

# Tiny end-to-end pipeline verification of rl_train.py + DRIFT cold-start (K=1).
CONDA_ENV="porygon"
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/imitation"
SCR_ROOT="/storage/scratch1/8/lwang831"
cd "${WORKDIR}" || exit 1

source ~/.bashrc
conda activate "${CONDA_ENV}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export MUJOCO_GL=egl EGL_DEVICE_ID=0

EXP_NAME_BC="drift_multitask_lib90"
RL_NAME="rl_smoke_drift_simplevla"
CKPT="${SCR_ROOT}/imitation/cold_start/libero/libero_90/${EXP_NAME_BC}/multitask_model_latest.pth"
LOG="${SCR_ROOT}/${RL_NAME}.log"
RES="${SCR_ROOT}/${RL_NAME}_results.txt"
mkdir -p "${SCR_ROOT}/imitation/logs"

[ -f "${CKPT}" ] || { echo "missing ckpt: ${CKPT}" | tee -a "${RES}"; exit 1; }

echo "[$(date)] START ${RL_NAME}" | tee -a "${RES}"

python rl_train.py \
  algo.num_inference_steps=1 \
  cold_start_checkpoint="${CKPT}" \
  'task_indices=[8]' \
  rl.n_iters=1 \
  rl.group_size=4 \
  rl.inits_per_iter=1 \
  rl.ppo_epochs=1 \
  rl.minibatch_size=8 \
  rl.eval_interval=1 \
  rl.eval_rollouts_per_env=2 \
  rl.filter_accuracy=true \
  logging.mode=disabled \
  exp_name=${RL_NAME} \
  hydra.run.dir="${SCR_ROOT}/imitation/hydra/${RL_NAME}_${SLURM_JOB_ID}" \
  > "${LOG}" 2>&1

RC=$?
echo "[$(date)] DONE rc=${RC}" | tee -a "${RES}"
tail -30 "${LOG}" | tee -a "${RES}"
