#!/usr/bin/env bash
# Local LIBERO-90 deterministic eval of libero_bc_r2 (vanilla baseline).
# Runs per-task: rl_train.py n_iters=0 → evaluate(0) twice (= 10 rolls/task by default,
# we set eval_rollouts_per_env=5 to halve that → 5 rolls × 2 evaluate-calls = 10 rolls).
# Pure eval, no PPO updates.

set -o pipefail   # no -u — cluster /etc/bashrc trips on BASHRCSOURCED

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/imitation"
SCR_ROOT="/storage/scratch1/8/lwang831"
CKPT="${SCR_ROOT}/dialga_outputs/imitation/libero/libero_90/libero_bc_r2/multitask_model_latest.pth"
RES="${SCR_ROOT}/eval_libero_bc_r2/per_task.txt"
LOGDIR="${SCR_ROOT}/eval_libero_bc_r2/logs"

TASKS="${TASKS:-$(seq 0 89)}"      # subset: TASKS="0 1 2"
ROLLS="${ROLLS:-5}"                # rollouts per env

mkdir -p "${LOGDIR}" "$(dirname ${RES})"
cd "${WORKDIR}" || exit 1

source ~/.bashrc
conda activate porygon

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export MUJOCO_GL=egl
export EGL_DEVICE_ID=0

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: r2 ckpt missing at ${CKPT}" >&2
    exit 1
fi

echo "[$(date)] eval_libero_bc_r2 ckpt=${CKPT}" | tee -a "${RES}"

for T in ${TASKS}; do
    LOG="${LOGDIR}/task_${T}.log"
    if [[ -f "${LOG}" ]] && grep -q "deterministic eval success rate" "${LOG}"; then
        EVALS=$(grep "deterministic eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
        echo "[$(date)] CACHED task=${T} evals=[${EVALS}]" | tee -a "${RES}"
        continue
    fi

    python eval_libero_bc.py \
        algo=fm_policy_r2 \
        cold_start_checkpoint="${CKPT}" \
        task_index=${T} \
        rl.eval_rollouts_per_env=${ROLLS} \
        logging.mode=disabled \
        exp_name="eval_libero_bc_r2_t${T}" \
        hydra.run.dir="${SCR_ROOT}/eval_libero_bc_r2/hydra/t${T}" \
        > "${LOG}" 2>&1 || true

    EVALS=$(grep "deterministic eval success rate" "${LOG}" | grep -oE "[0-9.]+$" | tr '\n' ' ')
    echo "[$(date)] DONE task=${T} evals=[${EVALS}]" | tee -a "${RES}"
done

echo "[$(date)] EVAL DONE" | tee -a "${RES}"
