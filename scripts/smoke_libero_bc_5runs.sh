#!/usr/bin/env bash
# Smoke-test all 5 configs end-to-end.  Tiny config: 1 task, 2 demos, 1 epoch,
# 20 batches, no rollout, no wandb.  Confirms encoder + policy + data pipeline
# integrate and BC loss is finite + dropping.
#
# Usage:
#   bash scripts/smoke_libero_bc_5runs.sh                 # all 5 sequentially
#   bash scripts/smoke_libero_bc_5runs.sh r1              # just r1
#   bash scripts/smoke_libero_bc_5runs.sh r1 r3 r4        # subset

set -eo pipefail
RUNS=(r1 r2 r3 r4 r5)
if [ $# -gt 0 ]; then RUNS=("$@"); fi

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/imitation"
SCR_ROOT="/storage/scratch1/8/lwang831/dialga_outputs/imitation_smoke"
PROJ_TMP="/storage/project/r-agarg35-0/lwang831/tmp"

cd "${WORKDIR}"
mkdir -p "${SCR_ROOT}/logs" "${PROJ_TMP}"

export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TMPDIR="${PROJ_TMP}"
export MPLCONFIGDIR="${PROJ_TMP}/matplotlib"
export TORCH_HOME="${PROJ_TMP}/torch_hub"
export XDG_CACHE_HOME="${PROJ_TMP}/xdg_cache"
export MUJOCO_GL=egl
mkdir -p "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${HF_HOME}"

PY="/storage/project/r-agarg35-0/lwang831/conda/envs/river/bin/python"

for run in "${RUNS[@]}"; do
    LOG="${SCR_ROOT}/logs/smoke_${run}.log"
    echo "===== smoking ${run}  -> ${LOG} ====="
    "${PY}" -u train.py --config-name=train \
        task=libero \
        task.benchmark_name=libero_90 \
        task.task_subset=[0] \
        task.demos_per_env=2 \
        algo=fm_policy_${run} \
        algo.batch_size=4 \
        training.n_epochs=1 \
        training.cut=15 \
        training.use_tqdm=false \
        training.log_interval=5 \
        training.save_interval=999 \
        training.resume=false \
        rollout.enabled=false \
        logging.mode=disabled \
        make_unique_experiment_dir=true \
        exp_name="smoke_${run}" \
        variant_name=smoke \
        data_prefix="${WORKDIR}/data" \
        output_prefix="${SCR_ROOT}" \
        hydra.run.dir="${SCR_ROOT}/hydra/${run}_$(date +%s)" \
        > "${LOG}" 2>&1 && echo "  [ok] ${run}" \
        || { echo "  [FAIL] ${run}  --  see ${LOG}"; tail -25 "${LOG}"; exit 1; }
done

echo "All smoke tests passed."
