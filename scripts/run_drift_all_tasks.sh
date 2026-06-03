#!/usr/bin/env bash
# Sequentially train PolicyDrifting (lambertae port) on each LIBERO-90 task
# in turn. One run per task = one checkpoint dir under:
#   /storage/scratch1/8/lwang831/imitation/local/libero/libero_90/drift_single_t<T>_lambertae/
# Skips any task whose 50-epoch checkpoint already exists (idempotent).
set -e
cd /storage/home/hcoda1/8/lwang831/workspace/imitation
source ~/.bashrc
conda activate porygon
export MUJOCO_GL=egl
export EGL_DEVICE_ID=0
SCR=/storage/scratch1/8/lwang831

EPOCHS=50
DEMOS=50

mkdir -p ${SCR}/imitation/local ${SCR}/imitation/hydra ${SCR}/imitation/logs
SUMMARY=${SCR}/imitation/logs/drift_all_tasks_summary.txt
echo "[$(date)] starting all-tasks sweep | EPOCHS=${EPOCHS} DEMOS=${DEMOS}" | tee -a "${SUMMARY}"

for T in $(seq 0 89); do
    EXP=drift_single_t${T}_lambertae
    OUT=${SCR}/imitation/local/libero/libero_90/${EXP}
    FINAL=${OUT}/multitask_model_epoch_$(printf '%04d' ${EPOCHS}).pth
    LOG=${SCR}/imitation/logs/${EXP}.out

    if [ -f "${FINAL}" ]; then
        FINAL_MSE=$(grep "Epoch  ${EPOCHS}" "${OUT}/training.log" 2>/dev/null | tail -1 | grep -oE 'action MSE: [0-9.]+' | awk '{print $3}')
        echo "[$(date)] SKIP task=${T} (epoch ${EPOCHS} ckpt exists; final MSE=${FINAL_MSE})" | tee -a "${SUMMARY}"
        continue
    fi

    echo "[$(date)] START task=${T}" | tee -a "${SUMMARY}"
    python train.py --config-name=train \
        task=libero algo=policy_drifting_S \
        task.task_subset=[${T}] task.demos_per_env=${DEMOS} \
        algo.chunk_size=16 algo.action_horizon=8 \
        training.n_epochs=${EPOCHS} \
        training.save_interval=5 \
        rollout.enabled=false \
        logging.mode=disabled \
        exp_name=${EXP} \
        data_prefix=${SCR}/imitation/data \
        output_prefix=${SCR}/imitation/local \
        hydra.run.dir=${SCR}/imitation/hydra/${EXP}_$(date +%s) \
        > "${LOG}" 2>&1

    FINAL_MSE=$(grep "Epoch  ${EPOCHS}" "${OUT}/training.log" 2>/dev/null | tail -1 | grep -oE 'action MSE: [0-9.]+' | awk '{print $3}')
    echo "[$(date)] DONE  task=${T} final action MSE=${FINAL_MSE}" | tee -a "${SUMMARY}"
done

echo "[$(date)] all tasks complete." | tee -a "${SUMMARY}"
