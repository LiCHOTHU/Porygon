#!/usr/bin/env bash
# monitor_libero_bc.sh — keep the 5 LIBERO BC experiments (r1..r5) alive.
#
# Loop: every SLEEP_INTERVAL_MIN minutes, for each run in $RUNS
#   • If it is currently in the queue (PENDING / RUNNING / REQUEUED), skip.
#   • Else, if any .out log for that run already shows the final epoch line
#     "Epoch  <TARGET_EPOCH> |", it is done — skip.
#   • Else, resubmit scripts/sbatch_libero_bc_5runs.sbatch with
#       -J libero_bc_${run} --export=ALL,RUN=${run}
#     Auto-resume in train.py (training.resume=true in the sbatch) picks up
#     from <out_dir>/multitask_model_latest.pth.
#
# Job-name convention: each run is submitted as libero_bc_${run} so that
#   squeue --name=libero_bc_${run} uniquely identifies it.
#
# Env overrides (defaults shown):
#   TARGET_EPOCH=30                  # must match N_EPOCHS in the sbatch
#   SLEEP_INTERVAL_MIN=15
#   SLEEP_BETWEEN_SUBMITS=2
#   RUNS="r1 r2 r3 r4 r5"            # subset = e.g. "r1 r3"
#   DRY_RUN=                         # if non-empty, print sbatch cmd instead
#                                    # of running it (for smoke testing)
#   ONESHOT=                         # if non-empty, do one pass and exit
#
# Usage:
#   bash scripts/monitor_libero_bc.sh
#   RUNS="r1 r3" bash scripts/monitor_libero_bc.sh
#   DRY_RUN=1 ONESHOT=1 bash scripts/monitor_libero_bc.sh   # smoke test
#   nohup bash scripts/monitor_libero_bc.sh > /tmp/monitor_libero_bc.log 2>&1 &
#
# Stop with Ctrl-C (or kill the nohup pid). Idempotent — re-running is harmless.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SBATCH_PATH="${SCRIPT_DIR}/sbatch_libero_bc_5runs.sbatch"
LOG_DIR="/storage/scratch1/8/lwang831/dialga_outputs/imitation/logs"

TARGET_EPOCH="${TARGET_EPOCH:-30}"
SLEEP_INTERVAL_MIN="${SLEEP_INTERVAL_MIN:-15}"
SLEEP_BETWEEN_SUBMITS="${SLEEP_BETWEEN_SUBMITS:-2}"
RUNS="${RUNS:-r1 r2 r3 r4 r5}"
DRY_RUN="${DRY_RUN:-}"
ONESHOT="${ONESHOT:-}"

if [[ ! -f "${SBATCH_PATH}" ]]; then
    echo "[error] sbatch script missing: ${SBATCH_PATH}" >&2
    exit 1
fi

echo "▶ Monitoring LIBERO BC jobs (libero_bc_* prefix)"
echo "   user:         ${USER}"
echo "   runs:         ${RUNS}"
echo "   target_epoch: ${TARGET_EPOCH}"
echo "   interval:     ${SLEEP_INTERVAL_MIN}m"
echo "   sbatch:       ${SBATCH_PATH}"
echo "   log_dir:      ${LOG_DIR}"
echo

# Returns 0 if any .out log for libero_bc_${run} contains the final
# "Epoch  <TARGET_EPOCH> |" line emitted by the trainer.
is_done() {
    local run="$1"
    # Tolerate variable whitespace in "Epoch%3d |" formatting.
    local pattern="Epoch[[:space:]]+${TARGET_EPOCH}[[:space:]]\|"
    shopt -s nullglob
    local logs=(${LOG_DIR}/libero_bc_${run}_*.out)
    shopt -u nullglob
    if (( ${#logs[@]} == 0 )); then
        return 1
    fi
    grep -El "${pattern}" "${logs[@]}" > /dev/null 2>&1
}

# Returns the highest "Epoch  N |" number seen across all .out logs for a run,
# or "?" if none.
latest_epoch() {
    local run="$1"
    shopt -s nullglob
    local logs=(${LOG_DIR}/libero_bc_${run}_*.out)
    shopt -u nullglob
    if (( ${#logs[@]} == 0 )); then
        echo "?"
        return
    fi
    local n
    n=$(grep -hE "Epoch[[:space:]]+[0-9]+[[:space:]]\|" "${logs[@]}" 2>/dev/null \
        | sed -E 's/.*Epoch[[:space:]]+([0-9]+)[[:space:]]\|.*/\1/' \
        | sort -n | tail -1)
    echo "${n:-?}"
}

submit_one() {
    local run="$1"
    local job_name="libero_bc_${run}"
    if [[ -n "${DRY_RUN}" ]]; then
        echo "[DRY]  would submit: sbatch -J ${job_name} --export=ALL,RUN=${run} ${SBATCH_PATH}"
        return
    fi
    sbatch -J "${job_name}" --export=ALL,RUN="${run}" "${SBATCH_PATH}" \
        || echo "[ERR]  sbatch returned non-zero for ${job_name}"
}

while true; do
    now="$(date '+%Y-%m-%d %H:%M:%S')"
    active="$(squeue -u "${USER}" -h -o "%j" 2>/dev/null || true)"

    echo "============== ${now} =============="
    for run in ${RUNS}; do
        job_name="libero_bc_${run}"
        ep="$(latest_epoch "${run}")"

        # 1. Already in queue (PENDING / RUNNING / REQUEUED / CONFIGURING / ...)?
        if printf '%s\n' "${active}" | awk -v j="${job_name}" '$0==j {f=1} END {exit !f}'; then
            echo "[OK]   ${job_name} active in queue (last logged epoch=${ep})"
            continue
        fi

        # 2. Reached target epoch in some prior slot?
        if is_done "${run}"; then
            echo "[DONE] ${job_name} reached epoch ${TARGET_EPOCH}"
            continue
        fi

        # 3. Otherwise, resubmit. Auto-resume continues from the latest .pth.
        echo "[MISS] ${job_name} not in queue (last logged epoch=${ep}) — submitting"
        submit_one "${run}"

        if (( SLEEP_BETWEEN_SUBMITS > 0 )); then
            sleep "${SLEEP_BETWEEN_SUBMITS}"
        fi
    done

    if [[ -n "${ONESHOT}" ]]; then
        echo "ONESHOT set — exiting after one pass."
        exit 0
    fi

    echo "Sleeping ${SLEEP_INTERVAL_MIN} minutes..."
    sleep "${SLEEP_INTERVAL_MIN}m"
done
