#!/usr/bin/env bash
# monitor_libero_bc.sh — keep the 5 LIBERO BC experiments (r1..r5) alive.
#
# Loop: every SLEEP_INTERVAL_MIN minutes, for each run in $RUNS
#   • If it is currently in the queue (under EITHER the new short prefix
#     "lbc_" OR the old "libero_bc_" prefix), skip.
#   • Else, if any .out log for that run (under either prefix) already shows
#     the final epoch line "Epoch  <TARGET_EPOCH> |", it is done — skip.
#   • Else, resubmit scripts/sbatch_libero_bc_5runs.sbatch with
#       -J lbc_${run} --export=ALL,RUN=${run}
#     Auto-resume in train.py (training.resume=true in the sbatch) picks up
#     from <out_dir>/multitask_model_latest.pth.
#
# Job-name convention: monitor SUBMITS as `lbc_${run}` (6 chars — fits the
# narrow squeue %j column), but RECOGNIZES both `lbc_${run}` and the older
# `libero_bc_${run}` as the same run, so it doesn't double-submit during
# the transition while pre-existing long-named jobs are still cycling.
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
#   nohup bash scripts/monitor_libero_bc.sh \
#       > /storage/project/r-agarg35-0/lwang831/tmp/monitor_libero_bc.log 2>&1 &
#
# Stop with Ctrl-C (or kill the nohup pid). Idempotent — re-running is harmless.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SBATCH_PATH="${SCRIPT_DIR}/sbatch_libero_bc_5runs.sbatch"
LOG_DIR="/storage/scratch1/8/lwang831/dialga_outputs/imitation/logs"

# Both prefixes recognized; only the new one is used for fresh submits.
JOB_PREFIX_NEW="lbc_"
JOB_PREFIX_OLD="libero_bc_"

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

echo "▶ Monitoring LIBERO BC jobs (submits as ${JOB_PREFIX_NEW}*, recognizes ${JOB_PREFIX_OLD}* too)"
echo "   user:         ${USER}"
echo "   runs:         ${RUNS}"
echo "   target_epoch: ${TARGET_EPOCH}"
echo "   interval:     ${SLEEP_INTERVAL_MIN}m"
echo "   sbatch:       ${SBATCH_PATH}"
echo "   log_dir:      ${LOG_DIR}"
echo

# Populate LOGS_FOR_RUN with every .out log matching either prefix.
collect_logs() {
    local run="$1"
    shopt -s nullglob
    LOGS_FOR_RUN=( \
        ${LOG_DIR}/${JOB_PREFIX_OLD}${run}_*.out \
        ${LOG_DIR}/${JOB_PREFIX_NEW}${run}_*.out \
    )
    shopt -u nullglob
}

# Returns 0 if either job-name variant is currently in $active.
is_in_queue() {
    local run="$1"
    local jn
    for jn in "${JOB_PREFIX_NEW}${run}" "${JOB_PREFIX_OLD}${run}"; do
        if printf '%s\n' "${active}" | awk -v j="${jn}" '$0==j {f=1} END {exit !f}'; then
            ACTIVE_NAME="${jn}"
            return 0
        fi
    done
    ACTIVE_NAME=""
    return 1
}

# Returns 0 if any log for ${run} contains "Epoch  <TARGET_EPOCH> |".
is_done() {
    local run="$1"
    local pattern="Epoch[[:space:]]+${TARGET_EPOCH}[[:space:]]\|"
    collect_logs "${run}"
    if (( ${#LOGS_FOR_RUN[@]} == 0 )); then
        return 1
    fi
    grep -El "${pattern}" "${LOGS_FOR_RUN[@]}" > /dev/null 2>&1
}

# Highest "Epoch  N |" across all logs for the run, or "?" if none.
latest_epoch() {
    local run="$1"
    collect_logs "${run}"
    if (( ${#LOGS_FOR_RUN[@]} == 0 )); then
        echo "?"
        return
    fi
    local n
    n=$(grep -hE "Epoch[[:space:]]+[0-9]+[[:space:]]\|" "${LOGS_FOR_RUN[@]}" 2>/dev/null \
        | sed -E 's/.*Epoch[[:space:]]+([0-9]+)[[:space:]]\|.*/\1/' \
        | sort -n | tail -1)
    echo "${n:-?}"
}

submit_one() {
    local run="$1"
    local job_name="${JOB_PREFIX_NEW}${run}"
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
        ep="$(latest_epoch "${run}")"

        # 1. Already in queue under either prefix?
        if is_in_queue "${run}"; then
            echo "[OK]   ${ACTIVE_NAME} active in queue (last logged epoch=${ep})"
            continue
        fi

        # 2. Reached target epoch in some prior slot?
        if is_done "${run}"; then
            echo "[DONE] ${JOB_PREFIX_NEW}${run} reached epoch ${TARGET_EPOCH}"
            continue
        fi

        # 3. Otherwise, resubmit. Auto-resume continues from the latest .pth.
        echo "[MISS] ${JOB_PREFIX_NEW}${run} not in queue (last logged epoch=${ep}) — submitting"
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
