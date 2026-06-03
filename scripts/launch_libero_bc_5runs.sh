#!/usr/bin/env bash
# Launch the 5 LIBERO BC experiments with the same FlowMatchingPolicy + DiT,
# differing only in encoder flags. All use frame_stack=33 to satisfy the
# DIALGA encoder's pretrain chunk size.
#
# Usage:
#   bash scripts/launch_libero_bc_5runs.sh            # all 5 in series
#   bash scripts/launch_libero_bc_5runs.sh r1 r3      # just runs 1 and 3
#
# Environment expectations:
#   - Run from imitation/ root
#   - PYTHONPATH should NOT need editing — the wan_dialga encoder sys.paths
#     in the Dialga repo at /storage/home/hcoda1/8/lwang831/workspace/Dialga
#   - WAN_HF_HOME / HF_HUB_CACHE set to /storage/project/r-agarg35-0/lwang831/hf_cache

set -eo pipefail

RUNS=(r1 r2 r3 r4 r5)
if [ $# -gt 0 ]; then RUNS=("$@"); fi

EXP_NAME="${EXP_NAME:-libero_bc_5runs_$(date +%Y%m%d)}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/storage/scratch1/8/lwang831/dialga_outputs/imitation}"
DATA_PREFIX="${DATA_PREFIX:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LIBERO-datasets}"

export HF_HOME="${HF_HOME:-/storage/project/r-agarg35-0/lwang831/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TMPDIR="${TMPDIR:-/storage/project/r-agarg35-0/lwang831/tmp}"

for run in "${RUNS[@]}"; do
    echo "================================================================"
    echo "Launching ${run}"
    echo "================================================================"
    python train.py \
        algo=fm_policy_${run} \
        task=libero \
        task.benchmark_name=libero_90 \
        exp_name="${EXP_NAME}" \
        output_prefix="${OUTPUT_PREFIX}" \
        data_prefix="${DATA_PREFIX}" \
        seed=10000
done

echo "All requested runs done."
