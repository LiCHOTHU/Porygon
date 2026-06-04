#!/usr/bin/env bash
# Local r1 training smoke test on the .npy memmap cache.
# Runs on the local H100 with an ISOLATED output_prefix so it cannot touch
# the cluster's libero_bc_r1 checkpoint dir. Limited to 1 epoch.
# Kill manually after a few hundred batches once iter/sec stabilizes.

set -o pipefail   # no -u — cluster /etc/bashrc trips on BASHRCSOURCED

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/imitation"
ISO_ROOT="/storage/scratch1/8/lwang831/local_h5_test"
NPY_CACHE_DIR="/storage/scratch1/8/lwang831/dialga_outputs/imitation/libero_bc_wan_cache"

cd "${WORKDIR}" || exit 1
source ~/.bashrc
conda activate river

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export MUJOCO_GL=egl
export EGL_DEVICE_ID=0

# HF caches need to point to project storage (login-node defaults are read-only).
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export MPLCONFIGDIR="/storage/project/r-agarg35-0/lwang831/tmp/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

mkdir -p "${ISO_ROOT}/hydra"

# r1 = full DIALGA (z_dyn + z_static), most representative of the cache-using runs.
python -u train.py --config-name=train \
    task=libero \
    task.benchmark_name=libero_90 \
    algo=fm_policy_r1 \
    algo.batch_size=16 \
    algo.encoder.wan_cache_dir="${NPY_CACHE_DIR}" \
    training.n_epochs=1 \
    training.save_interval=999 \
    training.log_interval=50 \
    training.resume=false \
    rollout.enabled=false \
    logging.mode=disabled \
    exp_name="local_r1_h5_test" \
    data_prefix="${WORKDIR}/data" \
    output_prefix="${ISO_ROOT}" \
    hydra.run.dir="${ISO_ROOT}/hydra/local_r1_h5_test"
