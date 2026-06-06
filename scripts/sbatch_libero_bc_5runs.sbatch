#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=libero_bc
# NOTE: override per-run with `sbatch -J libero_bc_${RUN} ...` (see loop below).
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
# NOTE: `--account=gts-agarg35` + embers routes to V100-16GB regardless of -p,
# and 16 GB is too small for Wan-VAE 5B (~10 GB) + bs=16 × 33-frame activations.
# Use the l40s account to land on L40S-48GB.  See memory/project_vae_unfreeze_runbook.md.
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/dialga_outputs/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/dialga_outputs/imitation/logs/%x_%j.err

# ============================================================
# Run ONE of the 5 LIBERO BC experiments on full LIBERO-90.
# Driven by env var RUN ∈ {r1,r2,r3,r4,r5}.
#
# Submit with:
#   for r in r1 r2 r3 r4 r5; do
#       sbatch -J libero_bc_${r} --export=ALL,RUN=${r} scripts/sbatch_libero_bc_5runs.sbatch
#   done
#
# All 5 share: FlowMatchingPolicy + DiT, frame_stack=33, chunk=16,
# horizon=8, lr=1e-4, 30 epochs.  Only encoder flags differ.
# Auto-resumes on requeue (training.resume=true is default).
# ============================================================
set -eo pipefail
: "${RUN:?must export RUN=r1|r2|r3|r4|r5}"

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/imitation"
SCR_ROOT="/storage/scratch1/8/lwang831/dialga_outputs/imitation"
PROJ_TMP="/storage/project/r-agarg35-0/lwang831/tmp"

cd "${WORKDIR}" || exit 1
mkdir -p "${SCR_ROOT}/logs" "${PROJ_TMP}/matplotlib"

source ~/.bashrc
conda activate river

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
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

echo "[$(date)] JobID=${SLURM_JOB_ID}  Node=${SLURM_NODELIST}  RUN=${RUN}"
nvidia-smi || true

EXP_NAME="libero_bc_${RUN}"
N_EPOCHS=30

# Wan-VAE latent cache. r1/r3/r4/r5 use it; r2 doesn't touch Wan-VAE so we skip
# the cache to avoid wasted I/O.
#
# Two backends supported (transparent to the dataset patch — it picks .npy first
# if present in the dir, else .h5):
#   libero_bc_wan_cache/      .h5 chunked (h5py reads at ~6 ms/sample on scratch)
#   libero_bc_wan_cache_npy/  np.memmap  (~0.001 ms/sample once OS page cache is warm)
WAN_CACHE_DIR="${SCR_ROOT}/libero_bc_wan_cache_npy"
WAN_OVERRIDE=""
case "${RUN}" in
    r1|r3|r4|r5) WAN_OVERRIDE="algo.encoder.wan_cache_dir=${WAN_CACHE_DIR}" ;;
esac

python -u train.py --config-name=train \
    task=libero \
    task.benchmark_name=libero_90 \
    algo=fm_policy_${RUN} \
    algo.batch_size=16 \
    training.n_epochs=${N_EPOCHS} \
    training.save_interval=2 \
    training.log_interval=100 \
    training.resume=true \
    rollout.enabled=false \
    logging.mode=online \
    logging.group="${EXP_NAME}" \
    exp_name="${EXP_NAME}" \
    data_prefix="${WORKDIR}/data" \
    output_prefix="${SCR_ROOT}" \
    hydra.run.dir="${SCR_ROOT}/hydra/${EXP_NAME}" \
    ${WAN_OVERRIDE}

echo "[$(date)] done."
