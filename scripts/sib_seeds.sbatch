#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s,gpu-a100,gpu-h100
#SBATCH --time=06:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=sib_seeds
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --requeue
#SBATCH --array=0-8
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%A_%a.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%A_%a.err

# Seed replication.  Seed 10000 gave sib 0.560 / sam 0.440 / vanilla 0.340 under
# cam_shift -- sib over vanilla was p=0.077 and sib over sam p=0.581 on a paired
# sign test across the 20 tasks, i.e. the ordering is suggestive and nothing is
# established.  One seed cannot separate method from seed noise.
#
# Three arms x three new seeds.  mixup is excluded deliberately: it scored 0.370
# clean against every other arm's 0.85-0.97, which is an untuned alpha rather
# than a fair baseline, and replicating a broken baseline three times buys
# nothing.  It needs an alpha sweep before it can be reported at all.
#
# vanilla and sam are both kept because sam, not vanilla, is the comparison the
# novelty claim rests on -- beating vanilla is table stakes.

set -o pipefail
ARMS=(fm_sib_S fm_policy_S fm_sam_S)
NAMES=(sib vanilla sam)
SEEDS=(10001 10002 10003)
A=$((SLURM_ARRAY_TASK_ID % 3))
S=$((SLURM_ARRAY_TASK_ID / 3))
ARM=${ARMS[$A]}; NAME=${NAMES[$A]}; SEED=${SEEDS[$S]}

cd /storage/home/hcoda1/8/lwang831/workspace/imitation || exit 1
source ~/.bashrc
conda activate porygon
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MUJOCO_GL=egl
export EGL_DEVICE_ID=0
export HF_HOME=/storage/project/r-agarg35-0/lwang831/tmp/hf
export TRANSFORMERS_CACHE=$HF_HOME
export MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/mpl
export TORCH_HOME=/storage/project/r-agarg35-0/lwang831/tmp/torch

# Preflight: atl1-1-03-007-29-0 has a GPU throwing uncorrectable ECC errors, and
# it killed six of nine jobs in this array in ~20s each.  Because it fails fast it
# frees its slot immediately and Slurm hands it the next job, so an array can be
# eaten one element at a time.  --exclude cannot be used: the site submit filter
# overwrites ExcNodeList with its own.  So probe the GPU and bounce off a sick
# node instead of training on it.
if ! python -c "import torch; torch.zeros(64,64,device='cuda').matmul(torch.zeros(64,64,device='cuda')); torch.cuda.synchronize()" 2>/dev/null; then
    echo "!!! GPU unhealthy on $(hostname) -- requeueing instead of failing"
    scontrol requeue "${SLURM_JOB_ID}"
    sleep 60
    exit 1
fi

echo "=== ${NAME} seed ${SEED} on $(hostname) at $(date -Is) ==="
python train.py --config-name=train task=libero algo="${ARM}" \
    output_prefix=/storage/scratch1/8/lwang831/imitation/experiments \
    seed="${SEED}" \
    'task.task_subset=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]' \
    task.demos_per_env=50 \
    training.n_epochs=15 \
    rollout.enabled=true rollout.interval=15 rollout.rollouts_per_env=5 \
    logging.mode=disabled \
    exp_name="sib_seed${SEED}_${NAME}"
# Propagate python's status.  Without this the trailing echo becomes the script's
# exit code, so a crashed run reports COMPLETED to sacct -- which is how six jobs
# killed by a bad GPU looked like successes until their logs were read.
rc=$?
echo "=== done rc=${rc} $(date -Is) ==="
exit $rc
