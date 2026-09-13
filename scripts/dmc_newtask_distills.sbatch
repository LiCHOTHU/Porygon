#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100,gpu-v100,gpu-l40s
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=32G
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.out
#SBATCH --error=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.err
# All three replacement-task distills in one job (they are small; one GPU
# holds all three in parallel). Login-node attempts died twice on session
# teardown; a batch job survives.
set -x
source ~/.bashrc
conda activate /storage/project/r-agarg35-0/lwang831/conda/envs/dice-rl
cd /storage/home/hcoda1/8/lwang831/workspace/dice_rl_official/dice-rl || exit 1
source ../set_path_noninteractive.sh
export MUJOCO_GL=egl EGL_DEVICE_ID=0
export NUMBA_CACHE_DIR=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/.numba_cache
L=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/dice_rl_official/log_dir
for t in finger-spin ball_in_cup-catch reacher-hard; do
  python script/run.py --config-name=pre_drifting_mlp --config-dir=cfg/dmc/pretrain/$t \
    wandb=null seed=42 logdir=$L/dmc-pretrain/$t ++train.auto_resume=true &
done
wait
echo "NEWTASK_DISTILLS_DONE"
