#!/usr/bin/env bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1
#SBATCH -p gpu-v100,gpu-a100 --time=08:00:00 --mem-per-gpu=48G
#SBATCH --job-name=grpo_st --qos=embers --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.out
#SBATCH --error=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.err
# Matched-budget single-task GRPO (SimpleVLA-style), drift or FM base.
#   sbatch scripts/grpo_single_task.sbatch <drift|fm> <TASK> <SEED>
# Budget matches the Porygon protocol: 40 iters x (group 8 x 2 inits) = 640 episodes.
BASE="${1:?drift|fm}"; TASK="${2:?task}"; SEED="${3:-10000}"
source ~/.bashrc; conda activate porygon
cd /storage/home/hcoda1/8/lwang831/workspace/imitation
export OMP_NUM_THREADS=8 MUJOCO_GL=egl EGL_DEVICE_ID=0
CEDAR=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch
SCR=/storage/scratch1/8/lwang831
case "$BASE" in
  drift) CKPT=$SCR/imitation/cold_start/libero/libero_90/drift_multitask_lib90/multitask_model_latest.pth; NIS=1 ;;
  fm)    CKPT=$SCR/imitation/cold_start/libero/libero_90/cold_multitask_lib90/multitask_model_latest.pth;  NIS=10 ;;
  *) echo "unknown base"; exit 1 ;;
esac
NAME=grpo_st_${BASE}_t${TASK}_s${SEED}
python rl_train.py \
  algo.num_inference_steps=${NIS} cold_start_checkpoint="${CKPT}" \
  task_indices=[${TASK}] seed=${SEED} \
  rl.n_iters=40 rl.group_size=8 rl.inits_per_iter=2 rl.ppo_epochs=4 \
  rl.lr=1.0e-6 rl.target_kl=0.05 rl.init_kl_coef=1.0e-3 rl.kl_horizon=1000 \
  rl.task_balanced_minibatches=true rl.filter_accuracy=true rl.filter_low=0.1 rl.filter_high=0.9 \
  rl.eval_interval=10 rl.eval_rollouts_per_env=20 \
  logging.mode=disabled output_prefix=${CEDAR}/imitation/experiments_rl \
  exp_name=${NAME} hydra.run.dir="${CEDAR}/imitation/hydra/${NAME}_${SLURM_JOB_ID}" \
  > ${CEDAR}/${NAME}.log 2>&1
echo "DONE ${NAME}"
