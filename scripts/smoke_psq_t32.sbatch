#!/usr/bin/env bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1
#SBATCH -p gpu-a100,gpu-v100 --time=03:00:00 --mem-per-gpu=48G
#SBATCH --job-name=smoke_psq_t32 --qos=inferno --account=gts-agarg35
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%j.out
#SBATCH --error=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%j.err
# SMOKE: single-task Porygon (task 32, drift base), per_state_q_scale ON vs OFF.
# In a single-task batch per-state scale ~= global scale, so the two short runs
# should track closely; validates the fix end-to-end in the real training loop.
source ~/.bashrc
conda activate porygon
cd /storage/home/hcoda1/8/lwang831/workspace/imitation
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MUJOCO_GL=egl EGL_DEVICE_ID=0
CEDAR=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch
CKPT=/storage/scratch1/8/lwang831/imitation/cold_start/libero/libero_90/drift_multitask_lib90/multitask_model_latest.pth
for PSQ in true false; do
  NAME=smoke_psq_t32_${PSQ}
  echo "[$(date)] ==== RUN $NAME ===="
  python dice_train.py \
    algo.num_inference_steps=1 cold_start_checkpoint="${CKPT}" seed=10000 \
    'task_indices=[32]' \
    dice.bc_loss_weight=100 dice.n_iters=6 dice.episodes_per_iter=8 \
    dice.warmup_episodes=16 dice.gradient_steps=200 dice.eval_interval=3 \
    dice.save_interval=100 dice.replay_size=100000 dice.use_rlpd=true \
    dice.disable_q_loss_for_expert_data=true dice.always_retain_bc_loss_for_expert_data=true \
    dice.replay_flow_warmup_steps=150 dice.use_soft_q_filtering=false \
    dice.q_filtering_warmup_steps=400 dice.q_depends_on_noise=false \
    dice.eval_strategy=max_q_min dice.eval_num_samples=10 \
    dice.online_explore_strategy=max_q_min dice.num_exploration_samples=16 \
    dice.num_multi_z=16 dice.use_q_normalization=true \
    dice.actor_mode=field_pointwise dice.field.q_source=grad \
    dice.field.q_step_size=0.5 dice.field.bc_step_size=0.05 dice.field.total_max_norm=0.15 \
    +dice.field.restore_step_size=1.0 +dice.field.restore_radius=0.05 \
    +dice.per_state_q_scale=${PSQ} \
    logging.mode=disabled output_prefix=${CEDAR}/imitation/experiments_dice \
    exp_name=${NAME} hydra.run.dir="${CEDAR}/imitation/hydra/${NAME}_${SLURM_JOB_ID}" \
    2>&1 | grep -E "\[iter|eval success|actor=|residual" | head -40
done
echo "[$(date)] SMOKE DONE"
