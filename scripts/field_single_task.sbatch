#!/usr/bin/env bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1
#SBATCH -p gpu-v100,gpu-a100 --time=08:00:00 --mem-per-gpu=48G
#SBATCH --job-name=field_st --qos=embers --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.out
#SBATCH --error=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.err
# SINGLE-TASK Porygon/backprop on one LIBERO hard-8 task from the drift multitask base.
#   sbatch scripts/field_single_task.sbatch <A|B> <TASK> <SEED>
EXP="${1:?A|B}"; TASK="${2:?task index}"; SEED="${3:-10000}"; TAG="${4:-}"; EXTRA=("${@:5}")
source ~/.bashrc
conda activate porygon
cd /storage/home/hcoda1/8/lwang831/workspace/imitation
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MUJOCO_GL=egl EGL_DEVICE_ID=0
CEDAR=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch
CKPT=/storage/scratch1/8/lwang831/imitation/cold_start/libero/libero_90/drift_multitask_lib90/multitask_model_latest.pth
case "${EXP}" in
  A) RL_NAME="field_st_A_t${TASK}${TAG}_s${SEED}"; ACTOR=(dice.actor_mode=residual) ;;
  # B = the COMMITTED Porygon LIBERO recipe (frozen 2026-07-30): q1 + rms dead-zone + 3-step returns
  B) RL_NAME="field_st_B_t${TASK}${TAG}_s${SEED}"; ACTOR=(dice.actor_mode=field_pointwise dice.field.q_source=grad
       dice.field.q_step_size=1.0 dice.field.bc_step_size=0.05 dice.field.total_max_norm=0.15
       +dice.field.restore_step_size=1.0 +dice.field.restore_radius=0.05
       ++dice.field.restore_radius_rms=true ++dice.use_n_step=true ++dice.n_step=3) ;;
  # B0 = legacy pre-freeze recipe (ablation reference)
  B0) RL_NAME="field_st_B0_t${TASK}${TAG}_s${SEED}"; ACTOR=(dice.actor_mode=field_pointwise dice.field.q_source=grad
       dice.field.q_step_size=0.5 dice.field.bc_step_size=0.05 dice.field.total_max_norm=0.15
       +dice.field.restore_step_size=1.0 +dice.field.restore_radius=0.05) ;;
  *) echo "unknown EXP=${EXP}"; exit 1 ;;
esac
RESUME_FLAG=""
for RC in "${CEDAR}/imitation/experiments_dice/libero/libero_90/${RL_NAME}/dice_latest.pth" \
          "${CEDAR}/imitation/experiments_dice/libero/libero_90/${RL_NAME}/${SEED}/dice_latest.pth"; do
  [ -f "$RC" ] && { RESUME_FLAG="+dice_resume_checkpoint=${RC}"; echo "RESUME $RC"; break; }
done
LOG="${CEDAR}/${RL_NAME}.log"
echo "[$(date)] START ${RL_NAME}"
python dice_train.py \
  algo.num_inference_steps=1 cold_start_checkpoint="${CKPT}" ${RESUME_FLAG} seed=${SEED} \
  "task_indices=[${TASK}]" \
  dice.bc_loss_weight=100 dice.n_iters=40 dice.episodes_per_iter=16 \
  dice.warmup_episodes=32 dice.gradient_steps=400 dice.eval_interval=10 \
  dice.save_interval=5 dice.replay_size=100000 dice.use_rlpd=true \
  dice.disable_q_loss_for_expert_data=true dice.always_retain_bc_loss_for_expert_data=true \
  dice.replay_flow_warmup_steps=300 dice.use_soft_q_filtering=false \
  dice.q_filtering_warmup_steps=1000 dice.q_depends_on_noise=false \
  dice.eval_strategy=max_q_min dice.eval_num_samples=10 \
  dice.online_explore_strategy=max_q_min dice.num_exploration_samples=16 \
  dice.num_multi_z=16 dice.use_q_normalization=true \
  "${ACTOR[@]}" \
  "${EXTRA[@]}" \
  logging.mode=disabled output_prefix=${CEDAR}/imitation/experiments_dice \
  exp_name=${RL_NAME} hydra.run.dir="${CEDAR}/imitation/hydra/${RL_NAME}_${SLURM_JOB_ID}" \
  >> "${LOG}" 2>&1
echo "[$(date)] DONE exit=$?"
