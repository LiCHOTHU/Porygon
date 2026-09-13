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
# Experiment 1 arms: hinge (DICE + dead-zone hinge penalty) or proj (hard
# projection, no restoring field). Overrides are copied verbatim from the
# existing seed-10000 runs so new seeds are protocol-identical.
# usage: sbatch exp1_arm.sbatch <hinge|proj> <task> <seed>
set -x
ARM="${1:?hinge|proj}"; TASK="${2:?task}"; SEED="${3:?seed}"
source ~/.bashrc; conda activate porygon
export MUJOCO_GL=egl EGL_DEVICE_ID=0
export XDG_CACHE_HOME=/storage/scratch1/8/lwang831/.cache
export NUMBA_CACHE_DIR=$XDG_CACHE_HOME/numba
cd /storage/home/hcoda1/8/lwang831/workspace/imitation || exit 1
CEDAR=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch
CK=/storage/scratch1/8/lwang831/imitation/cold_start/libero/libero_90/drift_multitask_lib90/multitask_model_latest.pth
COMMON="algo.num_inference_steps=1 cold_start_checkpoint=$CK seed=${SEED} task_indices=[${TASK}]
  dice.bc_loss_weight=100 dice.n_iters=40 dice.episodes_per_iter=16 dice.warmup_episodes=32
  dice.gradient_steps=400 dice.eval_interval=10 dice.save_interval=5 dice.replay_size=100000
  dice.use_rlpd=true dice.disable_q_loss_for_expert_data=true dice.always_retain_bc_loss_for_expert_data=true
  dice.replay_flow_warmup_steps=300 dice.use_soft_q_filtering=false dice.q_filtering_warmup_steps=1000
  dice.q_depends_on_noise=false dice.eval_strategy=max_q_min dice.eval_num_samples=10
  dice.online_explore_strategy=max_q_min dice.num_exploration_samples=16 dice.num_multi_z=16
  dice.use_q_normalization=true logging.mode=disabled
  output_prefix=${CEDAR}/imitation/experiments_dice"
case "$ARM" in
  hinge) NAME="field_st_A_t${TASK}_hinge_s${SEED}"
    EXTRA="dice.actor_mode=residual +dice.bc_hinge_rho=0.05" ;;
  proj)  NAME="field_st_BNONE_t${TASK}_proj_s${SEED}"
    EXTRA="dice.actor_mode=field_pointwise dice.field.q_source=grad dice.field.q_step_size=1.0
      dice.field.bc_step_size=0.0 dice.field.total_max_norm=1e9 dice.field.q_max_norm=1e9
      +dice.field.restore_step_size=0.0 +dice.field.restore_radius=0.05
      ++dice.field.restore_radius_rms=true ++dice.use_n_step=true ++dice.n_step=3
      +dice.project_radius=0.05" ;;
  *) echo "unknown ARM"; exit 1 ;;
esac
RESUME=""
LATEST=${CEDAR}/imitation/experiments_dice/libero/libero_90/${NAME}/dice_latest.pth
[ -f "$LATEST" ] && RESUME="+dice_resume_checkpoint=$LATEST"
python dice_train.py $COMMON $EXTRA $RESUME exp_name=${NAME} \
  hydra.run.dir="${CEDAR}/imitation/hydra/${NAME}_${SLURM_JOB_ID}"
echo "DONE ${NAME}"
