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
# base checkpoint + inference steps are overridable so the same protocol can be
# run on the drifting base (default, 1 step) or the flow-matching base (10 steps)
CKPT="${BASE_CKPT:-/storage/scratch1/8/lwang831/imitation/cold_start/libero/libero_90/drift_multitask_lib90/multitask_model_latest.pth}"
NIS="${NUM_INF_STEPS:-1}"
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
  # ---------------------------------------------------------------------------
  # H2: is an ACTION-SPACE CONSTRAINT necessary?  Clean 2x2 over
  #     (clip on/off) x (anchor on/off), everything else = the frozen B recipe.
  #
  # WHY THESE EXIST: the published rho-sweep does NOT isolate the two guards.
  # Every Porygon arm ships bc_step_size=0.05, and in field_pointwise mode
  # bc_field = -old_res (distill_rl.py:905) with unit_normalize/adaptive_bc both
  # false, so the update is
  #     total_delta = clip( q_step*V_Q + 0.05*(-r) + restore_delta , total_max_norm )
  # restore_radius->inf only zeroes restore_delta; the -0.05*r pull SURVIVES.
  # So "rho=inf" is clip + a weak always-on anchor, not clip-only. Removing the
  # anchor requires bc_step_size=0 AND restore_step_size=0, which is what B_CLIP
  # and B_NONE do below.
  #
  # "clip" = BOTH action-space displacement bounds (q_max_norm on V_Q itself and
  # total_max_norm on the summed displacement). Turning the clip off must turn
  # off both, else V_Q is still bounded at 1.0 and the corner is contaminated the
  # same way rho=inf was. 1e9 is a no-op in clip_field_norm (scale clamps to 1.0).
  #
  #   B      = clip ON,  anchor ON   (the frozen recipe; already above)
  #   B_CLIP = clip ON,  anchor OFF
  #   B_ANC  = clip OFF, anchor ON
  #   B_NONE = clip OFF, anchor OFF  -> the unconstrained action-gradient update,
  #                                     i.e. what the DIPO/QSM family runs
  B_CLIP) RL_NAME="field_st_BCLIP_t${TASK}${TAG}_s${SEED}"; ACTOR=(dice.actor_mode=field_pointwise dice.field.q_source=grad
       dice.field.q_step_size=1.0 dice.field.bc_step_size=0.0 dice.field.total_max_norm=0.15
       dice.field.q_max_norm=1.0
       +dice.field.restore_step_size=0.0 +dice.field.restore_radius=0.05
       ++dice.field.restore_radius_rms=true ++dice.use_n_step=true ++dice.n_step=3) ;;
  B_ANC)  RL_NAME="field_st_BANC_t${TASK}${TAG}_s${SEED}";  ACTOR=(dice.actor_mode=field_pointwise dice.field.q_source=grad
       dice.field.q_step_size=1.0 dice.field.bc_step_size=0.0 dice.field.total_max_norm=1e9
       dice.field.q_max_norm=1e9
       +dice.field.restore_step_size=1.0 +dice.field.restore_radius=0.05
       ++dice.field.restore_radius_rms=true ++dice.use_n_step=true ++dice.n_step=3) ;;
  B_NONE) RL_NAME="field_st_BNONE_t${TASK}${TAG}_s${SEED}"; ACTOR=(dice.actor_mode=field_pointwise dice.field.q_source=grad
       dice.field.q_step_size=1.0 dice.field.bc_step_size=0.0 dice.field.total_max_norm=1e9
       dice.field.q_max_norm=1e9
       +dice.field.restore_step_size=0.0 +dice.field.restore_radius=0.05
       ++dice.field.restore_radius_rms=true ++dice.use_n_step=true ++dice.n_step=3) ;;
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
  algo.num_inference_steps=${NIS} cold_start_checkpoint="${CKPT}" ${RESUME_FLAG} seed=${SEED} \
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
