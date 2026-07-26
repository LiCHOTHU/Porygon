#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=field_hard8_drift
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.out
#SBATCH --error=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.err

# ============================================================
# Two-field residual actor ("our method") vs baseline on the ONE-STEP DRIFT base,
# MULTITASK over hard-8. Inherits the established rl_hard8_drift_dice_bcw10 recipe
# EXACTLY (K=1, RLPD, n_iters=100 ep=32 warmup=64 grad=800 bcw=10); the ONLY diff
# between arms is the actor update, and seeds are matched across A/C:
#   A  actor_mode=residual                                reference drift-DICE
#   C  actor_mode=field_distributional q_source=zeroth    our method (drift-field + BC + RL)
#   usage:  sbatch scripts/field_hard8_drift.sbatch <A|C> <SEED>
# ============================================================
EXP="${1:?usage: sbatch field_hard8_drift.sbatch <A|C> <SEED> [hard|guarded]}"
SEED="${2:?usage: sbatch field_hard8_drift.sbatch <A|C> <SEED> [hard|guarded]}"
FILTER="${3:-hard}"
EXTRA=("${@:4}")
case "${FILTER}" in
  hard)    SOFTQ=false; FTAG="" ;;
  guarded) SOFTQ=true;  FTAG="_gf" ;;
  *) echo "unknown FILTER=${FILTER} (use hard or guarded)"; exit 1 ;;
esac
CONDA_ENV="porygon"; WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/imitation"
SCR="/storage/scratch1/8/lwang831"
CEDAR="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch"
cd "${WORKDIR}" || exit 1; source ~/.bashrc; conda activate "${CONDA_ENV}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MUJOCO_GL=egl EGL_DEVICE_ID=0 HYDRA_FULL_ERROR=1
nvidia-smi || true

CKPT="${SCR}/imitation/cold_start/libero/libero_90/drift_multitask_lib90/multitask_model_latest.pth"
if [ ! -f "${CKPT}" ]; then echo "ERROR: drift base ckpt missing at ${CKPT}"; exit 1; fi

case "${EXP}" in
  A) RL_NAME="field_hard8_ff_A_residual${FTAG}_s${SEED}";      ACTOR=(dice.actor_mode=residual) ;;
  B) RL_NAME="field_hard8_ff_B_grad${FTAG}_s${SEED}";          ACTOR=(dice.actor_mode=field_pointwise dice.field.q_source=grad
        dice.field.q_step_size=0.2 dice.field.bc_step_size=0.05 dice.field.total_max_norm=0.05) ;;
  C) RL_NAME="field_hard8_ff_C_zeroth${FTAG}_s${SEED}";        ACTOR=(dice.actor_mode=field_distributional dice.field.q_source=zeroth
        dice.field.q_step_size=0.2 dice.field.bc_step_size=0.05 dice.field.total_max_norm=0.05 dice.field.topk=4) ;;
  T) RL_NAME="field_hard8_ff_T_tilted${FTAG}_s${SEED}";        ACTOR=(dice.actor_mode=field_distributional dice.field.q_source=tilted
        dice.field.q_step_size=0.2 dice.field.bc_step_size=0.05 dice.field.total_max_norm=0.05 +dice.field.q_temp=1.0) ;;
  # *2 arms: family-shared TUNED recipe (robomimic winner: q_step 0.5, clip 0.15, restore 1.0)
  B2) RL_NAME="field_hard8_ff_B2_tuned${FTAG}_s${SEED}";       ACTOR=(dice.actor_mode=field_pointwise dice.field.q_source=grad
        dice.field.q_step_size=0.5 dice.field.bc_step_size=0.05 dice.field.total_max_norm=0.15 +dice.field.restore_step_size=1.0 +dice.field.restore_radius=0.05) ;;
  C2) RL_NAME="field_hard8_ff_C2_tuned${FTAG}_s${SEED}";       ACTOR=(dice.actor_mode=field_distributional dice.field.q_source=zeroth
        dice.field.q_step_size=0.5 dice.field.bc_step_size=0.05 dice.field.total_max_norm=0.15 +dice.field.restore_step_size=1.0 +dice.field.restore_radius=0.05 dice.field.topk=4) ;;
  T2) RL_NAME="field_hard8_ff_T2_tuned${FTAG}_s${SEED}";       ACTOR=(dice.actor_mode=field_distributional dice.field.q_source=tilted
        dice.field.q_step_size=0.5 dice.field.bc_step_size=0.05 dice.field.total_max_norm=0.15 +dice.field.restore_step_size=1.0 +dice.field.restore_radius=0.05 +dice.field.q_temp=1.0) ;;
  *) echo "unknown EXP=${EXP} (use A, B, C, T, B2, C2 or T2)"; exit 1 ;;
esac
LOG="${CEDAR}/${RL_NAME}.log"
# dice_train saves under exp_name/ or exp_name/<seed>/ depending on layout — check both.
RESUME_FLAG=""
for RESUME_CKPT in \
    "${CEDAR}/imitation/experiments_dice/libero/libero_90/${RL_NAME}/dice_latest.pth" \
    "${CEDAR}/imitation/experiments_dice/libero/libero_90/${RL_NAME}/${SEED}/dice_latest.pth"; do
  if [ -f "${RESUME_CKPT}" ]; then RESUME_FLAG="+dice_resume_checkpoint=${RESUME_CKPT}"; echo "[$(date)] RESUME ${RESUME_CKPT}"; break; fi
done

echo "[$(date)] START ${RL_NAME} (EXP=${EXP} seed=${SEED}) hard-8 drift base, K=1 RLPD bcw=10"
python dice_train.py \
  algo.num_inference_steps=1 \
  cold_start_checkpoint="${CKPT}" \
  ${RESUME_FLAG} \
  seed=${SEED} \
  'task_indices=[8,21,32,53,65,73,75,81]' \
  dice.bc_loss_weight=100 \
  dice.n_iters=100 \
  dice.episodes_per_iter=32 \
  dice.warmup_episodes=64 \
  dice.gradient_steps=800 \
  dice.eval_interval=10 \
  dice.save_interval=5 \
  dice.replay_size=100000 \
  dice.use_rlpd=true \
  dice.disable_q_loss_for_expert_data=true \
  dice.always_retain_bc_loss_for_expert_data=true \
  dice.replay_flow_warmup_steps=600 \
  dice.use_soft_q_filtering=${SOFTQ} \
  dice.q_filtering_warmup_steps=2000 \
  dice.q_depends_on_noise=false \
  dice.eval_strategy=max_q_min \
  dice.eval_num_samples=10 \
  dice.online_explore_strategy=max_q_min \
  dice.num_exploration_samples=16 \
  dice.num_multi_z=16 \
  dice.use_q_normalization=true \
  "${ACTOR[@]}" \
  "${EXTRA[@]}" \
  logging.mode=disabled \
  output_prefix=${CEDAR}/imitation/experiments_dice \
  exp_name=${RL_NAME} \
  hydra.run.dir="${CEDAR}/imitation/hydra/${RL_NAME}_${SLURM_JOB_ID}" \
  >> "${LOG}" 2>&1 || true

echo "[$(date)] DONE ${RL_NAME}"
grep -E "eval success rate|residual_norm" "${LOG}" | tail -30 || true
