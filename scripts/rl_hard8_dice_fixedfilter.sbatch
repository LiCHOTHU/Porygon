#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=rl_hard8_dice_ff
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.out
#SBATCH --error=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.err

# ============================================================
# Hard-8 DICE-RL relaunch with the BC-FILTER SEMANTICS FIX (2026-07-04).
# Bug: our port applied the BC anchor unconditionally when use_soft_q_filtering=false;
# official semantics = HARD filter (drop BC where Q(student) > Q(teacher)). The
# paper's "selective behavior regularization" never fired in ANY previous run
# (qfilt_keep==1.000 always; residual pinned ~0.003; DICE==BC).
# This relaunch is the faithful official recipe: bcw=100 (official cfg value,
# designed to work WITH the filter), K=16, max_q_min collect+eval, RLPD, n_step=3.
#   FM    = multi-step flow base (10 steps)  — the paper's setting
#   DRIFT = one-step drift base              — our drifting model
#   usage:  sbatch scripts/rl_hard8_dice_fixedfilter.sbatch <FM|DRIFT>
# Watch: qfilt_keep should drop BELOW 1.0 post-warmup and residual_norm should
# move off 0.003 — the two signatures the bug suppressed.
# ============================================================
BASE="${1:?usage: sbatch rl_hard8_dice_fixedfilter.sbatch <FM|DRIFT>}"
CONDA_ENV="porygon"; WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/imitation"
SCR="/storage/scratch1/8/lwang831"
CEDAR="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch"
cd "${WORKDIR}" || exit 1; source ~/.bashrc; conda activate "${CONDA_ENV}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MUJOCO_GL=egl EGL_DEVICE_ID=0 HYDRA_FULL_ERROR=1
nvidia-smi || true

case "${BASE}" in
  FM)    CKPT="${SCR}/imitation/cold_start/libero/libero_90/cold_multitask_lib90/multitask_model_latest.pth";  NSTEPS=10 ;;
  DRIFT) CKPT="${SCR}/imitation/cold_start/libero/libero_90/drift_multitask_lib90/multitask_model_latest.pth"; NSTEPS=1  ;;
  *) echo "unknown BASE=${BASE} (use FM or DRIFT)"; exit 1 ;;
esac
RL_NAME="rl_hard8_dice_ff_$(echo ${BASE} | tr 'A-Z' 'a-z')"
LOG="${CEDAR}/${RL_NAME}.log"
RESUME_CKPT="${CEDAR}/imitation/experiments_dice/libero/libero_90/${RL_NAME}/dice_latest.pth"
RESUME_FLAG=""; [ -f "${RESUME_CKPT}" ] && RESUME_FLAG="+dice_resume_checkpoint=${RESUME_CKPT}" && echo "RESUME ${RESUME_CKPT}"
[ -f "${CKPT}" ] || { echo "ERROR: base ckpt missing ${CKPT}"; exit 1; }

echo "[$(date)] START ${RL_NAME} — FIXED-FILTER faithful recipe, base=${BASE} K_steps=${NSTEPS} bcw=100"
python dice_train.py \
  algo.num_inference_steps=${NSTEPS} \
  cold_start_checkpoint="${CKPT}" \
  ${RESUME_FLAG} \
  'task_indices=[8,21,32,53,65,73,75,81]' \
  dice.bc_loss_weight=100 \
  dice.use_soft_q_filtering=false \
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
  dice.q_filtering_warmup_steps=2000 \
  dice.q_depends_on_noise=false \
  dice.eval_strategy=max_q_min \
  dice.eval_num_samples=10 \
  dice.online_explore_strategy=max_q_min \
  dice.num_exploration_samples=16 \
  dice.num_multi_z=16 \
  dice.use_q_normalization=true \
  logging.mode=disabled \
  output_prefix=${CEDAR}/imitation/experiments_dice \
  exp_name=${RL_NAME} \
  hydra.run.dir="${CEDAR}/imitation/hydra/${RL_NAME}_${SLURM_JOB_ID}" \
  >> "${LOG}" 2>&1 || true

echo "[$(date)] DONE ${RL_NAME}"
echo "=== signatures: qfilt_keep must drop <1.0, residual must move off 0.003 ==="
grep -oE "qfilt_keep=[0-9.]+" "${LOG}" | tail -5
grep -E "eval success rate|BC baseline" "${LOG}" | tail -15