#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100,gpu-v100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=rl_hard8_dice_t2
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.out
#SBATCH --error=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/%x_%A.err

# ============================================================
# Tier-1 hard-8 DICE-RL experiment matrix (see README "Experiment plan").
#   usage: sbatch scripts/rl_hard8_dice_tier1.sbatch <FM|DRIFT> <hard|guarded> <SEED>
#   FILTER=hard    -> dice.use_soft_q_filtering=false = our unguarded hard filter
#                     (drop BC where Q(edited) > Q(base); diverges from official
#                     False=no-filter semantics on purpose)
#   FILTER=guarded -> dice.use_soft_q_filtering=true  = the paper's exact Eq. 6
#                     (advantage condition AND overestimation guard vs MC return,
#                     eps = q_underestimation_threshold = -0.5)
# Everything else is the faithful official recipe (bcw=100, K=16, max_q_min
# collect+eval, RLPD, n_step, ensemble). 4-deep afternotok chains resume from
# dice_latest.pth.
# ============================================================
BASE="${1:?usage: sbatch rl_hard8_dice_tier1.sbatch <FM|DRIFT> <hard|guarded> <SEED>}"
FILTER="${2:?usage: sbatch rl_hard8_dice_tier1.sbatch <FM|DRIFT> <hard|guarded> <SEED>}"
SEED="${3:?usage: sbatch rl_hard8_dice_tier1.sbatch <FM|DRIFT> <hard|guarded> <SEED>}"
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
case "${FILTER}" in
  hard)    SOFTQ=false; FTAG=ff ;;
  guarded) SOFTQ=true;  FTAG=gf ;;
  *) echo "unknown FILTER=${FILTER} (use hard or guarded)"; exit 1 ;;
esac
RL_NAME="rl_hard8_dice_aln0_${FTAG}_$(echo ${BASE} | tr 'A-Z' 'a-z')_s${SEED}"
LOG="${CEDAR}/${RL_NAME}.log"
# Non-default seeds save under a <seed>/ subdir of the experiment dir; check both.
EXPDIR="${CEDAR}/imitation/experiments_dice/libero/libero_90/${RL_NAME}"
RESUME_CKPT="${EXPDIR}/${SEED}/dice_latest.pth"
[ -f "${RESUME_CKPT}" ] || RESUME_CKPT="${EXPDIR}/dice_latest.pth"
RESUME_FLAG=""; [ -f "${RESUME_CKPT}" ] && RESUME_FLAG="+dice_resume_checkpoint=${RESUME_CKPT}" && echo "RESUME ${RESUME_CKPT}"
[ -f "${CKPT}" ] || { echo "ERROR: base ckpt missing ${CKPT}"; exit 1; }

echo "[$(date)] START ${RL_NAME} base=${BASE} filter=${FILTER} seed=${SEED} K_steps=${NSTEPS} bcw=100"
python dice_train.py \
  seed=${SEED} \
  algo.num_inference_steps=${NSTEPS} \
  cold_start_checkpoint="${CKPT}" \
  ${RESUME_FLAG} \
  'task_indices=[8,21,32,53,65,73,75,81]' \
  dice.bc_loss_weight=100 \
  dice.use_soft_q_filtering=${SOFTQ} \
  dice.actor_layernorm=false \
  dice.q_underestimation_threshold=-0.5 \
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
grep -oE "qfilt_keep=[0-9.]+" "${LOG}" | tail -3
grep -E "eval success rate|BC baseline" "${LOG}" | tail -12
