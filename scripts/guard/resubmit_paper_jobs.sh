#!/usr/bin/env bash
# Resubmit any of the paper's long runs that are not currently queued or running.
#
# Every one of these is a 12h-5d job on 8-hour preemptible embers slots, so each
# will be killed several times before it finishes. Hand-resubmitting has meant
# runs sitting dead for hours until someone noticed. This sweep is idempotent:
# a job already PENDING/RUNNING is left alone, anything else is relaunched with
# auto_resume so it continues from its last checkpoint.
#
# usage:  bash scripts/guard/resubmit_paper_jobs.sh [--dry-run]
set -u
DRY=${1:-}
DR=/storage/home/hcoda1/8/lwang831/workspace/dice_rl_official
IM=/storage/home/hcoda1/8/lwang831/workspace/imitation
L=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/dice_rl_official/log_dir
D=$L/robomimic-finetune
P=$L/robomimic-pretrain
CEDAR=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch
E=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/experiments_dice/libero/libero_90
CK=$P/square_diff_baselinearch_42/checkpoint/state_8000.pt
EX=$(cat $IM/scripts/flaky_nodes_exclude.txt 2>/dev/null | tr -d '\n')
COMMON="--qos=embers --account=gts-agarg35 --time=8:00:00 --requeue"
[ -n "$EX" ] && COMMON="$COMMON --exclude=$EX"

alive () {  # is a job with this name already queued or running?
  squeue -u "$USER" -h -o "%j %T" 2>/dev/null | awk -v n="$1" '$1==n && ($2=="RUNNING"||$2=="PENDING"){f=1} END{exit !f}'
}

done_already () {  # result-file  [reference-file]
  # An evaluation whose output already exists must not be resubmitted. Without
  # this, a completed eval is queued again on every sweep: four of them burned
  # 111 GPU-hours over 40 submissions re-deriving numbers already on disk.
  [ -f "$1" ] || return 1
  [ -n "${2:-}" ] && [ "$2" -nt "$1" ] && return 1   # checkpoint newer => restale
  return 0
}

go () {  # name  command...
  local name=$1; shift
  if alive "$name"; then echo "  $name: already alive, skipping"; return; fi
  if [ "$DRY" = "--dry-run" ]; then echo "  $name: WOULD resubmit"; return; fi
  local jid
  jid=$(sbatch --parsable --job-name="$name" $COMMON "$@" 2>&1)
  case "$jid" in ''|*[!0-9]*) echo "  $name: FAILED -> $(echo "$jid" | head -c 80)";;
    *) echo "  $name -> $jid";;
  esac
}


# ---------------------------------------------------------------------------
# PRIORITY GATE. Table 1's diffusion base row is the blocking deliverable, and
# the 54-job submission cap is shared: secondary work refilling it is what made
# every base evaluation fail with QOSMaxSubmitJobPerUserLimit while 39 jobs
# "ran". Until a base number exists, only the retrains, their evaluations and
# the watchdog are resubmitted. Delete base_evals/ or touch the override file to
# restore the full sweep.
BASE_DONE=0
for f in $L/base_evals/*/evaluation_results.csv; do
  [ -f "$f" ] && grep -q pretrained "$f" 2>/dev/null && BASE_DONE=1
done
[ -f "$IM/scripts/guard/.run_all" ] && BASE_DONE=1
if [ "$BASE_DONE" -eq 0 ]; then
  echo "PRIORITY: no diffusion base number yet -- running only the base work."
fi
secondary () {  # skip a secondary section while the base is still outstanding
  [ "$BASE_DONE" -eq 1 ] && return 1
  echo "  (skipped: base row still outstanding)"
  return 0
}

echo "=== robomimic long runs (dice-rl repo) ==="
cd "$DR" || exit 1
# dppo_b279 removed: superseded by nb_dppo on the 0.600 base.
go dql_cast scripts/dice_rl_generic.sbatch finetune square ft_dql_diffusion_mlp 42 \
   base_policy_path=$CK train.n_train_itr=300 +train.anchor_rho=0.05 \
   logdir=$D/square_dqlCAST_s42 ++train.auto_resume=true
go square_dipoCAST_anchor scripts/dice_rl_generic.sbatch finetune square ft_dipo_diffusion_mlp 42 \
   base_policy_path=$CK logdir=$D/square_dipoCAST_anchor train.n_train_itr=300 \
   +train.anchor_to_base_radius=0.05 +train.keep_buffer_actions=true ++train.auto_resume=true
go square_dipoCAST_full scripts/dice_rl_generic.sbatch finetune square ft_dipo_diffusion_mlp 42 \
   base_policy_path=$CK logdir=$D/square_dipoCAST_full train.n_train_itr=300 \
   +train.anchor_to_base_radius=0.05 +train.action_trust_radius=0.15 \
   +train.keep_buffer_actions=true train.action_lr=0.005 ++train.auto_resume=true

echo "=== table-7 square column, block (b) ==="
if ! secondary; then
go sq_regress_default scripts/dice_rl_generic.sbatch finetune square ft_distill_residual_drift_field_mlp 42 \
   logdir=$D/sq_regress_default ++train.auto_resume=true
go sq_hinge scripts/dice_rl_generic.sbatch finetune square ft_distill_residual_drift_field_mlp 42 \
   logdir=$D/sq_hinge model.actor_mode=residual +model.bc_hinge_rho=0.05 ++train.auto_resume=true
go sq_proj scripts/dice_rl_generic.sbatch finetune square ft_distill_residual_drift_field_mlp 42 \
   logdir=$D/sq_proj model.field.total_max_norm=1e9 model.field.bc_step_size=0.0 \
   +model.field.project_radius=0.05 ++train.auto_resume=true
go sq_rho0 scripts/dice_rl_generic.sbatch finetune square ft_distill_residual_drift_field_mlp 42 \
   logdir=$D/sq_rho0 +model.field.restore_step_size=1.0 +model.field.restore_radius=0.0 ++train.auto_resume=true

fi
echo "=== LIBERO GRPO cells (matched to the ceiling recipe) ==="
if ! secondary; then
cd "$IM" || exit 1
CEIL="rl.n_iters=200 rl.group_size=16 rl.inits_per_iter=6 rl.filter_low=0.05 rl.filter_high=0.95 rl.eval_interval=5 rl.eval_rollouts_per_env=50"
for t in 8 21 73 81; do
  go grpoC_t${t} scripts/grpo_single_task.sbatch fm ${t} 10000 $CEIL
done

fi
echo "=== table-2 transport (3 seeds x 2 arms) ==="
if ! secondary; then
cd "$DR" || exit 1
for sd in 42 43 44; do
  go transport_DICE_s${sd} scripts/dice_rl_generic.sbatch finetune transport ft_distill_residual_drift_field_mlp ${sd} \
     model.actor_mode=residual logdir=$D/transport_DICE_s${sd} ++train.auto_resume=true
  go transport_CAST_s${sd} scripts/dice_rl_generic.sbatch finetune transport ft_distill_residual_drift_field_mlp ${sd} \
     logdir=$D/transport_CAST_s${sd} ++train.auto_resume=true
done

fi
echo "=== table-1 classic baselines on the 0.563 base ==="
TD64=$P/square_pre_diffusion_mlp_ta4_td20/fixed_42/checkpoint/state_8000.pt
for m in dipo qsm dql idql awr; do
  go b64_${m} scripts/dice_rl_generic.sbatch finetune square ft_${m}_diffusion_mlp 42 \
     base_policy_path=$TD64 model.actor.time_dim=64 \
     '~model.actor.mlp_dims' '~model.actor.cond_mlp_dims' '~model.actor.residual_style' \
     logdir=$D/square_${m}_b64_s42 ++train.auto_resume=true
done

# table-5 factorial training arms removed: all 20 cells are evaluated and in the paper.

# table-3 FM tuning arms removed: the eta=2 column is final in the paper.

echo "=== retrain the square diffusion base (table 1 base row is 0.279 and must move) ==="
cd "$DR" || exit 1
# Pretraining has NO auto_resume: a resubmitted run restarts at epoch 0 and
# OVERWRITES its own checkpoints (state_1000.pt was rewritten after the
# 12000-epoch checkpoint existed). Resubmit only while the target epoch count
# has not been reached.
pretrain_done () { [ -f "$1" ]; }

pretrain_done "$L/robomimic-pretrain/square_pre_diffusion_mlp_ta4_td20_long/checkpoint/state_24000.pt" && echo "  sqdiff_long: finished, not resubmitting" || go sqdiff_long scripts/dice_rl_generic.sbatch pretrain square pre_diffusion_mlp 42 \
   train.n_epochs=24000 logdir=$L/robomimic-pretrain/square_pre_diffusion_mlp_ta4_td20_long
pretrain_done "$L/robomimic-pretrain/square_pre_diffusion_wide_td20/checkpoint/state_12000.pt" && echo "  sqdiff_wide: finished, not resubmitting" || go sqdiff_wide scripts/dice_rl_generic.sbatch pretrain square pre_diffusion_mlp 42 \
   '+model.network.mlp_dims=[1024,1024,1024]' '+model.network.cond_mlp_dims=[512,64]' \
   +model.network.residual_style=True train.n_epochs=12000 \
   logdir=$L/robomimic-pretrain/square_pre_diffusion_wide_td20
pretrain_done "$L/robomimic-pretrain/square_pre_diffusion_wide_td20_lr5e5/checkpoint/state_24000.pt" && echo "  sqdiff_wide_long: finished, not resubmitting" || go sqdiff_wide_long scripts/dice_rl_generic.sbatch pretrain square pre_diffusion_mlp 42 \
   '+model.network.mlp_dims=[1024,1024,1024]' '+model.network.cond_mlp_dims=[512,64]' \
   +model.network.residual_style=True train.n_epochs=24000 train.learning_rate=5e-5 \
   logdir=$L/robomimic-pretrain/square_pre_diffusion_wide_td20_lr5e5
cd "$IM" || exit 1

# Table 1 is final (user sign-off 2026-09-09): the nb_* diffusion rows are
# quoted at matched 496K and their runs are stopped; do not resubmit.

# The strict-metric re-emission block is removed. Its two flow rows kept
# failing on resume (their logdirs hold a 512-wide flow policy, the config
# builds 128) and their table-1 cells are already final measurements; the two
# sq64 rows are superseded by the nb_* runs on the 0.600 base.

echo "=== gym dense-reward suite: pretrains, then the 8-arm fleet per env ==="
cd "$DR" || exit 1
for env in hopper-medium-v2 walker2d-medium-v2 halfcheetah-medium-v2; do
  short=$(echo $env | cut -d- -f1)
  PD=$L/gym-pretrain/${env}_diff
  # Gate the fleet on epoch 1000, not 3000. The official 3000-epoch pretrain is
  # ~4 GPU-days per env on these 1M-transition datasets; diffusion BC saturates
  # far earlier, and quoting every one of the 8 methods from the SAME epoch-1000
  # base keeps the comparison internally fair. Stated in the table caption.
  FINAL=$PD/checkpoint/state_1000.pt
  if [ ! -f "$FINAL" ]; then
    go gymP_${short} scripts/dice_rl_generic.sbatch gym-pretrain $env pre_diffusion_mlp 42 logdir=$PD
    echo "  (fleet for $env waits on its pretrain)"; continue
  fi
  # base exists: the fleet. Official configs for the baselines; our CAST config
  # (grafted onto the env's official env-block) for the paired rows.
  ENV2=$(echo $env | sed 's/-medium//')      # cfg dirs are hopper-v2 etc.
  for m in dipo qsm dql idql awr ppo; do
    go gym_${short}_${m} scripts/dice_rl_generic.sbatch gym-finetune $ENV2 ft_${m}_diffusion_mlp 42 \
       base_policy_path=$FINAL logdir=$L/gym-finetune/${env}_${m} ++train.auto_resume=true
  done
  go gym_${short}_DICE scripts/dice_rl_generic.sbatch gym-finetune $ENV2 ft_distill_residual_diffusion_field_mlp 42 \
     base_policy_path=$FINAL model.actor_mode=residual logdir=$L/gym-finetune/${env}_DICE ++train.auto_resume=true
  go gym_${short}_CAST scripts/dice_rl_generic.sbatch gym-finetune $ENV2 ft_distill_residual_diffusion_field_mlp 42 \
     base_policy_path=$FINAL logdir=$L/gym-finetune/${env}_CAST ++train.auto_resume=true
done
cd "$IM" || exit 1

echo "=== DMC distills for the four good teachers, then their three arms ==="
cd "$DR" || exit 1
for t in quadruped-run walker-run humanoid-walk cartpole-balance; do
  PD=$L/dmc-pretrain/${t}
  FINAL=$(ls $PD/checkpoint/state_*.pt 2>/dev/null | sed 's/.*state_//;s/\.pt//' | sort -n | tail -1)
  if [ "${FINAL:-0}" -lt 3000 ]; then
    go dmcP_${t} scripts/dice_rl_generic.sbatch dmc-pretrain $t pre_drifting_mlp 42 \
       logdir=$PD ++train.auto_resume=true
    echo "  (arms for $t wait on the distill)"; continue
  fi
  CKPT=$PD/checkpoint/state_${FINAL}.pt
  go dmc_${t}_CAST scripts/dice_rl_generic.sbatch dmc-finetune $t ft_distill_residual_drift_field_mlp 42 \
     base_policy_path=$CKPT logdir=$L/dmc-finetune/${t}_CAST ++train.auto_resume=true
  go dmc_${t}_BP scripts/dice_rl_generic.sbatch dmc-finetune $t ft_distill_residual_drift_mlp 42 \
     base_policy_path=$CKPT logdir=$L/dmc-finetune/${t}_BP ++train.auto_resume=true
  go dmc_${t}_FREE scripts/dice_rl_generic.sbatch dmc-finetune $t ft_distill_residual_drift_field_mlp 42 \
     base_policy_path=$CKPT model.field.total_max_norm=1e9 model.field.q_max_norm=1e9 \
     model.field.bc_step_size=0.0 +model.field.restore_step_size=0.0 \
     logdir=$L/dmc-finetune/${t}_FREE ++train.auto_resume=true
done
cd "$IM" || exit 1

echo "=== DMC pair builds (acrobot, cartpole-balance) ==="
cd "$IM" || exit 1
DMC=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/dmc_base
for spec in "humanoid_walk:humanoid walk 1500000" "humanoid_run:humanoid run 2000000" \
            "quadruped_run:quadruped run 800000" "manip_ball:manipulator bring_ball 1500000" \
            "walker_run:walker run 600000"; do
  tag=${spec%%:*}; args=${spec##*:}
  dom=$(echo $args | cut -d' ' -f1); tsk=$(echo $args | cut -d' ' -f2)
  [ -f "$DMC/demos_${dom}_${tsk}.npz" ] && continue
  go dmcT_${tag} scripts/dmc_new_task.sbatch $args
done
[ -f "$DMC/demos_acrobot_swingup.npz" ] || go dmcpair_acrobot scripts/dmc_new_pair.sbatch acrobot swingup swingup_sparse
[ -f "$DMC/demos_cartpole_balance.npz" ] || go dmcpair_cartpole_bal scripts/dmc_new_pair.sbatch cartpole balance balance_sparse

echo "=== table-5 factorial evaluations (tasks 8/53/75) ==="
for t in 8 53 75; do
  for arm in BNONE BCLIP BANC; do
    d=$E/field_st_${arm}_t${t}_s10000
    ck=$(ls $d/dice_latest.pth $d/*/dice_latest.pth 2>/dev/null | head -1)
    [ -z "$ck" ] && { echo "  ${arm}_t${t}: no checkpoint yet"; continue; }
    res=$CEDAR/powered_eval_one_t${t}_${arm}_t${t}.json
    if done_already "$res" "$ck"; then echo "  pe_${arm}_t${t}: result on disk, skipping"; continue; fi
    go pe_${arm}_t${t} --export=ALL,CELL=t${t},LABEL=${arm}_t${t},CKPT=$ck,TASKS="[${t}]" \
       scripts/powered_eval_one.sbatch
  done
done

echo "=== tab:eta-sweep evaluations (fill the 6 appendix cells as training finishes) ==="
cd "$IM" || exit 1
for t in 32 65 81; do
  for tag in 40 80; do
    d=$E/field_st_B_t${t}_fm_eta${tag}_s10000
    ck=$(ls $d/dice_latest.pth $d/*/dice_latest.pth 2>/dev/null | head -1)
    [ -z "$ck" ] && { echo "  fmEta${tag}_t${t}: no checkpoint yet"; continue; }
    res=$CEDAR/powered_eval_one_t${t}_fmEta${tag}_t${t}.json
    if done_already "$res" "$ck"; then echo "  pe_fmEta${tag}_t${t}: result on disk"; continue; fi
    go pe_fmEta${tag}_t${t} --export=ALL,CELL=t${t},LABEL=fmEta${tag}_t${t},CKPT=$ck,TASKS="[${t}]" \
       scripts/powered_eval_one.sbatch
  done
done

echo "=== tab:eta-sweep, flow-matching eta=4 and eta=8 ==="
for t in 32 65 81; do
  for eta in 4.0 8.0; do
    tag=$(echo $eta | tr -d '.')
    go fmEta${tag}_t${t} --export=ALL,BASE_CKPT=$FMCK,NUM_INF_STEPS=10 \
       scripts/field_single_task.sbatch B ${t} 10000 _fm_eta${tag} \
       dice.field.q_step_size=${eta}
  done
done

echo "=== table-4 multitask evaluations ==="
for spec in "castMT_s10001:field_hard8_ff_B_grad_s10001" \
            "topk_s10000:field_hard8_ff_C_zeroth_s10000" \
            "topk_s10002:field_hard8_ff_C_zeroth_s10002" \
            "tilted_s10001:field_hard8_ff_T_tilted_s10001"; do
  lab=${spec%%:*}; arm=${spec##*:}
  ck=$(ls $E/$arm/dice_latest.pth $E/$arm/*/dice_latest.pth 2>/dev/null | head -1)
  [ -z "$ck" ] && { echo "  $lab: no checkpoint"; continue; }
  res=$CEDAR/powered_eval_one_hard8_${lab}.json
  if done_already "$res" "$ck"; then echo "  pe_${lab}: result already on disk, skipping"; continue; fi
  go pe_${lab} --export=ALL,CELL=hard8,LABEL=${lab},CKPT=$ck scripts/powered_eval_one.sbatch
done
echo "done."
