#!/usr/bin/env bash
# Idempotent resubmit sweep, run by the watchdog every 20 minutes.
#
# CURRENT PHASE: DMC focus (user directive 2026-09-09) -- verify CAST on both
# dense and sparse reward. Only DMC work and the transport diagnostic run;
# the gym dense suite is parked behind DMC_ONLY=0. Sections for tables that
# are finished and signed off have been removed (history: sweep.bak / git).
set -u
DRY=${1:-}
DMC_ONLY=1
DR=/storage/home/hcoda1/8/lwang831/workspace/dice_rl_official
IM=/storage/home/hcoda1/8/lwang831/workspace/imitation
L=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/dice_rl_official/log_dir
D=$L/robomimic-finetune
DMC=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/dmc_base
EX=$(cat $IM/scripts/flaky_nodes_exclude.txt 2>/dev/null | tr -d '\n')
COMMON="--qos=embers --account=gts-agarg35 --time=8:00:00 --requeue"
[ -n "$EX" ] && COMMON="$COMMON --exclude=$EX"

alive () { squeue -u "$USER" -h -o "%j %T" 2>/dev/null | awk -v n="$1" '$1==n && ($2=="RUNNING"||$2=="PENDING"){f=1} END{exit !f}'; }
go () {
  local name=$1; shift
  if alive "$name"; then echo "  $name: alive"; return; fi
  if [ "$DRY" = "--dry-run" ]; then echo "  $name: WOULD resubmit"; return; fi
  local jid; jid=$(sbatch --parsable --job-name="$name" $COMMON "$@" 2>&1)
  case "$jid" in ''|*[!0-9]*) echo "  $name: FAILED -> $(echo "$jid"|head -c 70)";;
    *) echo "  $name -> $jid";; esac
}

echo "=== DMC teachers still needed ==="
cd "$IM" || exit 1
[ -f "$DMC/demos_manipulator_bring_ball.npz" ] || go dmcT_manip_ball scripts/dmc_new_task.sbatch manipulator bring_ball 4000000
[ -f "$DMC/demos_finger_spin.npz" ] || go dmcT_finger_spin scripts/dmc_new_task.sbatch finger spin 500000
[ -f "$DMC/demos_ball_in_cup_catch.npz" ] || go dmcT_bic_catch scripts/dmc_new_task.sbatch ball_in_cup catch 400000
[ -f "$DMC/demos_reacher_hard.npz" ] || go dmcT_reacher_hard scripts/dmc_new_task.sbatch reacher hard 400000

echo "=== replacement-task distills (one job, three tasks) ==="
cd "$IM" || exit 1
NT_DONE=1
for t in finger-spin ball_in_cup-catch reacher-hard; do
  hi=$(ls $L/dmc-pretrain/${t}/checkpoint/state_*.pt 2>/dev/null | sed 's/.*state_//;s/\.pt//' | sort -n | tail -1)
  [ "${hi:-0}" -lt 600 ] && NT_DONE=0
done
[ "$NT_DONE" -eq 0 ] && go dmcP_newtasks scripts/dmc_newtask_distills.sbatch
if [ "$NT_DONE" -eq 1 ]; then
  cd "$DR" || exit 1
  for t in finger-spin ball_in_cup-catch reacher-hard; do
    CKPT=$L/dmc-pretrain/${t}/checkpoint/state_600.pt
    go dmc_${t}_CAST scripts/dice_rl_generic.sbatch dmc-finetune $t ft_distill_residual_drift_field_mlp 42 \
       base_policy_path=$CKPT logdir=$L/dmc-finetune/${t}_CAST ++train.auto_resume=true
    go dmc_${t}_BP scripts/dice_rl_generic.sbatch dmc-finetune $t ft_distill_residual_drift_mlp 42 \
       base_policy_path=$CKPT logdir=$L/dmc-finetune/${t}_BP ++train.auto_resume=true
    go dmc_${t}_FREE scripts/dice_rl_generic.sbatch dmc-finetune $t ft_distill_residual_drift_field_mlp 42 \
       base_policy_path=$CKPT model.field.total_max_norm=1e9 +model.field.q_max_norm=1e9 \
       model.field.bc_step_size=0.0 ++model.field.restore_step_size=0.0 \
       logdir=$L/dmc-finetune/${t}_FREE ++train.auto_resume=true
  done
  cd "$IM" || exit 1
fi

echo "=== DMC distills (600 epochs), then three arms per task ==="
cd "$DR" || exit 1
for t in quadruped-run cartpole-balance; do   # walker/humanoid deleted: distillation-infeasible
  PD=$L/dmc-pretrain/${t}
  FINAL=$(ls $PD/checkpoint/state_*.pt 2>/dev/null | sed 's/.*state_//;s/\.pt//' | sort -n | tail -1)
  if [ "${FINAL:-0}" -lt 600 ]; then
    go dmcP_${t} scripts/dice_rl_generic.sbatch dmc-pretrain $t pre_drifting_mlp 42 \
       logdir=$PD ++train.auto_resume=true
    continue
  fi
  CKPT=$PD/checkpoint/state_${FINAL}.pt
  go dmc_${t}_CAST scripts/dice_rl_generic.sbatch dmc-finetune $t ft_distill_residual_drift_field_mlp 42 \
     base_policy_path=$CKPT logdir=$L/dmc-finetune/${t}_CAST ++train.auto_resume=true
  go dmc_${t}_BP scripts/dice_rl_generic.sbatch dmc-finetune $t ft_distill_residual_drift_mlp 42 \
     base_policy_path=$CKPT logdir=$L/dmc-finetune/${t}_BP ++train.auto_resume=true
  go dmc_${t}_FREE scripts/dice_rl_generic.sbatch dmc-finetune $t ft_distill_residual_drift_field_mlp 42 \
     base_policy_path=$CKPT model.field.total_max_norm=1e9 +model.field.q_max_norm=1e9 \
     model.field.bc_step_size=0.0 ++model.field.restore_step_size=0.0 \
     logdir=$L/dmc-finetune/${t}_FREE ++train.auto_resume=true
done

echo "=== sparse arms: cartpole-balance_sparse (shares the dense distill) ==="
PD=$L/dmc-pretrain/cartpole-balance
FINAL=$(ls $PD/checkpoint/state_*.pt 2>/dev/null | sed 's/.*state_//;s/\.pt//' | sort -n | tail -1)
if [ "${FINAL:-0}" -ge 600 ]; then
  CKPT=$PD/checkpoint/state_${FINAL}.pt
  go dmc_balS_CAST scripts/dice_rl_generic.sbatch dmc-finetune cartpole-balance_sparse ft_distill_residual_drift_field_mlp 42 \
     base_policy_path=$CKPT logdir=$L/dmc-finetune/cartpole-balance_sparse_CAST ++train.auto_resume=true
  go dmc_balS_BP scripts/dice_rl_generic.sbatch dmc-finetune cartpole-balance_sparse ft_distill_residual_drift_mlp 42 \
     base_policy_path=$CKPT logdir=$L/dmc-finetune/cartpole-balance_sparse_BP ++train.auto_resume=true
  go dmc_balS_FREE scripts/dice_rl_generic.sbatch dmc-finetune cartpole-balance_sparse ft_distill_residual_drift_field_mlp 42 \
     base_policy_path=$CKPT model.field.total_max_norm=1e9 +model.field.q_max_norm=1e9 \
     model.field.bc_step_size=0.0 ++model.field.restore_step_size=0.0 \
     logdir=$L/dmc-finetune/cartpole-balance_sparse_FREE ++train.auto_resume=true
fi

echo "=== tuned CAST variants (quadruped, balance) and wide rescue distills ==="
cd "$DR" || exit 1
QP=$L/dmc-pretrain/quadruped-run/checkpoint/state_600.pt
BPC=$L/dmc-pretrain/cartpole-balance/checkpoint/state_600.pt
go dmcT_quad_e2 scripts/dice_rl_generic.sbatch dmc-finetune quadruped-run ft_distill_residual_drift_field_mlp 42 \
  base_policy_path=$QP model.field.q_step_size=1.0 \
  logdir=$L/dmc-finetune/quadruped-run_CASTe2 ++train.auto_resume=true
go dmcT_quad_dz scripts/dice_rl_generic.sbatch dmc-finetune quadruped-run ft_distill_residual_drift_field_mlp 42 \
  base_policy_path=$QP ++model.field.restore_radius=0.05 ++model.field.restore_radius_rms=true \
  model.field.total_max_norm=0.25 \
  logdir=$L/dmc-finetune/quadruped-run_CASTdz ++train.auto_resume=true
go dmcT_bal_e2 scripts/dice_rl_generic.sbatch dmc-finetune cartpole-balance ft_distill_residual_drift_field_mlp 42 \
  base_policy_path=$BPC model.field.q_step_size=1.0 \
  logdir=$L/dmc-finetune/cartpole-balance_CASTe2 ++train.auto_resume=true
# dmcP3 wide distills run LOCALLY (do not resubmit; logdirs owned by local run)
cd "$IM" || exit 1

echo "=== Table 7 drifting-base square cells (matched reward field) ==="
cd "$DR" || exit 1
DB=$L/robomimic-pretrain/square_pre_drifting_mlp_ta4_td1/fixed_42/checkpoint/state_8000.pt
for sd in 42 43 44; do
  go sqL_rho0_s$sd scripts/dice_rl_generic.sbatch finetune square ft_distill_residual_drift_field_mlp $sd \
    base_policy_path=$DB model.field.q_source=grad +model.field.restore_step_size=1.0 +model.field.restore_radius=0.0 \
    logdir=$L/robomimic-finetune/sqL_rho0_s$sd ++train.auto_resume=true
  go sqL_proj_s$sd scripts/dice_rl_generic.sbatch finetune square ft_distill_residual_drift_field_mlp $sd \
    base_policy_path=$DB model.field.q_source=grad model.field.bc_step_size=0.0 +model.field.restore_step_size=0.0 +model.field.project_radius=0.05 \
    logdir=$L/robomimic-finetune/sqL_proj_s$sd ++train.auto_resume=true
done
cd "$IM" || exit 1

echo "=== Experiment 1: square hard-projection (proper field), 3 seeds ==="
cd "$DR" || exit 1
NB=$L/robomimic-pretrain/square_pre_diffusion_wide_td20_lr5e5/checkpoint/state_24000.pt
for sd in 42 43 44; do
  go sq_hinge2_s${sd} scripts/dice_rl_generic.sbatch finetune square ft_distill_residual_diffusion_field_mlp ${sd} \
     base_policy_path=$NB model.actor_mode=residual +model.bc_hinge_rho=0.05 \
     logdir=$L/robomimic-finetune/square_hinge2_s${sd} ++train.auto_resume=true
  go sq_proj2_s${sd} scripts/dice_rl_generic.sbatch finetune square ft_distill_residual_diffusion_field_mlp ${sd} \
     base_policy_path=$NB model.field.bc_step_size=0.0 ++model.field.restore_step_size=0.0 \
     +model.field.project_radius=0.05 \
     logdir=$L/robomimic-finetune/square_proj2_s${sd} ++train.auto_resume=true
done
cd "$IM" || exit 1

echo "=== Experiment 1: hinge/proj seeds on LIBERO t65/t32, then powered evals ==="
cd "$IM" || exit 1
EDIR=$CEDAR/imitation/experiments_dice/libero/libero_90 2>/dev/null || EDIR=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/experiments_dice/libero/libero_90
CED=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch
for spec in "hinge 65 10002 field_st_A_t65_hinge_s10002" "hinge 32 10002 field_st_A_t32_hinge_s10002" \
            "proj 65 10001 field_st_BNONE_t65_proj_s10001" "proj 65 10002 field_st_BNONE_t65_proj_s10002" \
            "proj 32 10001 field_st_BNONE_t32_proj_s10001" "proj 32 10002 field_st_BNONE_t32_proj_s10002" \
            "hinge 65 10001 field_st_A_t65_hinge_s10001" "hinge 32 10001 field_st_A_t32_hinge_s10001" \
            "B 65 10001 field_st_B_t65_s10001" "B 32 10001 field_st_B_t32_s10001" \
            "B 65 20002 field_st_B_t65_s20002" "B 32 20002 field_st_B_t32_s20002"; do
  set -- $spec; arm=$1; t=$2; sd=$3; name=$4
  ck=$EDIR/$name/dice_latest.pth
  res=$CED/powered_eval_one_t${t}_${arm}_s${sd}.json
  if [ ! -f "$ck" ]; then
    go e1_${arm}_t${t}_s${sd} scripts/exp1_arm.sbatch $arm $t $sd
  elif [ ! -f "$res" ] || [ "$ck" -nt "$res" ]; then
    go pe_e1_${arm}_t${t}_s${sd} --export=ALL,CELL=t${t},LABEL=${arm}_s${sd},CKPT=$ck,TASKS="[${t}]" \
       scripts/powered_eval_one.sbatch
  fi
done

echo "=== transport leash diagnostic (answers whether the bound caps CAST there) ==="
go transport_CASTloose scripts/dice_rl_generic.sbatch finetune transport ft_distill_residual_drift_field_mlp 42 \
   ++model.field.restore_radius=0.15 model.field.total_max_norm=0.45 \
   logdir=$D/transport_CASTloose_s42 ++train.auto_resume=true

if [ "$DMC_ONLY" -eq 0 ]; then
echo "=== gym dense-reward suite (parked) ==="
for env in hopper-medium-v2 walker2d-medium-v2 halfcheetah-medium-v2; do
  short=$(echo $env | cut -d- -f1)
  PD=$L/gym-pretrain/${env}_diff
  FINAL=$PD/checkpoint/state_1000.pt
  if [ ! -f "$FINAL" ]; then
    go gymP_${short} scripts/dice_rl_generic.sbatch gym-pretrain $env pre_diffusion_mlp 42 \
       logdir=$PD ++train.auto_resume=true
    continue
  fi
  ENV2=$(echo $env | sed 's/-medium//')
  for m in dipo qsm dql idql awr ppo; do
    go gym_${short}_${m} scripts/dice_rl_generic.sbatch gym-finetune $ENV2 ft_${m}_diffusion_mlp 42 \
       base_policy_path=$FINAL logdir=$L/gym-finetune/${env}_${m} ++train.auto_resume=true
  done
  go gym_${short}_DICE scripts/dice_rl_generic.sbatch gym-finetune $ENV2 ft_distill_residual_diffusion_field_mlp 42 \
     base_policy_path=$FINAL model.actor_mode=residual logdir=$L/gym-finetune/${env}_DICE ++train.auto_resume=true
  go gym_${short}_CAST scripts/dice_rl_generic.sbatch gym-finetune $ENV2 ft_distill_residual_diffusion_field_mlp 42 \
     base_policy_path=$FINAL logdir=$L/gym-finetune/${env}_CAST ++train.auto_resume=true
done
fi
echo "done."
