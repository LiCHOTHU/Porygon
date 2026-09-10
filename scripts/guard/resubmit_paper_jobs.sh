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

echo "=== DMC distills (600 epochs), then three arms per task ==="
cd "$DR" || exit 1
for t in quadruped-run walker-run humanoid-walk cartpole-balance; do
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
