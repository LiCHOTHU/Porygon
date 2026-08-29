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
E=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/experiments_dice/libero/libero_90
CK=$P/square_diff_baselinearch_42/checkpoint/state_8000.pt
EX=$(cat $IM/scripts/flaky_nodes_exclude.txt 2>/dev/null | tr -d '\n')
COMMON="--qos=embers --account=gts-agarg35 --time=8:00:00 --requeue"
[ -n "$EX" ] && COMMON="$COMMON --exclude=$EX"

alive () {  # is a job with this name already queued or running?
  squeue -u "$USER" -h -o "%j %T" 2>/dev/null | awk -v n="$1" '$1==n && ($2=="RUNNING"||$2=="PENDING"){f=1} END{exit !f}'
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

echo "=== robomimic long runs (dice-rl repo) ==="
cd "$DR" || exit 1
go dppo_b279 scripts/dice_rl_generic.sbatch finetune square ft_ppo_diffusion_mlp 42 \
   base_policy_path=$CK model.actor.time_dim=32 \
   '+model.actor.mlp_dims=[1024,1024,1024]' '+model.actor.cond_mlp_dims=[512,64]' \
   +model.actor.residual_style=True logdir=$L/square_dppo_b279_s42 ++train.auto_resume=true
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

echo "=== LIBERO GRPO cells (matched to the ceiling recipe) ==="
cd "$IM" || exit 1
CEIL="rl.n_iters=200 rl.group_size=16 rl.inits_per_iter=6 rl.filter_low=0.05 rl.filter_high=0.95 rl.eval_interval=5 rl.eval_rollouts_per_env=50"
for t in 8 21 73 81; do
  go grpoC_t${t} scripts/grpo_single_task.sbatch fm ${t} 10000 $CEIL
done

echo "=== table-4 multitask evaluations ==="
for spec in "castMT_s10001:field_hard8_ff_B_grad_s10001" \
            "topk_s10000:field_hard8_ff_C_zeroth_s10000" \
            "topk_s10002:field_hard8_ff_C_zeroth_s10002" \
            "tilted_s10001:field_hard8_ff_T_tilted_s10001"; do
  lab=${spec%%:*}; arm=${spec##*:}
  ck=$(ls $E/$arm/dice_latest.pth $E/$arm/*/dice_latest.pth 2>/dev/null | head -1)
  [ -z "$ck" ] && { echo "  $lab: no checkpoint"; continue; }
  go pe_${lab} --export=ALL,CELL=hard8,LABEL=${lab},CKPT=$ck scripts/powered_eval_one.sbatch
done
echo "done."
