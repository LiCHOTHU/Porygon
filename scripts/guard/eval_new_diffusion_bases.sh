#!/usr/bin/env bash
# Evaluate every square diffusion pretrain checkpoint that does not yet have a
# base number, so table 1's base row can be updated from real measurements
# rather than waiting for a full retrain to finish. A finetune job with
# n_train_itr=1 writes an eval_type=pretrained row at 300 episodes, which is
# exactly the base success rate in the harness table 1 uses.
set -u
DR=/storage/home/hcoda1/8/lwang831/workspace/dice_rl_official
L=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/dice_rl_official/log_dir
P=$L/robomimic-pretrain
OUT=$L/base_evals
mkdir -p "$OUT"
EX=$(cat /storage/home/hcoda1/8/lwang831/workspace/imitation/scripts/flaky_nodes_exclude.txt 2>/dev/null | tr -d '\n')
COMMON="--qos=embers --account=gts-agarg35 --time=2:00:00 --requeue"
[ -n "$EX" ] && COMMON="$COMMON --exclude=$EX"
cd "$DR" || exit 1

alive () { squeue -u "$USER" -h -o "%j %T" 2>/dev/null | awk -v n="$1" '$1==n && ($2=="RUNNING"||$2=="PENDING"){f=1} END{exit !f}'; }

for run in square_pre_diffusion_mlp_ta4_td20_long square_pre_diffusion_wide_td20 square_pre_diffusion_wide_td20_lr5e5; do
  # Newest checkpoint only. Submitting three per variant filled the 54-job cap
  # and every base evaluation then failed with QOSMaxSubmitJobPerUserLimit --
  # the retrains ran but nothing scored them, which is the whole point.
  for ck in $(ls -t $P/$run/*/checkpoint/state_*.pt $P/$run/checkpoint/state_*.pt 2>/dev/null | head -1); do
    ep=$(basename "$ck" .pt | sed 's/state_//')
    tag="basev_${run: -12}_${ep}"
    dst=$OUT/${run}_${ep}
    [ -f "$dst/evaluation_results.csv" ] && grep -q pretrained "$dst/evaluation_results.csv" 2>/dev/null && continue
    alive "$tag" && { echo "  $tag: alive"; continue; }
    # wide runs need their trunk declared so the checkpoint loads
    ARCH=()
    case "$run" in *wide*) ARCH=(model.actor.time_dim=64
        '+model.actor.mlp_dims=[1024,1024,1024]' '+model.actor.cond_mlp_dims=[512,64]'
        +model.actor.residual_style=True) ;;
      *) ARCH=(model.actor.time_dim=64 '~model.actor.mlp_dims' '~model.actor.cond_mlp_dims'
        '~model.actor.residual_style') ;;
    esac
    jid=$(sbatch --parsable --job-name="$tag" $COMMON scripts/dice_rl_generic.sbatch \
      finetune square ft_distill_residual_drift_field_mlp 42 \
      base_policy_path="$ck" "${ARCH[@]}" train.n_train_itr=1 logdir="$dst" 2>&1)
    case "$jid" in ''|*[!0-9]*) echo "  $tag: FAILED -> $(echo "$jid"|head -c 60)";;
      *) echo "  $tag -> $jid";; esac
  done
done
