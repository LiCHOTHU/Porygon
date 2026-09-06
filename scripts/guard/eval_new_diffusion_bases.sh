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
  # Highest EPOCH, not newest mtime: a restarted pretrain rewrites low-epoch
  # checkpoints with fresh timestamps, so mtime ordering picks epoch 1000 over
  # the finished 24000 and the "base number" quietly becomes a barely-trained net.
  for ck in $(ls $P/$run/*/checkpoint/state_*.pt $P/$run/checkpoint/state_*.pt 2>/dev/null \
              | sort -t_ -k2 -n | sed 's/.*state_//;s/\.pt//' >/dev/null; \
              ls $P/$run/*/checkpoint/state_*.pt $P/$run/checkpoint/state_*.pt 2>/dev/null \
              | awk -F'state_' '{split($2,a,".pt"); print a[1], $0}' | sort -n | tail -1 | cut -d' ' -f2); do
    ep=$(basename "$ck" .pt | sed 's/state_//')
    tag="basev_${run: -12}_${ep}"
    dst=$OUT/${run}_${ep}
    [ -f "$dst/evaluation_results.csv" ] && grep -q pretrained "$dst/evaluation_results.csv" 2>/dev/null && continue
    alive "$tag" && { echo "  $tag: alive"; continue; }
    # NO architecture overrides. DistillResidualRLModel._load_pretrained_policy
    # rebuilds the base from the checkpoint's OWN .hydra/config.yaml, so this
    # config has no model.actor node at all and any such override aborts the job
    # in ~13s with "Key 'actor' is not in struct". Passing base_policy_path alone
    # is what makes a wide or narrow trunk load correctly.
    ARCH=()
    jid=$(sbatch --parsable --job-name="$tag" $COMMON scripts/dice_rl_generic.sbatch \
      finetune square ft_distill_residual_drift_field_mlp 42 \
      base_policy_path="$ck" train.n_train_itr=1 logdir="$dst" 2>&1)
    case "$jid" in ''|*[!0-9]*) echo "  $tag: FAILED -> $(echo "$jid"|head -c 60)";;
      *) echo "  $tag -> $jid";; esac
  done
done
