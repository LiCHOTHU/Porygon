#!/bin/bash
# Run the 5 official diffusion-RL baselines START-TO-FINISH in ONE job each.
#
# Why not embers+resume: these are off-policy learners whose replay buffers are
# deque locals inside run(). Any resume that does not also persist those buffers
# silently restarts learning from a near-empty buffer, which wrecked DQL
# (critic loss 0.08 -> 39 -> 67) and collapsed QSM to 0.000. Rather than
# reimplement their training loop to checkpoint buffers, we run the OFFICIAL
# code unmodified inside a single non-preemptible 30h slot (GPU partitions
# allow 3 days). ~20h needed at the measured 4 min/iter.
set -eu
BASE=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/dice_rl_official/log_dir/robomimic-pretrain/square_diff_baselinearch_42/checkpoint/state_8000.pt
ROOT=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/dice_rl_official/log_dir/robomimic-finetune
SEED=${SEED:-42}
ACCT=${ACCT:-gts-agarg35-ideas_l40s}
[ -f "$BASE" ] || { echo "FATAL: base missing"; exit 1; }

for b in dipo qsm dql idql awr; do
  jid=$(sbatch --parsable --job-name=${b}_inf --qos=inferno --time=30:00:00 --account=${ACCT} \
    --cpus-per-task=12 \
    ~/workspace/dice_rl_official/scripts/dice_rl_generic.sbatch \
    finetune square ft_${b}_diffusion_mlp ${SEED} \
    base_policy_path=$BASE logdir=$ROOT/square_${b}_inferno_s${SEED} \
    train.n_train_itr=300)
  echo "  ${b}: ${jid}"
done
