#!/usr/bin/env bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=1 --mem=4G
#SBATCH -p cpu-small,inferno
#SBATCH --time=8:00:00
#SBATCH --qos=embers --account=gts-agarg35 --requeue
#SBATCH --job-name=watchdog
#SBATCH --output=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs/watchdog_%A.out
#
# Runs the resubmit sweep and the base-checkpoint evaluator every 20 minutes,
# then resubmits itself. Without this, preempted work sits dead until someone
# notices -- which has repeatedly cost a full day.
IM=/storage/home/hcoda1/8/lwang831/workspace/imitation
END=$(( $(date +%s) + 7*3600 + 1800 ))
while [ "$(date +%s)" -lt "$END" ]; do
  echo "=== $(date) ==="
  bash $IM/scripts/guard/resubmit_paper_jobs.sh 2>&1 | grep -vE "already alive|result on disk|skipping" | head -30
  bash $IM/scripts/guard/eval_new_diffusion_bases.sh 2>&1 | head -10
  sleep 1200
done
sbatch $IM/scripts/guard/watchdog.sh
