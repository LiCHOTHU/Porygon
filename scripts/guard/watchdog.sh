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
# Submit the successor NOW, not at exit: a successor submitted at exit sits in
# the queue with zero accumulated priority while every preempted job stays dead
# -- that cost 13 hours overnight (watchdog ended 23:58, successor started
# 13:15). Submitted here, it accrues queue priority for 8h while this one runs.
# Successor guard: count ALL watchdog jobs other than this one, in any state.
# Guarding only on PENDING caused a spawn loop -- a successor that started
# running on a free CPU node no longer counted, so every generation submitted
# another; 13 accumulated within hours and ate 13 of the 54 submit slots.
n_others=$(squeue -u "$USER" -h -o "%i %j" | awk -v me="$SLURM_JOB_ID" '$2=="watchdog" && $1!=me' | wc -l)
if [ "$n_others" -eq 0 ]; then
  sbatch $IM/scripts/guard/watchdog.sh
fi
END=$(( $(date +%s) + 7*3600 + 1800 ))
while [ "$(date +%s)" -lt "$END" ]; do
  echo "=== $(date) ==="
  bash $IM/scripts/guard/resubmit_paper_jobs.sh 2>&1 | grep -vE "already alive|result on disk|skipping" | head -30
  bash $IM/scripts/guard/eval_new_diffusion_bases.sh 2>&1 | head -10
  sleep 1200
done
# successor already submitted at start
