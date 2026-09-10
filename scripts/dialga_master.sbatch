#!/bin/bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1
#SBATCH -p gpu-v100,gpu-l40s,gpu-a100 --time=08:00:00 --mem-per-gpu=48G
#SBATCH --job-name=dialga_master --qos=embers --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%j.err
#
# Self-chaining master job: runs the full queued pipeline in order, resume-safe
# (each driver skips arms/rows already recorded in its results file):
#   1. ext_strict.sh      - Table-3 external strict arms (smoke-first each)
#   2. icl_chain2.sh      - ICL ablations + external keys + 5-task multitask
#   3. seeds.sh           - zdyn/zstatic strict seed repeats
#   4. tokenizer_stage.sh - VidTwin mechanism + VidTwin/VidTok/Cosmos/CV-VAE recon
# Payload is bounded to leave time for the self-resubmit check; if any DONE marker
# is missing at the end, the script resubmits itself.
TMP=/storage/project/r-agarg35-0/lwang831/tmp
mkdir -p /storage/scratch1/8/lwang831/imitation/logs

payload () {
  bash $TMP/ext_strict.sh
  bash $TMP/icl_chain2.sh
  bash $TMP/seeds.sh
  bash $TMP/tokenizer_stage.sh
  bash $TMP/ssv2_warp_stage.sh
}

# 7h20m budget for the payload; the remainder handles the resubmit decision.
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0
timeout 26400 bash -c "$(declare -f payload); payload"

done_all=yes
grep -q "EXT_STRICT DONE"      $TMP/ext_strict_results.txt 2>/dev/null || done_all=no
grep -q "ICL CHAIN2 DONE"      $TMP/icl2_results.txt       2>/dev/null || done_all=no
grep -q "SEEDS DONE"           $TMP/seeds_results.txt      2>/dev/null || done_all=no
grep -q "TOKENIZER SWEEP DONE" $TMP/tokenizer_results.txt  2>/dev/null || done_all=no
grep -q "SSV2_WARP_DONE"        $TMP/ssv2_warp_results.txt   2>/dev/null || done_all=no
if [ "$done_all" = "no" ]; then
  # resubmit only if no other dialga_master is already queued (avoid double-run)
  others=$(squeue -u lwang831 -h -o "%i %j %T" 2>/dev/null | grep dialga_master | grep -v "^${SLURM_JOB_ID} " | wc -l)
  if [ "$others" -eq 0 ]; then
    echo "[master] pipeline incomplete -> self-resubmit $(date)"
    sbatch /storage/home/hcoda1/8/lwang831/workspace/imitation/scripts/dialga_master.sbatch
  else
    echo "[master] incomplete but $others dialga_master already queued -> no resubmit $(date)"
  fi
else
  echo "[master] ALL DONE MARKERS PRESENT $(date)"
fi
