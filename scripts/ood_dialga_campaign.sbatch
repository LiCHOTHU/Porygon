#!/bin/bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1
#SBATCH -p gpu-v100,gpu-a100,gpu-l40s --time=08:00:00 --mem-per-gpu=48G
#SBATCH --job-name=ood_dialga --qos=embers --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/scratch1/8/lwang831/imitation/logs/%x_%A.out
#SBATCH --error=/storage/scratch1/8/lwang831/imitation/logs/%x_%A.err
#
# OOD generalization: multi-task train on 80 LIBERO-90 tasks, eval on 10 disjoint
# held-out tasks (seed=42 split). 5 arms. PREEMPT-SAFE: --requeue + train.py resume
# (training.resume=true reloads multitask_model_latest.pth) + skip arms whose OOD
# result is already recorded REAL. Additive only; Dialga repo untouched.
set -o pipefail
source ~/.bashrc 2>/dev/null; conda activate river
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/mpl
export TORCH_HOME=/storage/project/r-agarg35-0/lwang831/torch_cache
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub HUGGINGFACE_HUB_CACHE=$HF_HOME/hub TRANSFORMERS_CACHE=$HF_HOME/hub HF_HUB_OFFLINE=1
mkdir -p /storage/scratch1/8/lwang831/imitation/logs
REPO=/storage/home/hcoda1/8/lwang831/workspace/imitation
DATA=$REPO/data                                   # 90-task processed hdf5 (symlinked)
CACHE=/storage/scratch1/8/lwang831/dialga_outputs/imitation/libero_bc_wan_cache_npy
ROOT=/storage/scratch1/8/lwang831/dialga_ood
RESULTS=$ROOT/ood_results.txt
cd $REPO
mkdir -p $ROOT

NEPOCHS=${NEPOCHS:-60}
EPISODES=${EPISODES:-10}          # per held-out task
PAR=${PAR:-10}                    # parallel envs for eval

# seed=42 split
TRAIN='[1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,23,24,25,27,28,29,30,31,32,33,34,35,36,37,38,41,42,43,45,46,47,48,49,50,51,52,53,54,56,57,58,59,60,61,62,63,64,66,67,68,69,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89]'
HELD='[0,10,22,26,39,40,44,55,65,70]'

echo "=== OOD CAMPAIGN $(date) host=$(hostname) | epochs=$NEPOCHS eval_ep=$EPISODES par=$PAR ===" | tee -a $RESULTS

run_arm () {
  local arm="$1"; local algo="$2"; local cache_arg="$3"; local enc_extra="$4"
  # skip if a REAL OOD result already recorded (preempt resume)
  if grep -qE "^RESULT $arm .*\[REAL\]" $RESULTS 2>/dev/null; then
    echo ">>> [$arm] already REAL — skipping" | tee -a $RESULTS; return; fi
  local out=$ROOT/$arm
  mkdir -p "$out"
  echo ">>> [$arm] TRAIN(resume) start $(date)" | tee -a $RESULTS
  python train.py --config-name=train \
    task=libero "task.task_subset=$TRAIN" \
    task.img_height=128 task.img_width=128 \
    algo=$algo algo.frame_stack=1 $cache_arg $enc_extra \
    data_prefix=$DATA \
    training.n_epochs=$NEPOCHS training.resume=true training.save_interval=5 \
    rollout.enabled=false logging.mode=disabled \
    exp_name=ood variant_name=$arm output_prefix=$out 2>&1 | tail -c 4000
  local ck=$(find "$out" -name "multitask_model_latest.pth" 2>/dev/null | head -1)
  echo ">>> [$arm] TRAIN done ckpt=$ck $(date)" | tee -a $RESULTS
  [ -z "$ck" ] && { echo "RESULT $arm TRAIN_FAILED" | tee -a $RESULTS; return; }
  # ---- OOD eval on the 10 held-out tasks (fixed eval) ----
  echo ">>> [$arm] OOD-EVAL start $(date)" | tee -a $RESULTS
  rm -rf $REPO/experiments/evaluate/libero/libero_90/ood_$arm 2>/dev/null
  find $REPO/experiments/evaluate -name progress.json -delete 2>/dev/null
  local t0=$(date +%s)
  python evaluate.py --config-name=evaluate \
    exp_name=ood_$arm \
    checkpoint_path="$ck" algo=$algo \
    task=libero "task.task_subset=$HELD" task.img_height=128 task.img_width=128 \
    data_prefix=$DATA \
    +overrides.encoder.frame_stack=33 +overrides.encoder.wan_cache_dir=null \
    rollout.rollouts_per_env=$EPISODES rollout.num_parallel_envs=$PAR rollout.n_video=0 \
    > $ROOT/eval_$arm.log 2>&1
  local dt=$(( $(date +%s) - t0 ))
  local sr=$(grep -oiE "success rate: [0-9.]+" $ROOT/eval_$arm.log | tail -1)
  # gate: 10 held tasks * ceil(EPISODES/PAR) batches * ~60s. Require >= held_tasks*30s.
  local gate=$((10*30)); local valid="REAL"; [ $dt -lt $gate ] && valid="FAKE(<${gate}s)"
  echo "RESULT $arm walltime=${dt}s [$valid] OOD_$sr" | tee -a $RESULTS
}

run_arm r3_zdyn    fm_policy_r3 "algo.encoder.wan_cache_dir=$CACHE"
run_arm r5_zstatic fm_policy_r5 "algo.encoder.wan_cache_dir=$CACHE"
run_arm r1_both    fm_policy_r1 "algo.encoder.wan_cache_dir=$CACHE"
run_arm r4_wanflat fm_policy_r4 "algo.encoder.wan_cache_dir=$CACHE"
run_arm r2_baseline fm_policy_r2 ""

# ---- no-RGB (DIALGA-feature-only) arms: isolate what each code carries ----
run_arm nrgb_zdyn_strict   fm_policy_r3 "algo.encoder.wan_cache_dir=$CACHE" "algo.encoder.use_rgb=false algo.encoder.lowdim_encoder_factory=null"
run_arm nrgb_proprio_only  fm_policy_r2 ""                                   "algo.encoder.use_rgb=false"
run_arm nrgb_zdyn_proprio  fm_policy_r3 "algo.encoder.wan_cache_dir=$CACHE" "algo.encoder.use_rgb=false"
run_arm nrgb_wanflat_prop  fm_policy_r4 "algo.encoder.wan_cache_dir=$CACHE" "algo.encoder.use_rgb=false"
run_arm nrgb_zstatic_strict fm_policy_r5 "algo.encoder.wan_cache_dir=$CACHE" "algo.encoder.use_rgb=false algo.encoder.lowdim_encoder_factory=null"

echo "=== OOD CAMPAIGN DONE $(date) ===" | tee -a $RESULTS
grep -E "^RESULT " $RESULTS | tee -a $RESULTS
