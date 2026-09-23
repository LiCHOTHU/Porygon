#!/bin/bash
#SBATCH -J rm_learn
#SBATCH -N1 --gres=gpu:1 -c4 --mem=24G -t1:30:00
#SBATCH -o /storage/home/hcoda1/8/lwang831/workspace/imitation/icra_video/task_demos/raw/rm_%A.log
#SBATCH -e /storage/home/hcoda1/8/lwang831/workspace/imitation/icra_video/task_demos/raw/rm_%A.log
set -x
cd /storage/home/hcoda1/8/lwang831/workspace/dice_rl_official/dice-rl
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export DPPO_DATA_DIR=/storage/home/hcoda1/8/lwang831/workspace/dice_rl_official/dice-rl/data_dir
export DPPO_LOG_DIR=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/dice_rl_official/log_dir
PY=/storage/project/r-agarg35-0/lwang831/conda/envs/dice-rl/bin/python
S=/storage/home/hcoda1/8/lwang831/workspace/imitation/icra_video/scripts/render_robomimic_learning.py
O=/storage/home/hcoda1/8/lwang831/workspace/imitation/icra_video/task_demos/raw
$PY $S /storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/dice_rl_official/log_dir/robomimic-finetune/newbase_CASTv2_s42/model_step_2000.pth  $O/rm_square_early.mp4
$PY $S /storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/dice_rl_official/log_dir/robomimic-finetune/newbase_CASTv2_s42/model_step_16000.pth $O/rm_square_mid.mp4
$PY $S /storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/dice_rl_official/log_dir/robomimic-finetune/newbase_CASTv2_s42/model_step_31000.pth $O/rm_square_final.mp4
echo DONE; ls -la $O/*.mp4
