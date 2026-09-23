#!/bin/bash
#SBATCH -J render_demos
#SBATCH -N1 --gres=gpu:1 -c6 --mem=32G -t2:00:00
#SBATCH -o /storage/home/hcoda1/8/lwang831/workspace/imitation/icra_video/task_demos/render_%A.log
#SBATCH -e /storage/home/hcoda1/8/lwang831/workspace/imitation/icra_video/task_demos/render_%A.log
set -x
cd /storage/home/hcoda1/8/lwang831/workspace/imitation
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export ICRA_LIBERO_ASSETS=$(ls -d /storage/project/r-agarg35-0/lwang831/conda/envs/porygon/lib/python3.10/site-packages/libero/libero/assets 2>/dev/null)
CONDA=/storage/project/r-agarg35-0/lwang831/conda
echo "=== MANIP (porygon) ==="
$CONDA/envs/porygon/bin/python icra_video/scripts/render_manip.py
echo "=== DMC (dice-rl) ==="
$CONDA/envs/dice-rl/bin/python icra_video/scripts/render_dmc.py
echo "=== DONE ==="; ls -la icra_video/task_demos/*.mp4
