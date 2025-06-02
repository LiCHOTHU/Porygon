#!/bin/bash
eval_exp_name=mimicgen_corl_interm_eval
exp_name=mimicgen_corl
export HYDRA_FULL_ERROR=1
seeds=(0)
algo=diffusion_policy
prefix=/storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready/mimicgen
output="/storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready"
task_names=(
    "square"
    "threading"
    "coffee"
)
variant_names=(
    "rgb"
    "rgbd"
    "dp3"
    "adapt3r_good"
)
tasks=(
    "mimicgen_rgb_base"
    "mimicgen_rgbd_base"
    "mimicgen_hybrid_base"
    "mimicgen_hybrid_base"
)
length=${#variant_names[@]}
cam_shifts=(0.2 0.4 0.8)
robots=("UR5e" "Kinova3" "IIWA")

for task_name in ${task_names[@]}; do
    for (( i = 0; i < $length; i++ )) ; do
        variant_name=${variant_names[i]}
        task=${tasks[i]}
        for seed in ${seeds[@]}; do
            # original
            echo python evaluate.py \
                exp_name=${eval_exp_name} \
                variant_name=mt_${variant_name} \
                task=${task} \
                task.task_name=${task_name} \
                algo=${algo} \
                +overrides.temporal_agg=false +overrides.action_horizon=2 \
                seed=${seed} \
                rollout.rollouts_per_env=100 \
                checkpoint_path=${prefix}/${task_name}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1/checkpoint.pth
            
            for cam_shift in ${cam_shifts[@]}; do
                # adapt3r cam_shift
                echo python evaluate.py \
                    exp_name=${eval_exp_name} \
                    variant_name=cam_shift_${cam_shift}_${variant_name} \
                    task=${task} \
                    task.task_name=${task_name} \
                    algo=${algo} \
                    +overrides.temporal_agg=false +overrides.action_horizon=2 \
                    task.cam_shift=${cam_shift} \
                    seed=${seed} \
                    rollout.rollouts_per_env=100 \
                    checkpoint_path=${prefix}/${task_name}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1/checkpoint.pth
            done

            for robot in ${robots[@]}; do
                # adapt3r cam_shift
                echo python evaluate.py \
                    exp_name=${eval_exp_name} \
                    variant_name=${robot}_${variant_name} \
                    task=${task} \
                    task.task_name=${task_name} \
                    algo=${algo} \
                    +overrides.temporal_agg=false +overrides.action_horizon=2 \
                    task.robot=${robot} \
                    seed=${seed} \
                    rollout.rollouts_per_env=100 \
                    checkpoint_path=${prefix}/${task_name}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1/checkpoint.pth
            done
        done
    done 
done

