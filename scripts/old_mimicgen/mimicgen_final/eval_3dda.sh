#!/bin/bash
eval_exp_name=mimicgen_corl_final
exp_name=mimicgen_corl_final
export HYDRA_FULL_ERROR=1
seeds=(3 4)
algo=diffuser_actor
prefix=/storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready/mimicgen_final
output="/storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready"
task_names=(
    "square"
    "threading"
    "coffee"
)
variant_names=(
    # "rgb"
    # "rgbd"
    # "dp3"
    "3dda"
)
tasks=(
    # "mimicgen_rgb_base"
    # "mimicgen_rgbd_base"
    # "mimicgen_hybrid_base"
    "mimicgen_hybrid_base"
)
length=${#variant_names[@]}

cam_shifts=(0.2 0.4 0.6 0.8 1.0)
robots=("UR5e" "Kinova3" "IIWA")

for seed in ${seeds[@]}; do
    for task_name in ${task_names[@]}; do
        for (( i = 0; i < $length; i++ )) ; do
            variant_name=${variant_names[i]}
            task=${tasks[i]}


            # original
            echo python evaluate.py \
                exp_name=${eval_exp_name} \
                variant_name=mt_${variant_name} \
                task=${task} \
                task.task_name=${task_name} \
                task.horizon=350 \
                algo=${algo} \
                seed=${seed} \
                rollout.rollouts_per_env=100 \
                checkpoint_path=${prefix}/${task_name}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1/multitask_model_epoch_0250.pth
            
            for cam_shift in ${cam_shifts[@]}; do
                # adapt3r cam_shift
                echo python evaluate.py \
                    exp_name=${eval_exp_name} \
                    variant_name=cam_shift_${cam_shift}_${variant_name} \
                    task=${task} \
                    task.task_name=${task_name} \
                    task.horizon=350 \
                    algo=${algo} \
                    task.cam_shift=${cam_shift} \
                    seed=${seed} \
                    rollout.rollouts_per_env=100 \
                    checkpoint_path=${prefix}/${task_name}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1/multitask_model_epoch_0250.pth
            done

            for robot in ${robots[@]}; do
                # adapt3r cam_shift
                echo python evaluate.py \
                    exp_name=${eval_exp_name} \
                    variant_name=${robot}_${variant_name} \
                    task=${task} \
                    task.task_name=${task_name} \
                    task.horizon=350 \
                    algo=${algo} \
                    task.robot=${robot} \
                    seed=${seed} \
                    rollout.rollouts_per_env=100 \
                    checkpoint_path=${prefix}/${task_name}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1/multitask_model_epoch_0250.pth
            done
        done
    done 
done

