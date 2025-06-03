#!/bin/bash
eval_exp_name=videos
exp_name=mimicgen_corl_final
export HYDRA_FULL_ERROR=1
seeds=(0)
algo=diffuser_actor
prefix=/home/albert/quest/data/checkpoints/mimicgen_final
task_names=(
    "square"
    "threading"
    "coffee"
)
variant_names=(
    "3dda"
)
tasks=(
    "mimicgen_hybrid_base"
)
length=${#variant_names[@]}

cam_shifts=(0.4)
robots=(
    "UR5e" 
    # "Kinova3" 
    # "IIWA"
)

for seed in ${seeds[@]}; do
    for task_name in ${task_names[@]}; do
        for (( i = 0; i < $length; i++ )) ; do
            variant_name=${variant_names[i]}
            task=${tasks[i]}


            # original
            echo python export_videos.py \
                exp_name=${eval_exp_name} \
                variant_name=mt_${variant_name} \
                make_unique_experiment_dir=false \
                task=${task} \
                task.task_name=${task_name} \
                task.horizon=350 \
                algo=${algo} \
                +task.env_factory.hd_rendering=true \
                seed=${seed} \
                rollout.rollouts_per_env=20 \
                checkpoint_path=${prefix}/${task_name}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1/multitask_model_epoch_0250.pth
            
            for cam_shift in ${cam_shifts[@]}; do
                # adapt3r cam_shift
                echo python export_videos.py \
                    exp_name=${eval_exp_name} \
                    variant_name=cam_shift_${cam_shift}_${variant_name} \
                    make_unique_experiment_dir=false \
                    task=${task} \
                    task.task_name=${task_name} \
                    task.horizon=350 \
                    algo=${algo} \
                    +task.env_factory.hd_rendering=true \
                    task.cam_shift=${cam_shift} \
                    seed=${seed} \
                    rollout.rollouts_per_env=20 \
                    checkpoint_path=${prefix}/${task_name}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1/multitask_model_epoch_0250.pth
            done

            for robot in ${robots[@]}; do
                # adapt3r cam_shift
                echo python export_videos.py \
                    exp_name=${eval_exp_name} \
                    variant_name=${robot}_${variant_name} \
                    make_unique_experiment_dir=false \
                    task=${task} \
                    task.task_name=${task_name} \
                    task.horizon=350 \
                    algo=${algo} \
                    +task.env_factory.hd_rendering=true \
                    task.robot=${robot} \
                    seed=${seed} \
                    rollout.rollouts_per_env=20 \
                    checkpoint_path=${prefix}/${task_name}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1/multitask_model_epoch_0250.pth
            done
        done
    done 
done

