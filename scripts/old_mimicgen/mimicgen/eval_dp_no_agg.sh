#!/bin/bash
exp_name=mimicgen_corl
export HYDRA_FULL_ERROR=1
seeds=(0 1)
action_horizons=(1 2 4)
prefix=/storage/home/hcoda1/7/mghanem8/p-agarg35-0/codes/quest_v0/experiments
task_names=("coffee" "square" "threading")
n_epochs=0050

for seed in ${seeds[@]}; do
    for task_name in ${task_names[@]}; do
        for action_horizon in ${action_horizons[@]}; do
            # adapt3r
            sbatch slurm/run_rtx6000.sbatch python evaluate.py \
                exp_name=${exp_name}_no_agg_${action_horizon} \
                variant_name=adapt3r \
                task=mimicgen_hybrid_base \
                task.task_name=${task_name} \
                algo=diffusion_policy \
                algo/encoder=hybrid \
                seed=${seed} \
                +overrides.temporal_agg=false +overrides.action_horizon=${action_horizon} \
                checkpoint_path=${prefix}/mimicgen/${task_name}/diffusion_policy/${exp_name}_new_hand_crop/adapt3r/${seed}/stage_1/multitask_model_epoch_${n_epochs}.pth

            # RGB
            # sbatch slurm/run_rtx6000.sbatch python evaluate.py \
            #     exp_name=${exp_name}_no_agg_${action_horizon} \
            #     variant_name=rgb \
            #     task=mimicgen_rgb_base \
            #     task.task_name=${task_name} \
            #     algo=diffusion_policy \
            #     seed=${seed} \
            #     +overrides.temporal_agg=false +overrides.action_horizon=${action_horizon} \
            #     checkpoint_path=${prefix}/mimicgen/${task_name}/diffusion_policy/${exp_name}_new_hand_crop/rgb/${seed}/stage_1/multitask_model_epoch_${n_epochs}.pth
        done
    done 
done 