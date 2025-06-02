#!/bin/bash
exp_name=mimicgen_corl_deterministic_old_crop
export HYDRA_FULL_ERROR=1
seeds=(1)
prefix=/storage/home/hcoda1/7/mghanem8/p-agarg35-0/codes/quest_v0/experiments
task_names=("coffee")
n_epochs=0050

for seed in ${seeds[@]}; do
    for task_name in ${task_names[@]}; do
        # adapt3r
        sbatch slurm/run_rtx6000.sbatch python evaluate.py \
            exp_name=${exp_name} \
            variant_name=adapt3r \
            task=mimicgen_hybrid_base \
            task.task_name=${task_name} \
            algo=diffusion_policy \
            algo/encoder=hybrid \
            algo.chunk_size=8 \
            +overrides.temporal_agg=false +overrides.action_horizon=2 \
            seed=${seed} \
            checkpoint_path=${prefix}/mimicgen/${task_name}/diffusion_policy/mimicgen_corl_new_hand_crop/adapt3r/${seed}/stage_1/multitask_model_epoch_${n_epochs}.pth

    #     # RGB
    #     sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    #         exp_name=${exp_name} \
    #         variant_name=rgb \
    #         task=mimicgen_rgb_base \
    #         task.task_name=${task_name} \
    #         algo=diffusion_policy \
    #         algo.chunk_size=16 \
    #         +overrides.temporal_agg=false +overrides.action_horizon=2 \
    #         seed=${seed} \
    #         checkpoint_path=${prefix}/mimicgen/${task_name}/diffusion_policy/mimicgen_corl/rgb/${seed}/stage_1/multitask_model_epoch_${n_epochs}.pth

    #     # RGBD
    #     sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    #         exp_name=${exp_name} \
    #         variant_name=rgbd \
    #         task=mimicgen_rgbd_base \
    #         task.task_name=${task_name} \
    #         algo=diffusion_policy \
    #         algo.chunk_size=16 \
    #         seed=${seed} \
    #         +overrides.temporal_agg=false +overrides.action_horizon=2 \
    #         checkpoint_path=${prefix}/mimicgen/${task_name}/diffusion_policy/mimicgen_corl/rgbd/${seed}/stage_1/multitask_model_epoch_${n_epochs}.pth

    #     # DP3
    #     sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    #         exp_name=${exp_name} \
    #         variant_name=dp3 \
    #         task=mimicgen_hybrid_base \
    #         task.task_name=${task_name} \
    #         algo=diffusion_policy \
    #         algo/encoder=hybrid_dp3 \
    #         algo.chunk_size=16 \
    #         seed=${seed} \
    #         +overrides.temporal_agg=false +overrides.action_horizon=2 \
    #         checkpoint_path=${prefix}/mimicgen/${task_name}/diffusion_policy/mimicgen_corl_new_hand_crop/dp3/${seed}/stage_1/multitask_model_epoch_${n_epochs}.pth
    done
done 