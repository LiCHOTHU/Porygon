#!/bin/bash
export HYDRA_FULL_ERROR=1
nepochs=0050
exp_name=mimicgen_corl
task_names=("square")
seeds=(0 1)
prefix=/storage/home/hcoda1/7/mghanem8/p-agarg35-0/codes/quest_v0/experiments

for task_name in ${task_names[@]}; do
    for seed in ${seeds[@]}; do
        sbatch slurm/run_rtx6000.sbatch python evaluate.py \
            exp_name=${exp_name}_nepochs_${nepochs} \
            variant_name=adapt3r_no-hand-crop \
            task=mimicgen_hybrid_base \
            task.task_name=${task_name} \
            algo=diffusion_policy \
            algo/encoder=hybrid \
            algo.chunk_size=8 \
            +overrides.temporal_agg=false +overrides.action_horizon=2 \
            seed=${seed} \
            checkpoint_path=${prefix}/mimicgen/${task_name}/diffusion_policy/${exp_name}/adapt3r_no-hand-crop/${seed}/stage_1/multitask_model_epoch_${nepochs}.pth

        sbatch slurm/run_rtx6000.sbatch python evaluate.py \
            exp_name=${exp_name}_nepochs_${nepochs} \
            variant_name=adapt3r_hand-crop-new \
            task=mimicgen_hybrid_base \
            task.task_name=${task_name} \
            algo=diffusion_policy \
            algo/encoder=hybrid \
            algo.chunk_size=8 \
            +overrides.temporal_agg=false +overrides.action_horizon=2 \
            seed=${seed} \
            checkpoint_path=${prefix}/mimicgen/${task_name}/diffusion_policy/${exp_name}/adapt3r_hand-crop-new/${seed}/stage_1/multitask_model_epoch_${nepochs}.pth

        sbatch slurm/run_rtx6000.sbatch python evaluate.py \
            exp_name=${exp_name}_nepochs_${nepochs} \
            variant_name=adapt3r_hand-crop-old \
            task=mimicgen_hybrid_base \
            task.task_name=${task_name} \
            algo=diffusion_policy \
            algo/encoder=hybrid \
            algo.chunk_size=8 \
            +overrides.temporal_agg=false +overrides.action_horizon=2 \
            seed=${seed} \
            checkpoint_path=${prefix}/mimicgen/${task_name}/diffusion_policy/${exp_name}/adapt3r_hand-crop-old/${seed}/stage_1/multitask_model_epoch_${nepochs}.pth
    done
done 