#!/bin/bash
export HYDRA_FULL_ERROR=1

nepochs=0100
exp_name=mimicgen_corl_adapt3r_mix
task_names=("threading")
num_points=(512 1024 2048)
seeds=(0)
prefix=/storage/home/hcoda1/7/mghanem8/p-agarg35-0/codes/quest_v0/experiments

for task_name in ${task_names[@]}; do
    for seed in ${seeds[@]}; do
        for num_point in ${num_points[@]}; do
            sbatch slurm/run_rtx6000.sbatch python export_videos.py \
                exp_name=${exp_name}_nepochs_${nepochs} \
                variant_name=adapt3r_${num_point}_finetune \
                task=mimicgen_hybrid_base \
                task.task_name=${task_name} \
                algo=diffusion_policy \
                algo/encoder=hybrid \
                algo.chunk_size=8 \
                +overrides.temporal_agg=false +overrides.action_horizon=2 \
                seed=${seed} \
                checkpoint_path=${prefix}/mimicgen/${task_name}/diffusion_policy/${exp_name}/adapt3r_${num_point}_finetune/${seed}/stage_1/multitask_model_epoch_${nepochs}.pth

            sbatch slurm/run_rtx6000.sbatch python export_videos.py \
                exp_name=${exp_name}_nepochs_${nepochs} \
                variant_name=adapt3r_${num_point}_no-finetune \
                task=mimicgen_hybrid_base \
                task.task_name=${task_name} \
                algo=diffusion_policy \
                algo/encoder=hybrid \
                algo.chunk_size=8 \
                +overrides.temporal_agg=false +overrides.action_horizon=2 \
                seed=${seed} \
                checkpoint_path=${prefix}/mimicgen/${task_name}/diffusion_policy/${exp_name}/adapt3r_${num_point}_no-finetune/${seed}/stage_1/multitask_model_epoch_${nepochs}.pth

            sbatch slurm/run_rtx6000.sbatch python export_videos.py \
                exp_name=${exp_name}_nepochs_${nepochs} \
                variant_name=adapt3r_${num_point}_finetune_no-hand-crop \
                task=mimicgen_hybrid_base \
                task.task_name=${task_name} \
                algo=diffusion_policy \
                algo/encoder=hybrid \
                algo.chunk_size=8 \
                +overrides.temporal_agg=false +overrides.action_horizon=2 \
                seed=${seed} \
                checkpoint_path=${prefix}/mimicgen/${task_name}/diffusion_policy/${exp_name}/adapt3r_${num_point}_finetune_no-hand-crop/${seed}/stage_1/multitask_model_epoch_${nepochs}.pth

            sbatch slurm/run_rtx6000.sbatch python export_videos.py \
                exp_name=${exp_name}_nepochs_${nepochs} \
                variant_name=adapt3r_${num_point}_no-finetune_no-hand-crop \
                task=mimicgen_hybrid_base \
                task.task_name=${task_name} \
                algo=diffusion_policy \
                algo/encoder=hybrid \
                algo.chunk_size=8 \
                +overrides.temporal_agg=false +overrides.action_horizon=2 \
                seed=${seed} \
                checkpoint_path=${prefix}/mimicgen/${task_name}/diffusion_policy/${exp_name}/adapt3r_${num_point}_no-finetune_no-hand-crop/${seed}/stage_1/multitask_model_epoch_${nepochs}.pth
        done
    done
done