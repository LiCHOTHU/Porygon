#!/bin/bash
exp_name=mimicgen_corl_new_hand_crop
export HYDRA_FULL_ERROR=1
seeds=(0 1)
prefix=/storage/home/hcoda1/7/mghanem8/p-agarg35-0/codes/quest_v0/experiments
task_names=("square" "threading")
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
            seed=${seed} \
            checkpoint_path=${prefix}/mimicgen/${task_name}/diffusion_policy/${exp_name}/adapt3r/${seed}/stage_1/multitask_model_epoch_${n_epochs}.pth

    done
done 