#!/bin/bash
eval_exp_name=fixed_sweep_actions_space
exp_name=fixed_sweep_actions_space
export HYDRA_FULL_ERROR=1
seeds=(0 1)
algo=diffusion_policy
prefix=/storage/home/hcoda1/1/awilcox31/vast/imitation/experiments/mimicgen
task_names=(
    "square_d1"
    "threading_d1"
)
variant_names=(
    adapt3r_ft
    adapt3r_ft_abs
    adapt3r_ft_abs_eecf
    adapt3r_ft_eecf
    adapt3r_ft_no_hf
    adapt3r_ft_no_hf_abs
    dp3
    dp3_abs
    rgb
    rgb_abs
)
task=mimicgen

for seed in ${seeds[@]}; do
    for task_name in ${task_names[@]}; do
        for variant_name in ${variant_names[@]}; do

            # original
            echo uv run evaluate.py \
                exp_name=${eval_exp_name} \
                variant_name=mt_${variant_name} \
                task=${task} \
                task.task_name=${task_name} \
                task.horizon=350 \
                algo=${algo} \
                +overrides.temporal_agg=false +overrides.action_horizon=2 \
                seed=${seed} \
                rollout.rollouts_per_env=100 \
                checkpoint_path=${prefix}/${task_name}/${exp_name}/${variant_name}/${seed}/
            
        done
    done 
done

