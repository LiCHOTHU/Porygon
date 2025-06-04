#!/bin/bash
eval_exp_name=mimicgen_6_4
export HYDRA_FULL_ERROR=1
seeds=(0 1 2)
# seeds=(1)
prefix=/storage/home/hcoda1/1/awilcox31/vast/imitation/experiments/mimicgen
exp_names=(
    # rand_sweep
    # rand_sweep_no_wrist
    # sweep_no_wrist
    mimicgen_oops
)
task_names=(
    # "square_d1"
    # "stack_d1"
    "threading_d1"
    "coffee_d1"
)
variant_names=(
    # "rgb"
    # "rgbd"
    # "dp3"
    # "diffusion_policy_adapt3r"
    # "diffusion_policy_rgb"
    adapt3r_ft
    rgb
)

cam_shifts=(0.2 0.4 0.6 0.8 1.0)
robots=("UR5e" "Kinova3" "IIWA")

for seed in ${seeds[@]}; do
    for exp_name in ${exp_names[@]}; do
        for task_name in ${task_names[@]}; do
            for variant_name in ${variant_names[@]}; do

                # original
                sbatch slurm/run_rtx6000.sbatch uv run evaluate.py \
                    exp_name=${eval_exp_name} \
                    variant_name=${exp_name}_${variant_name} \
                    task=mimicgen \
                    task.task_name=${task_name} \
                    task.horizon=350 \
                    +overrides.temporal_agg=false +overrides.action_horizon=2 \
                    seed=${seed} \
                    rollout.rollouts_per_env=100 \
                    checkpoint_path=${prefix}/${task_name}/${exp_name}/${variant_name}/${seed}/multitask_model_latest.pth
                
                # for cam_shift in ${cam_shifts[@]}; do
                #     # adapt3r cam_shift
                #     sbatch slurm/run_rtx6000.sbatch python evaluate.py \
                #         exp_name=${eval_exp_name} \
                #         variant_name=cam_shift_${cam_shift}_${variant_name} \
                #         task=mimicgen \
                #         task.task_name=${task_name} \
                #         task.horizon=350 \
                #         algo=${algo} \
                #         +overrides.temporal_agg=false +overrides.action_horizon=2 \
                #         task.cam_shift=${cam_shift} \
                #         seed=${seed} \
                #         rollout.rollouts_per_env=100 \
                #         checkpoint_path=${prefix}/${task_name}/${algo}/${exp_name}/${variant_name}/${seed}/multitask_model_epoch_0250.pth
                # done

                # for robot in ${robots[@]}; do
                #     # adapt3r cam_shift
                #     sbatch slurm/run_rtx6000.sbatch python evaluate.py \
                #         exp_name=${eval_exp_name} \
                #         variant_name=${robot}_${variant_name} \
                #         task=mimicgen \
                #         task.task_name=${task_name} \
                #         task.horizon=350 \
                #         algo=${algo} \
                #         +overrides.temporal_agg=false +overrides.action_horizon=2 \
                #         task.robot=${robot} \
                #         seed=${seed} \
                #         rollout.rollouts_per_env=100 \
                #         checkpoint_path=${prefix}/${task_name}/${algo}/${exp_name}/${variant_name}/${seed}/multitask_model_epoch_0250.pth
                # done
            done
        done
    done 
done
