#!/bin/bash
eval_exp_name=mimicgen_corl_sweeps_2
exp_name=mimicgen_corl
export HYDRA_FULL_ERROR=1
seeds=(0)
prefix=/storage/home/hcoda1/1/awilcox31/vast/quest_v0/experiments/mimicgen
task_names=("threading")
variant_names=(
    # "adapt3r_beefy_cs_8"
    "adapt3r_beefy_cs_8_ft"
    # "adapt3r_beefy_cs_16"
    "adapt3r_beefy_hd_120"
    # "adapt3r_beefy_no_ds"
    # "adapt3r_beefy_no_ds_ft"
    # "adapt3r_beefy_no_ds_no_attn"
    # "adapt3r_beefy_no_eecf"
    "adapt3r_beefy_np_1024"
)
cam_shifts=(0.2 0.4)

for cam_shift in ${cam_shifts[@]}; do
    for variant_name in ${variant_names[@]}; do
        for seed in ${seeds[@]}; do
            for task_name in ${task_names[@]}; do
                # # adapt3r
                # sbatch slurm/run_rtx6000.sbatch python evaluate.py \
                #     exp_name=${eval_exp_name} \
                #     variant_name=${variant_name} \
                #     task=mimicgen_hybrid_base \
                #     task.task_name=${task_name} \
                #     algo=diffusion_policy \
                #     +overrides.temporal_agg=false +overrides.action_horizon=2 \
                #     seed=${seed} \
                #     checkpoint_path=${prefix}/${task_name}/diffusion_policy/${exp_name}/${variant_name}/${seed}/stage_1/

                # adapt3r cam_sift
                sbatch slurm/run_rtx6000.sbatch python evaluate.py \
                    exp_name=${eval_exp_name} \
                    variant_name=cam_shift_${cam_shift}_${variant_name} \
                    task=mimicgen_hybrid_base \
                    task.task_name=${task_name} \
                    algo=diffusion_policy \
                    +overrides.temporal_agg=false +overrides.action_horizon=2 \
                    task.cam_shift=${cam_shift} \
                    seed=${seed} \
                    checkpoint_path=${prefix}/${task_name}/diffusion_policy/${exp_name}/${variant_name}/${seed}/stage_1/
            done
        done 
    done
done

# python export_videos.py \
#     task=mimicgen_hybrid_base \
#     task.task_name=threading \
#     +overrides.temporal_agg=false +overrides.action_horizon=2 \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready/mimicgen/threading/diffusion_policy/mimicgen_corl_frame_fixed/adapt3r/0/stage_1/checkpoint.pth
