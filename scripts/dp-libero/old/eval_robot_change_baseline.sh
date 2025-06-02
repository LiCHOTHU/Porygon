prefix="/storage/coda1/p-agarg35/0/shared/upce_models/cvpr_final_ckpts"
algo="diffusion_policy"
eval_exp_name="robot_change_final"
exp_name="final"
robots=("UR5e" "Kinova3" "IIWA")
variant_names=(
    "dp3_no_joint_demos_50_block_16"
    "idp3_no_joint_demos_50_block_16"
    "rgb_no_joint_demos_50_block_16"
    "rgbd_no_joint_demos_50_block_16"
    )
tasks=(
    "libero_90_hybrid"
    "libero_90_hybrid"
    "libero_90_rgb"
    "libero_90_rgbd"
    )
seeds=(0 1)



for robot in ${robots[@]}; do
    for i in {0..3}; do
        variant_name=${variant_names[i]}
        task=${tasks[i]}
        for seed in ${seeds[@]}; do
            sbatch slurm/run_l40s.sbatch python evaluate.py \
                exp_name=${eval_exp_name} \
                variant_name=${robot}_${variant_name} \
                task=${task} \
                algo=${algo} \
                task.robot=${robot} \
                rollout.rollouts_per_env=10 \
                rollout.num_parallel_envs=2 \
                checkpoint_path=${prefix}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1/multitask_model_epoch_0100.pth \
                seed=${seed}
        done
    done
done






