prefix="/storage/coda1/p-agarg35/0/shared/upce_models2/libero/libero_90"
eval_exp_name="diffusion_policy_cam_change"
seeds=(0 1)
algo="diffusion_policy"
variant_names=(
    "25_demo_rgb_block_16"
    "25_demo_rgbd_block_16"
    )
tasks=(
    "libero_90_rgb"
    "libero_90_rgbd"
    )


for seed in ${seeds[@]}; do
    for i in {0..2}; do
        variant_name=${variant_names[i]}
        task=${tasks[i]}
        # echo $exp_name
        sbatch slurm/run_v100.sbatch python evaluate.py \
            exp_name=${eval_exp_name} \
            variant_name=${variant_name} \
            task=${task} \
            algo=${algo} \
            rollout.rollouts_per_env=10 \
            rollout.num_parallel_envs=5 \
            checkpoint_path=${prefix}/${algo}/${variant_name}/${seed}/stage_1 \
            +task.env_factory.camera_pose_variations=true \
            seed=${seed}
    done
done


