prefix="/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/libero/libero_90"
eval_exp_name="diffusion_policy_cam_change"
seeds=(0 1)
algo="diffusion_policy"
exp_names=(
    "diffusion_policy_no_crop_3"
    )
variant_names=(
    "flagship_block_16"
    )
tasks=(
    "libero_90_hybrid"
    )


for seed in ${seeds[@]}; do
    for i in {0..0}; do
        exp_name=${exp_names[i]}
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
            checkpoint_path=${prefix}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1 \
            +task.env_factory.camera_pose_variations=true \
            seed=${seed}
    done
done



