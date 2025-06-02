# prefix="/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/libero/libero_90"
prefix="/storage/home/hcoda1/1/awilcox31/shared/upce_models/cvpr_final_ckpts"
eval_exp_name="camera_change_final"
seeds=(0 1)
algo="diffusion_policy"
exp_name="final"
changes=("small" "medium" "large")
variant_names=(    
    "tight_hand_crop_ds_512_clip_ft_false_block_8" 
    "tight_hand_crop_ds_512_resnet18_ft_false_block_8" 
    )
task="libero_90_hybrid"

for change in ${changes[@]}; do
    for variant_name in ${variant_names[@]}; do
        for seed in ${seeds[@]}; do
            sbatch slurm/run_l40s.sbatch python evaluate.py \
                exp_name=${eval_exp_name} \
                variant_name=${change}_${variant_name} \
                task=${task} \
                algo=${algo} \
                +task.env_factory.camera_pose_variations=${change} \
                rollout.rollouts_per_env=10 \
                rollout.num_parallel_envs=2 \
                checkpoint_path=${prefix}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1/multitask_model_epoch_0100.pth \
                seed=${seed}
        done
    done
done