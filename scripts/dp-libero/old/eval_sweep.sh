prefix="/storage/home/hcoda1/1/awilcox31/shared/upce_models/cvpr_final_ckpts"
eval_exp_name="final_ckpt_sweep"
seeds=(0 1)
algo="diffusion_policy"
exp_name="final"
task="libero_90_hybrid"
variant_names=(    
    "tight_hand_crop_ds_512_clip_ft_false_block_8" 
    "tight_hand_crop_ds_512_clip_ft_true_block_8" 
    "tight_hand_crop_ds_512_resnet18_ft_false_block_8" 
    "tight_hand_crop_ds_512_resnet18_ft_true_block_8" 
    )
checkpoints=(
    # "multitask_model_epoch_0020.pth"
    # "multitask_model_epoch_0040.pth"
    # "multitask_model_epoch_0060.pth"
    "multitask_model_epoch_0080.pth"
    "multitask_model_epoch_0100.pth"
)

for variant_name in ${variant_names[@]}; do
    for checkpoint in ${checkpoints[@]}; do
        for seed in ${seeds[@]}; do
            sbatch slurm/run_l40s.sbatch python evaluate.py \
                exp_name=${eval_exp_name} \
                variant_name=${variant_name}_${checkpoint} \
                task=${task} \
                algo=${algo} \
                ~task.shape_meta.observation.lowdim \
                +task.shape_meta.observation.lowdim={robot0_gripper_qpos:2} \
                rollout.rollouts_per_env=10 \
                rollout.num_parallel_envs=2 \
                checkpoint_path=${prefix}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1/${checkpoint} \
                seed=${seed}
        done
    done
done

