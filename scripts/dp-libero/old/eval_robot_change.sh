prefix="/storage/home/hcoda1/1/awilcox31/shared/upce_models/cvpr_final_ckpts"
# prefix="/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/libero/libero_90"
eval_exp_name="robot_change_final"
seeds=(0 1)
algo="diffusion_policy"
exp_name="final"
variant_names=(    
    "tight_hand_crop_ds_512_clip_ft_false_block_8" 
    "tight_hand_crop_ds_512_resnet18_ft_false_block_8" 
    )
robots=("UR5e" "Kinova3" "IIWA")
# robots=("IIWA")
task="libero_90_hybrid"

for robot in ${robots[@]}; do
    for variant_name in ${variant_names[@]}; do
        for seed in ${seeds[@]}; do
            sbatch slurm/run_l40s.sbatch python evaluate.py \
                exp_name=${eval_exp_name} \
                variant_name=${robot}_${variant_name} \
                task=${task} \
                algo=${algo} \
                task.robot=${robot} \
                ~task.shape_meta.observation.lowdim \
                +task.shape_meta.observation.lowdim={robot0_gripper_qpos:2} \
                rollout.rollouts_per_env=10 \
                rollout.num_parallel_envs=2 \
                checkpoint_path=${prefix}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1/multitask_model_epoch_0100.pth \
                seed=${seed}
        done
    done
done

