# prefix="/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/libero/libero_90"
prefix="/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/libero/libero_90"
eval_exp_name="round_2"
seeds=(0 1)
algo="diffuser_actor"
exp_name="final"
changes=("small" "medium" "large")
robots=("UR5e" "Kinova3" "IIWA")
# robots=("UR5e")
# robots=("Kinova3" "IIWA")
variant_names=(    
    # "block_4" 
    # "block_8" 
    "block_16" 
    # "no_feats" 
    # "no_feats_yes_rgb" 
    # "no_hand_crop" 
    # "no_nerf_pos_emb" 
    # "pos_based_fps" 
    )
task="libero_90_hybrid"

for variant_name in ${variant_names[@]}; do
    for seed in ${seeds[@]}; do
        sbatch slurm/run_l40s.sbatch python evaluate.py \
            exp_name=${eval_exp_name} \
            variant_name=mt_${variant_name} \
            task=${task} \
            algo=${algo} \
            rollout.rollouts_per_env=10 \
            rollout.num_parallel_envs=2 \
            checkpoint_path=${prefix}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1/multitask_model_epoch_0100.pth \
            seed=${seed}
    done
done

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

for robot in ${robots[@]}; do
    for variant_name in ${variant_names[@]}; do
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

