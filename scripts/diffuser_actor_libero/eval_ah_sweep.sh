# prefix="/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/libero/libero_90"
prefix="/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/libero/libero_90"
eval_exp_name="horizon_sweep_2"
seeds=(0)
algo="diffuser_actor"
exp_name="final-3"
changes=("small")
# changes=("small" "medium" "large")
robots=("UR5e")
# robots=("UR5e" "Kinova3" "IIWA")
# robots=("UR5e")
# robots=("Kinova3" "IIWA")
variant_names=(    
    # "beefy_block_4" 
    # "beefy_block_8" 
    # "beefy_block_16" 
    # "block_4" 
    "block_8" 
    # "block_16" 
    # "relative_block_4" 
    # "relative_block_8" 
    # "beefy_relative_block_16" 
    # "no_feats" 
    # "no_feats_yes_rgb" 
    # "no_hand_crop" 
    # "no_nerf_pos_emb" 
    # "pos_based_fps" 
    )
# action_horizons=(2 4)
action_horizons=(2 4 8)
# action_horizons=(2 4 8)
task="libero_90_hybrid"

for action_horizon in ${action_horizons[@]}; do
    for variant_name in ${variant_names[@]}; do
        for seed in ${seeds[@]}; do
            sbatch slurm/run_l40s.sbatch python evaluate.py \
                exp_name=${eval_exp_name} \
                variant_name=ah_${action_horizon}_mt_${variant_name} \
                task=${task} \
                algo=${algo} \
                rollout.rollouts_per_env=10 \
                rollout.num_parallel_envs=2 \
                +overrides.action_horizon=${action_horizon} \
                checkpoint_path=${prefix}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1/multitask_model_epoch_0100.pth \
                seed=${seed}
        done
    done

    for change in ${changes[@]}; do
        for variant_name in ${variant_names[@]}; do
            for seed in ${seeds[@]}; do
                sbatch slurm/run_l40s.sbatch python evaluate.py \
                    exp_name=${eval_exp_name} \
                    variant_name=ah_${action_horizon}_${change}_${variant_name} \
                    task=${task} \
                    algo=${algo} \
                    +task.env_factory.camera_pose_variations=${change} \
                    rollout.rollouts_per_env=10 \
                    rollout.num_parallel_envs=2 \
                    +overrides.action_horizon=${action_horizon} \
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
                    variant_name=ah_${action_horizon}_${robot}_${variant_name} \
                    task=${task} \
                    algo=${algo} \
                    task.robot=${robot} \
                    rollout.rollouts_per_env=10 \
                    rollout.num_parallel_envs=2 \
                    +overrides.action_horizon=${action_horizon} \
                    checkpoint_path=${prefix}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1/multitask_model_epoch_0100.pth \
                    seed=${seed}
            done
        done
    done
done


