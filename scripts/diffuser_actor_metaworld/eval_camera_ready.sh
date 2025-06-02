# prefix="/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/libero/libero_90"
prefix="/storage/home/hcoda1/1/awilcox31/shared/upce_models/camera_ready/libero"
output="/storage/home/hcoda1/1/awilcox31/shared/upce_models/camera_ready"
seeds=(0 1 2 3 4)
algo="diffuser_actor"
changes=("small" "medium" "large")
robots=("UR5e" "Kinova3" "IIWA")
variant_names=(    
    "block_16" 
    )
task="libero_90_hybrid"

for variant_name in ${variant_names[@]}; do
    for seed in ${seeds[@]}; do
        sbatch slurm/run_l40s.sbatch python evaluate.py \
            exp_name=multitask \
            variant_name=${variant_name} \
            task=${task} \
            algo=${algo} \
            rollout.rollouts_per_env=10 \
            rollout.num_parallel_envs=2 \
            +overrides.action_horizon=12 \
            checkpoint_path=${prefix}/${algo}/${variant_name}/${seed}/stage_1/multitask_model_epoch_0100.pth \
            output_prefix=${output} \
            seed=${seed}
    done
done

for change in ${changes[@]}; do
    for variant_name in ${variant_names[@]}; do
        for seed in ${seeds[@]}; do
            sbatch slurm/run_l40s.sbatch python evaluate.py \
                exp_name=camera_change \
                variant_name=${change}_${variant_name} \
                task=${task} \
                algo=${algo} \
                +task.env_factory.camera_pose_variations=${change} \
                rollout.rollouts_per_env=10 \
                rollout.num_parallel_envs=2 \
                +overrides.action_horizon=12 \
                checkpoint_path=${prefix}/${algo}/${variant_name}/${seed}/stage_1/multitask_model_epoch_0100.pth \
                output_prefix=${output} \
                seed=${seed}
        done
    done
done

for robot in ${robots[@]}; do
    for variant_name in ${variant_names[@]}; do
        for seed in ${seeds[@]}; do
            sbatch slurm/run_l40s.sbatch python evaluate.py \
                exp_name=robot_change \
                variant_name=${robot}_${variant_name} \
                task=${task} \
                algo=${algo} \
                task.robot=${robot} \
                rollout.rollouts_per_env=10 \
                rollout.num_parallel_envs=2 \
                +overrides.action_horizon=12 \
                checkpoint_path=${prefix}/${algo}/${variant_name}/${seed}/stage_1/multitask_model_epoch_0100.pth \
                output_prefix=${output} \
                seed=${seed}
        done
    done
done

