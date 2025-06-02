# prefix="/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/libero/libero_90"
prefix="/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/libero/libero_90"
eval_exp_name="round_4"
seeds=(0)
algo="diffuser_actor"
exp_name="final-3"
changes=("small" "medium" "large")
robots=("UR5e" "Kinova3" "IIWA")
# robots=("UR5e")
# robots=("Kinova3" "IIWA")
variant_names=(    
    # "beefy_block_4" 
    # "beefy_block_8" 
    # "beefy_block_16" 
    "relative_block_4" 
    "relative_block_8" 
    "relative_block_16" 
    # "beefy_relative_block_4" 
    # "beefy_relative_block_8" 
    # "beefy_relative_block_16" 
    # "block_4" 
    # "block_8" 
    # "block_16"
    # "faithful_block_4" 
    # "faithful_block_8" 
    # "faithful_block_16"
    )
task="libero_90_hybrid"

for variant_name in ${variant_names[@]}; do
    for seed in ${seeds[@]}; do
        sbatch slurm/run_rtx6000.sbatch python evaluate.py \
            exp_name=${eval_exp_name} \
            variant_name=mt_${variant_name} \
            task=${task} \
            algo=${algo} \
            rollout.rollouts_per_env=10 \
            rollout.num_parallel_envs=2 \
            +overrides.action_horizon=16 \
            checkpoint_path=${prefix}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1/ \
            seed=${seed}
    done
done


for change in ${changes[@]}; do
    for variant_name in ${variant_names[@]}; do
        for seed in ${seeds[@]}; do
            sbatch slurm/run_rtx6000.sbatch python evaluate.py \
                exp_name=${eval_exp_name} \
                variant_name=${change}_${variant_name} \
                task=${task} \
                algo=${algo} \
                +task.env_factory.camera_pose_variations=${change} \
                rollout.rollouts_per_env=10 \
                rollout.num_parallel_envs=2 \
                +overrides.action_horizon=16 \
                checkpoint_path=${prefix}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1/ \
                seed=${seed}
        done
    done
done

for robot in ${robots[@]}; do
    for variant_name in ${variant_names[@]}; do
        for seed in ${seeds[@]}; do
            sbatch slurm/run_rtx6000.sbatch python evaluate.py \
                exp_name=${eval_exp_name} \
                variant_name=${robot}_${variant_name} \
                task=${task} \
                algo=${algo} \
                task.robot=${robot} \
                rollout.rollouts_per_env=10 \
                rollout.num_parallel_envs=2 \
                +overrides.action_horizon=16 \
                checkpoint_path=${prefix}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1/ \
                seed=${seed}
        done
    done
done

