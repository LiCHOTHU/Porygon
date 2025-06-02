# prefix="/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/libero/libero_90"
prefix="/storage/home/hcoda1/1/awilcox31/vast/quest_v0/experiments/libero/libero_90"
eval_exp_name="head_sweep_2"
seeds=(0)
algos=(
    # "dit_head"
    "dit_head_2"
)
exp_name="head_sweep"
changes=("small" "medium" "large")
robots=("UR5e" "Kinova3" "IIWA")
variant_names=(        
    "dit_head_2_rot_aug_hd_256_nl_4"
    "dit_head_2_rot_aug_hd_256_nl_8"
    "dit_head_2_rot_aug_hd_512_nl_4"
    "dit_head_2_rot_aug_hd_512_nl_8"
    )
task="libero_90_hybrid"

for algo in ${algos[@]}; do
    for variant_name in ${variant_names[@]}; do
        for seed in ${seeds[@]}; do
            echo python evaluate.py \
                exp_name=${eval_exp_name} \
                variant_name=mt_${variant_name} \
                task=${task} \
                algo=${algo} \
                rollout.rollouts_per_env=4 \
                rollout.num_parallel_envs=2 \
                checkpoint_path=${prefix}/${algo}/${exp_name}/${variant_name}/stage_1/ \
                seed=${seed}
        done
    done

    for change in ${changes[@]}; do
        for variant_name in ${variant_names[@]}; do
            for seed in ${seeds[@]}; do
                echo python evaluate.py \
                    exp_name=${eval_exp_name} \
                    variant_name=${change}_${variant_name} \
                    task=${task} \
                    algo=${algo} \
                    +task.env_factory.camera_pose_variations=${change} \
                    rollout.rollouts_per_env=4 \
                    rollout.num_parallel_envs=2 \
                    checkpoint_path=${prefix}/${algo}/${exp_name}/${variant_name}/stage_1/ \
                    seed=${seed}
            done
        done
    done
done

for robot in ${robots[@]}; do
    for variant_name in ${variant_names[@]}; do
        for seed in ${seeds[@]}; do
            echo python evaluate.py \
                exp_name=${eval_exp_name} \
                variant_name=${robot}_${variant_name} \
                task=${task} \
                algo=${algo} \
                task.robot=${robot} \
                rollout.rollouts_per_env=4 \
                rollout.num_parallel_envs=2 \
                checkpoint_path=${prefix}/${algo}/${exp_name}/${variant_name}/stage_1/ \
                seed=${seed}
        done
    done
done
