# prefix="/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/libero/libero_90"
prefix="/storage/home/hcoda1/1/awilcox31/vast/quest_v0/experiments/libero/libero_90"
eval_exp_name="head_sweep_3"
seeds=(0)
algo="dit_head"
exp_name="head_sweep_3"
changes=("small" "medium" "large")
robots=("UR5e" "Kinova3" "IIWA")
variant_names=(
    "baku_baseline"
    "baku_eecf"
    "da_style_head_no_eecf"
    "dit_head_1k"
    "dit_head_1k_no_act_eecf_no_abs"
    "dit_head_2_no_act_eecf"
    "dit_head_2_no_act_eecf_no_abs"
    "dit_head_2_no_eecf_no_abs"
    "dit_head_4k"
    "dit_head_4k_no_act_eecf_no_abs"
    "dit_head_16k"
    "dit_head_16k_no_act_eecf_no_abs"
    )
task="libero_90_hybrid"

for variant_name in ${variant_names[@]}; do
    for seed in ${seeds[@]}; do
        echo python evaluate.py \
            exp_name=${eval_exp_name} \
            variant_name=mt_${variant_name} \
            task=${task} \
            algo=${algo} \
            rollout.rollouts_per_env=4 \
            rollout.num_parallel_envs=2 \
            checkpoint_path=${prefix}/${exp_name}/${variant_name}/stage_1/ \
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
                checkpoint_path=${prefix}/${exp_name}/${variant_name}/stage_1/ \
                seed=${seed}
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
                checkpoint_path=${prefix}/${exp_name}/${variant_name}/stage_1/ \
                seed=${seed}
        done
    done
done


