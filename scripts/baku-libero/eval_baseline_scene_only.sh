# prefix="/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/libero/libero_90"
prefix="/storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready/libero"
output="/storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready"
seeds=(0 1 2 3 4)
algo="baku"
changes=(0.52 1.05 2.1)
robots=("UR5e" "Kinova3" "IIWA")
variant_names=(
    "rgb_scene_only"
    "rgbd_scene_only"
    "dp3_scene_only"
)
tasks=(
    "libero_90_rgb_scene_cam"
    "libero_90_rgbd_scene_cam"
    "libero_90_hybrid_scene_cam"
)

for seed in ${seeds[@]}; do
    for i in {0..2}; do
        variant_name=${variant_names[i]}
        task=${tasks[i]}
        echo python evaluate.py \
            exp_name=multitask \
            variant_name=${variant_name} \
            task=${task} \
            algo=${algo} \
            rollout.rollouts_per_env=10 \
            rollout.num_parallel_envs=2 \
            checkpoint_path=${prefix}/${algo}/${variant_name}/${seed}/stage_1/ \
            output_prefix=${output} \
            seed=${seed}

        for change in ${changes[@]}; do
            echo python evaluate.py \
                exp_name=camera_change \
                variant_name=cs_${change}_rad_${variant_name} \
                task=${task} \
                algo=${algo} \
                task.cam_shift=${change} \
                rollout.rollouts_per_env=10 \
                rollout.num_parallel_envs=2 \
                checkpoint_path=${prefix}/${algo}/${variant_name}/${seed}/stage_1/ \
                output_prefix=${output} \
                seed=${seed}
        done

        for robot in ${robots[@]}; do
            echo python evaluate.py \
                exp_name=robot_change \
                variant_name=${robot}_${variant_name} \
                task=${task} \
                algo=${algo} \
                task.robot=${robot} \
                rollout.rollouts_per_env=10 \
                rollout.num_parallel_envs=2 \
                checkpoint_path=${prefix}/${algo}/${variant_name}/${seed}/stage_1/ \
                output_prefix=${output} \
                seed=${seed}
        done
    done
done


