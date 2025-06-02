# prefix="/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/libero/libero_90"
prefix="/storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready/libero"
output="/storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready"
seeds=(0 1 2 3 4)
algo="baku"
changes=(0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0)
variant_names=(
    "adapt3r_scene_only"
    "rgb_scene_only"
    "rgbd_scene_only"
    "dp3_scene_only"
)
tasks=(
    "libero_90_hybrid_scene_cam"
    "libero_90_rgb_scene_cam"
    "libero_90_rgbd_scene_cam"
    "libero_90_hybrid_scene_cam"
)

for seed in ${seeds[@]}; do
    for i in {0..3}; do
        variant_name=${variant_names[i]}
        task=${tasks[i]}

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
    done
done


