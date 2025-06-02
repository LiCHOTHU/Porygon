task="libero_90_hybrid"
python export_videos.py \
    task=libero_90_hybrid \
    algo=dit_head_2 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/vast/quest_v0/experiments/libero/libero_90/dit_head_2/head_sweep/dit_head_2_hd_256_nl_4/stage_1/multitask_model_latest.pth


# prefix="/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/libero/libero_90"
prefix="/storage/home/hcoda1/1/awilcox31/vast/quest_v0/experiments/libero/libero_10"
eval_exp_name="adapt3r_head_testing"
seeds=(0)
algo="adapt3r_head"
exp_name="adapt3r_head_testing"
changes=("small" "medium" "large")
robots=("UR5e" "Kinova3" "IIWA")
variant_names=(        
    "dim_240_sa_2_xa_4" 
    "dim_240_sa_4_xa_4" 
    "dim_480_sa_2_xa_4" 
    "dim_480_sa_4_xa_4" 
    )

for variant_name in ${variant_names[@]}; do
    for seed in ${seeds[@]}; do
        echo python evaluate.py \
            exp_name=${eval_exp_name} \
            variant_name=mt_${variant_name} \
            task=${task} \
            algo=${algo} \
            rollout.rollouts_per_env=10 \
            rollout.num_parallel_envs=2 \
            checkpoint_path=${prefix}/${algo}/${exp_name}/${variant_name}/stage_1/multitask_model_epoch_0100.pth \
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
                rollout.rollouts_per_env=10 \
                rollout.num_parallel_envs=2 \
                checkpoint_path=${prefix}/${algo}/${exp_name}/${variant_name}/stage_1/multitask_model_epoch_0100.pth \
                seed=${seed}
        done
    done
done

# for robot in ${robots[@]}; do
#     for variant_name in ${variant_names[@]}; do
#         for seed in ${seeds[@]}; do
#             echo python evaluate.py \
#                 exp_name=${eval_exp_name} \
#                 variant_name=${robot}_${variant_name} \
#                 task=${task} \
#                 algo=${algo} \
#                 task.robot=${robot} \
#                 rollout.rollouts_per_env=10 \
#                 rollout.num_parallel_envs=2 \
#                 checkpoint_path=${prefix}/${algo}/${exp_name}/${variant_name}/stage_1/multitask_model_epoch_0100.pth \
#                 seed=${seed}
#         done
#     done
# done
