export HYDRA_FULL_ERROR=1

# algo_names=(
#     # fm_policy_S
#     # dit_policy_S
#     diffusion_policy
#     fm_policy_M
#     dit_policy_M
#     # fm_policy_L
#     # dit_policy_L 
# )
algo_name=diffusion_policy
encoders=(
    adapt3r
    rgb
)
task_names=(
    square_d1 
    stack_d1
)
seeds=(0 1 2)

for encoder in ${encoders[@]}; do
    for task_name in ${task_names[@]}; do
        for seed in ${seeds[@]}; do
            echo uv run train.py \
                --config-name=train.yaml \
                exp_name=rand_sweep_no_wrist \
                variant_name=${algo_name}_${task_name}_${encoder} \
                task=mimicgen_rand \
                task.task_name=${task_name} \
                ~task.shape_meta.observation.rgb.robot0_eye_in_hand_image \
                ~task.shape_meta.observation.depth.robot0_eye_in_hand_depth \
                algo=${algo_name} \
                algo/encoder=${encoder} \
                algo.chunk_size=16 \
                algo.abs_action=false \
                algo.policy.temporal_agg=false \
                algo.encoder.finetune=true \
                rollout.interval=25 \
                training.n_epochs=501 \
                pace_copy=true \
                seed=${seed} \
                $@
        done
    done
done

# uv run train.py \
#     --config-name=train_debug.yaml \
#     task=mimicgen \
#     task.task_name=square_d1 \
#     algo=fm_policy_M \
#     algo/encoder=rgb \
#     algo.chunk_size=15 \
#     algo.abs_action=false \
#     algo.policy.temporal_agg=false \
#     rollout.interval=25 \
#     training.n_epochs=501 \
#     pace_copy=true \
#     seed=0
