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
    three_piece_assembly_d1
    threading_d1
)
seeds=(0 1)

for seed in ${seeds[@]}; do
    for task_name in ${task_names[@]}; do
        for encoder in ${encoders[@]}; do
            echo uv run train.py \
                --config-name=train.yaml \
                exp_name=sweep_actions_space \
                variant_name=${encoder} \
                task=mimicgen \
                task.task_name=${task_name} \
                algo=${algo_name} \
                algo/encoder=${encoder} \
                algo.chunk_size=16 \
                algo.abs_action=false \
                algo.eecf=false \
                algo.policy.temporal_agg=false \
                rollout.interval=25 \
                training.n_epochs=251 \
                pace_copy=true \
                seed=${seed} \
                $@

            echo uv run train.py \
                --config-name=train.yaml \
                exp_name=sweep_actions_space \
                variant_name=${encoder}_eecf \
                task=mimicgen \
                task.task_name=${task_name} \
                algo=${algo_name} \
                algo/encoder=${encoder} \
                algo.chunk_size=16 \
                algo.abs_action=false \
                algo.eecf=true \
                algo.policy.temporal_agg=false \
                rollout.interval=25 \
                training.n_epochs=251 \
                pace_copy=true \
                seed=${seed} \
                $@

            echo uv run train.py \
                --config-name=train.yaml \
                exp_name=sweep_actions_space \
                variant_name=${encoder}_abs \
                task=mimicgen \
                task.task_name=${task_name} \
                algo=${algo_name} \
                algo/encoder=${encoder} \
                algo.chunk_size=16 \
                algo.abs_action=true \
                algo.eecf=false \
                algo.policy.temporal_agg=false \
                rollout.interval=25 \
                training.n_epochs=251 \
                pace_copy=true \
                seed=${seed} \
                $@

            echo uv run train.py \
                --config-name=train.yaml \
                exp_name=sweep_actions_space \
                variant_name=${encoder}_abs_eecf \
                task=mimicgen \
                task.task_name=${task_name} \
                algo=${algo_name} \
                algo/encoder=${encoder} \
                algo.chunk_size=16 \
                algo.abs_action=true \
                algo.eecf=true \
                algo.policy.temporal_agg=false \
                rollout.interval=25 \
                training.n_epochs=251 \
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
