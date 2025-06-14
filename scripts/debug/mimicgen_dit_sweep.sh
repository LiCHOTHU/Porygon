export HYDRA_FULL_ERROR=1

algo_names=(
    # fm_policy_S
    # dit_policy_S
    diffusion_policy
    fm_policy_M
    dit_policy_M
    # fm_policy_L
    # dit_policy_L 
)
task_names=(square_d1 coffee_d1 threading_d1)
# abs_actions=(true false)
seeds=(0 1 2)

for algo_name in ${algo_names[@]}; do
    for task_name in ${task_names[@]}; do
        # for abs_action in ${abs_actions[@]}; do
            for seed in ${seeds[@]}; do
                echo uv run train.py \
                    --config-name=train.yaml \
                    exp_name=mimicgen_dit_sweep_4 \
                    variant_name=${algo_name}_${task_name} \
                    task=mimicgen \
                    task.task_name=${task_name} \
                    algo=${algo_name} \
                    algo.chunk_size=16 \
                    algo.abs_action=false \
                    algo.policy.temporal_agg=false \
                    rollout.interval=25 \
                    training.n_epochs=501 \
                    pace_copy=true \
                    seed=${seed} \
                    $@
            # done
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
