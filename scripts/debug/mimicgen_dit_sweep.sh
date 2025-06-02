export HYDRA_FULL_ERROR=1

algo_names=(
    dit_policy_L 
    fm_policy_L
    dit_policy_M
    fm_policy_M
    dit_policy_S
    fm_policy_S
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
                    exp_name=mimicgen_dit_sweep_2 \
                    variant_name=${algo_name}_${task_name} \
                    task=mimicgen \
                    task.task_name=${task_name} \
                    algo=${algo_name} \
                    algo/encoder=rgb  \
                    algo.chunk_size=15 \
                    algo.abs_action=false \
                    algo.policy.temporal_agg=false \
                    rollout.interval=25 \
                    training.n_epochs=1001 \
                    pace_copy=true \
                    seed=${seed} \
                    $@
            # done
        done
    done
done
