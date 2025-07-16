export HYDRA_FULL_ERROR=1

uv run train.py \
    --config-name=train_debug.yaml \
    task=mimicgen \
    algo=baku \
    algo.chunk_size=8 \
    algo.temporal_agg=false \
    algo.action_horizon=4 \
    $@

