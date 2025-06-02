export HYDRA_FULL_ERROR=1

uv run train.py \
    --config-name=train_debug.yaml \
    task=mimicgen \
    algo=dit_policy_L \
    algo.chunk_size=15 \
    $@

