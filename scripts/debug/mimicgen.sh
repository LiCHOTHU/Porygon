export HYDRA_FULL_ERROR=1

uv run train.py \
    --config-name=train_debug.yaml \
    task=mimicgen \
    algo=diffusion_policy \
    algo.chunk_size=8 \
    $@

