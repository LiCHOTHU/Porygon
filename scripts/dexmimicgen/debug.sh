export HYDRA_FULL_ERROR=1

uv run train.py \
    --config-name=train_debug.yaml \
    task=dexmimicgen \
    algo=diffusion_policy \
    algo/encoder=rgb  \
    algo.chunk_size=8 \
    $@

