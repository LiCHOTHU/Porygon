export HYDRA_FULL_ERROR=1

uv run train.py \
    --config-name=train_debug.yaml \
    task=real \
    algo=diffusion_policy \
    algo/encoder=otter_3d  \
    algo.chunk_size=8 \
    $@

