export HYDRA_FULL_ERROR=1

uv run train.py \
    --config-name=train_debug.yaml \
    task=mimicgen_rand \
    task.task_name=stack_d1 \
    algo=diffusion_policy \
    algo/encoder=adapt3r \
    algo.chunk_size=8 \
    $@

