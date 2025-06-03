export HYDRA_FULL_ERROR=1

uv run train.py \
    --config-name=train_debug.yaml \
    task=mimicgen_rand \
    task.task_name=stack_d1 \
    ~task.shape_meta.observation.rgb.robot0_eye_in_hand_image \
    ~task.shape_meta.observation.depth.robot0_eye_in_hand_depth \
    algo=diffusion_policy \
    algo/encoder=rgb \
    algo.chunk_size=8 \
    $@

