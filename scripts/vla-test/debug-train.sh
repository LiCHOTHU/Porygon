python train.py \
    --config-name=train_debug.yaml \
    task=libero_90_hybrid \
    pace_copy=true \
    algo=adapt3r_head \
    algo.chunk_size=15 \
    training.save_interval=10 \
    rollout.interval=10 \
    task.demos_per_env=50 \
    training.n_epochs=101

python train.py \
    --config-name=train_debug.yaml \
    task=libero_10_hybrid \
    algo=adapt3r_head \
    algo.chunk_size=15 \
    training.save_interval=10 \
    rollout.interval=10 \
    task.demos_per_env=50 \
    training.n_epochs=101