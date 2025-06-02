python train.py \
    --config-name=train_debug.yaml \
    pace_copy=true  \
    task=libero_90_hybrid \
    algo=baku \
    algo/encoder=hybrid_dp3 \
    algo.chunk_size=15 \
    training.save_interval=20 \
    train_dataloader.num_workers=4 \
    rollout.interval=200 \
    task.demos_per_env=50 \
    training.n_epochs=101

python train.py \
    --config-name=train_debug.yaml \
    pace_copy=true  \
    task=libero_90_rgbd \
    algo=baku \
    algo.chunk_size=15 \
    training.save_interval=20 \
    train_dataloader.num_workers=4 \
    rollout.interval=200 \
    task.demos_per_env=50 \
    training.n_epochs=101
 

python train.py \
    --config-name=train_debug.yaml \
    pace_copy=true  \
    task=libero_90_hybrid \
    algo=diffusion_policy \
    algo/encoder=hybrid_dp3 \
    algo.chunk_size=16 \
    training.save_interval=20 \
    train_dataloader.num_workers=4 \
    rollout.interval=200 \
    task.demos_per_env=50 \
    training.n_epochs=101