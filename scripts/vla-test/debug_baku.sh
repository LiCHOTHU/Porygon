python train.py --config-name=train_debug.yaml \
    task=libero_10_hybrid \
    algo=dit_head_2 \
    algo.chunk_size=15 \
    algo.policy.hidden_dim=256 \
    algo.policy.num_layers=4 \
    training.save_interval=10 \
    train_dataloader.num_workers=4 \
    rollout.interval=10 \
    task.demos_per_env=50 \
    training.n_epochs=101



python export_videos.py \
    task=libero_10_hybrid \
    algo=baku \
    checkpoint_path=experiments/libero/libero_10/debug/run_021 \
    rollout.max_episode_length=100 \
    algo.abs_action=false


python train.py \
    --config-name=train_debug.yaml \
    task=libero_10_hybrid \
    algo=baku \
    algo/encoder=hybrid  \
    algo/aug=image \
    algo.chunk_size=10 \
    +algo.policy.eecf=true \
    algo.encoder.do_hand_crop=true \
    +algo.encoder.tight_crop=true \
    training.save_interval=20 \
    train_dataloader.num_workers=4 \
    rollout.interval=200 \
    task.demos_per_env=50 \
    training.n_epochs=101

python train.py \
    --config-name=train_debug.yaml \
    task=libero_10_hybrid \
    algo=baku \
    algo/encoder=hybrid  \
    algo/aug=image \
    algo.chunk_size=10 \
    algo.encoder.do_hand_crop=true \
    algo.abs_action=false \
    algo.eecf=false \
    +algo.encoder.tight_crop=true \
    training.save_interval=20 \
    train_dataloader.num_workers=4 \
    rollout.interval=200 \
    task.demos_per_env=50 \
    training.n_epochs=101
