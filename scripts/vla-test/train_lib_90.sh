sbatch slurm/run_l40s.sbatch python train.py \
    --config-name=train_prior.yaml \
    task=libero_90_hybrid \
    exp_name=adapt3r_head_testing \
    variant_name=dim_240_sa_2_xa_4 \
    pace_copy=true \
    algo=adapt3r_head \
    algo.chunk_size=15 \
    algo.policy.embedding_dim=240 \
    +algo.policy.num_sa_layers=2 \
    +algo.policy.num_xa_layers=4 \
    training.save_interval=10 \
    train_dataloader.num_workers=4 \
    rollout.interval=10 \
    +task.env_factory.camera_pose_variations=small \
    task.demos_per_env=50 \
    training.n_epochs=101

sbatch slurm/run_l40s.sbatch python train.py \
    --config-name=train_prior.yaml \
    task=libero_90_hybrid \
    exp_name=adapt3r_head_testing \
    variant_name=dim_240_sa_4_xa_4 \
    pace_copy=true \
    algo=adapt3r_head \
    algo.chunk_size=15 \
    algo.policy.embedding_dim=240 \
    +algo.policy.num_sa_layers=4 \
    +algo.policy.num_xa_layers=4 \
    training.save_interval=10 \
    train_dataloader.num_workers=4 \
    rollout.interval=10 \
    +task.env_factory.camera_pose_variations=small \
    task.demos_per_env=50 \
    training.n_epochs=101

sbatch slurm/run_l40s.sbatch python train.py \
    --config-name=train_prior.yaml \
    task=libero_90_hybrid \
    exp_name=adapt3r_head_testing \
    variant_name=dim_480_sa_2_xa_4 \
    pace_copy=true \
    algo=adapt3r_head \
    algo.chunk_size=15 \
    algo.policy.embedding_dim=480 \
    +algo.policy.num_sa_layers=2 \
    +algo.policy.num_xa_layers=4 \
    training.save_interval=10 \
    train_dataloader.num_workers=4 \
    rollout.interval=10 \
    +task.env_factory.camera_pose_variations=small \
    task.demos_per_env=50 \
    training.n_epochs=101


sbatch slurm/run_l40s.sbatch python train.py \
    --config-name=train_prior.yaml \
    task=libero_90_hybrid \
    exp_name=adapt3r_head_testing \
    variant_name=dim_480_sa_4_xa_4 \
    pace_copy=true \
    algo=adapt3r_head \
    algo.chunk_size=15 \
    algo.policy.embedding_dim=480 \
    +algo.policy.num_sa_layers=4 \
    +algo.policy.num_xa_layers=4 \
    training.save_interval=10 \
    train_dataloader.num_workers=4 \
    rollout.interval=10 \
    +task.env_factory.camera_pose_variations=small \
    task.demos_per_env=50 \
    training.n_epochs=101