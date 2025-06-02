sbatch slurm/run_l40s.sbatch python train.py \
    --config-name=train_prior.yaml \
    task=libero_10_hybrid \
    exp_name=dit_head_fix \
    variant_name=hd_256_nl_8 \
    pace_copy=true \
    algo=dit_head \
    algo.chunk_size=15 \
    algo.policy.hidden_dim=256 \
    algo.policy.num_layers=8 \
    training.save_interval=10 \
    train_dataloader.num_workers=4 \
    rollout.interval=10 \
    task.demos_per_env=50 \
    training.n_epochs=101

sbatch slurm/run_l40s.sbatch python train.py \
    --config-name=train_prior.yaml \
    task=libero_10_hybrid \
    exp_name=dit_head_fix \
    variant_name=hd_512_nl_8 \
    pace_copy=true \
    algo=dit_head \
    algo.chunk_size=15 \
    algo.policy.hidden_dim=512 \
    algo.policy.num_layers=8 \
    training.save_interval=10 \
    train_dataloader.num_workers=4 \
    rollout.interval=10 \
    task.demos_per_env=50 \
    training.n_epochs=101

sbatch slurm/run_l40s.sbatch python train.py \
    --config-name=train_prior.yaml \
    task=libero_10_hybrid \
    exp_name=dit_head_fix \
    variant_name=hd_256_nl_4 \
    pace_copy=true \
    algo=dit_head \
    algo.chunk_size=15 \
    algo.policy.hidden_dim=256 \
    algo.policy.num_layers=4 \
    training.save_interval=10 \
    train_dataloader.num_workers=4 \
    rollout.interval=10 \
    task.demos_per_env=50 \
    training.n_epochs=101

sbatch slurm/run_l40s.sbatch python train.py \
    --config-name=train_prior.yaml \
    task=libero_10_hybrid \
    exp_name=dit_head_fix \
    variant_name=hd_512_nl_4 \
    pace_copy=true \
    algo=dit_head \
    algo.chunk_size=15 \
    algo.policy.hidden_dim=512 \
    algo.policy.num_layers=4 \
    training.save_interval=10 \
    train_dataloader.num_workers=4 \
    rollout.interval=10 \
    task.demos_per_env=50 \
    training.n_epochs=101

