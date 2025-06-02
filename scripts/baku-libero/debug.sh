exp_name="corl-push"
seeds=(0 1 2 3 4)

for seed in ${seeds[@]}; do

    # adapt3r baseline
    sbatch slurm/run_l40s.sbatch python train.py \
        --config-name=train_prior.yaml \
        pace_copy=true  \
        task=libero_90_hybrid \
        exp_name=${exp_name} \
        variant_name=adapt3r \
        algo=baku \
        algo/encoder=hybrid  \
        algo.chunk_size=10 \
        training.save_interval=20 \
        train_dataloader.num_workers=4 \
        rollout.interval=100 \
        task.demos_per_env=50 \
        training.n_epochs=101 \
        seed=${seed}

done


python train.py \
    --config-name=train_debug.yaml \
    task=libero_90_hybrid \
    algo=baku \
    algo/encoder=hybrid  \
    algo/aug=image \
    algo.chunk_size=10 \
    training.save_interval=20 \
    train_dataloader.num_workers=4 \
    rollout.interval=100 \
    task.demos_per_env=50 \
    training.n_epochs=101


python train.py \
    --config-name=train_debug.yaml \
    task=libero_90_rgb \
    algo=baku \
    algo.chunk_size=15 \
    training.save_interval=20 \
    train_dataloader.num_workers=4 \
    rollout.interval=100 \
    task.demos_per_env=50 \
    training.n_epochs=101


python train.py \
    --config-name=train_debug.yaml \
    task=libero_90_rgbd \
    algo=baku \
    algo.chunk_size=15 \
    training.save_interval=20 \
    train_dataloader.num_workers=4 \
    rollout.interval=100 \
    task.demos_per_env=50 \
    training.n_epochs=101


python train.py \
    --config-name=train_debug.yaml \
    task=libero_90_hybrid \
    algo=diffusion_policy \
    algo/encoder=hybrid_dp3 \
    algo.chunk_size=16 \
    training.save_interval=20 \
    train_dataloader.num_workers=4 \
    rollout.interval=100 \
    task.demos_per_env=50 \
    training.n_epochs=101 pace_copy=true
