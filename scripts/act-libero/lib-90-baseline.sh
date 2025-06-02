exp_name="corl-push"
seeds=(0 1 2 3 4)
algo="act"

for seed in ${seeds[@]}; do

    # RGB
    sbatch slurm/run_l40s.sbatch python train.py \
        --config-name=train_prior.yaml \
        pace_copy=true  \
        task=libero_90_rgb \
        exp_name=${exp_name} \
        variant_name=rgb \
        algo=${algo} \
        algo.chunk_size=15 \
        training.save_interval=20 \
        train_dataloader.num_workers=4 \
        rollout.interval=200 \
        task.demos_per_env=50 \
        training.n_epochs=101 \
        seed=${seed}

    # RGBD
    sbatch slurm/run_l40s.sbatch python train.py \
        --config-name=train_prior.yaml \
        pace_copy=true  \
        task=libero_90_rgbd \
        exp_name=${exp_name} \
        variant_name=rgbd \
        algo=${algo} \
        algo.chunk_size=15 \
        training.save_interval=20 \
        train_dataloader.num_workers=4 \
        rollout.interval=200 \
        task.demos_per_env=50 \
        training.n_epochs=101 \
        seed=${seed}

    # DP3
    sbatch slurm/run_l40s.sbatch python train.py \
        --config-name=train_prior.yaml \
        pace_copy=true  \
        task=libero_90_hybrid \
        exp_name=${exp_name} \
        variant_name=dp3 \
        algo=${algo} \
        algo/encoder=hybrid_dp3 \
        algo.chunk_size=15 \
        training.save_interval=20 \
        train_dataloader.num_workers=4 \
        rollout.interval=200 \
        task.demos_per_env=50 \
        training.n_epochs=101 \
        seed=${seed}

done


# python train.py \
#     --config-name=train_debug.yaml \
#     pace_copy=true  \
#     task=libero_90_hybrid \
#     algo=act \
#     algo/encoder=hybrid_dp3 \
#     algo.chunk_size=15 \
#     training.save_interval=20 \
#     train_dataloader.num_workers=4 \
#     rollout.interval=20 \
#     task.demos_per_env=50 \
#     training.n_epochs=101

