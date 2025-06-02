exp_name="corl-push-2"
seeds=(0 1 2 3 4)
algo="diffusion_policy"

for seed in ${seeds[@]}; do

    # adapt3r baseline
    # echo python train.py \
    #     --config-name=train_prior.yaml \
    #     pace_copy=true  \
    #     task=libero_90_hybrid \
    #     exp_name=${exp_name} \
    #     variant_name=adapt3r \
    #     algo=${algo} \
    #     algo/encoder=hybrid  \
    #     algo.chunk_size=8 \
    #     training.save_interval=10 \
    #     train_dataloader.num_workers=4 \
    #     rollout.interval=50 \
    #     task.demos_per_env=50 \
    #     training.n_epochs=101 \
    #     seed=${seed}

    # RGB
    echo python train.py \
        --config-name=train_prior.yaml \
        pace_copy=true  \
        task=libero_90_rgb \
        exp_name=${exp_name} \
        variant_name=rgb \
        algo=${algo} \
        algo.chunk_size=16 \
        training.save_interval=10 \
        train_dataloader.num_workers=4 \
        rollout.interval=50 \
        task.demos_per_env=50 \
        training.n_epochs=101 \
        seed=${seed}

    # RGBD
    echo python train.py \
        --config-name=train_prior.yaml \
        pace_copy=true  \
        task=libero_90_rgbd \
        exp_name=${exp_name} \
        variant_name=rgbd \
        algo=${algo} \
        algo.chunk_size=16 \
        training.save_interval=10 \
        train_dataloader.num_workers=4 \
        rollout.interval=50 \
        task.demos_per_env=50 \
        training.n_epochs=101 \
        seed=${seed}

    # DP3
    echo python train.py \
        --config-name=train_prior.yaml \
        pace_copy=true  \
        task=libero_90_hybrid \
        exp_name=${exp_name} \
        variant_name=dp3 \
        algo=${algo} \
        algo/encoder=hybrid_dp3 \
        algo.chunk_size=16 \
        training.save_interval=10 \
        train_dataloader.num_workers=4 \
        rollout.interval=50 \
        task.demos_per_env=50 \
        training.n_epochs=101 \
        seed=${seed}

done


# python train.py \
#     --config-name=train_debug.yaml \
#     pace_copy=true  \
#     task=libero_90_hybrid \
#     algo=${algo} \
#     algo/encoder=hybrid_dp3 \
#     algo.chunk_size=15 \
#     training.save_interval=10 \
#     train_dataloader.num_workers=4 \
#     rollout.interval=50 \
#     task.demos_per_env=50 \
#     training.n_epoch5101
