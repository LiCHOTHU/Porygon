
exp_name="sweep"
task="metaworld_mt50_hybrid"
blocks=(4 8 16)
# seeds=(0 1 2 3 4)
seeds=(0 1)

export HYDRA_FULL_ERROR=1

# algo.policy.temporal_agg=true \
# algo/aug=color_jitter \
# ~task.shape_meta.observation.lowdim \
# sbatch slurm/run_rtx6000.sbatch 

for block in ${blocks[@]}; do
    for seed in ${seeds[@]}; do
        echo python train.py \
            --config-name=train_prior.yaml \
            task=${task} \
            exp_name=${exp_name} \
            variant_name=block_${block} \
            algo=diffuser_actor \
            algo/encoder=hybrid  \
            algo.chunk_size=${block} \
            train_dataloader.num_workers=4 \
            training.n_epochs=100 \
            rollout.interval=20 \
            task.demos_per_env=10 \
            pace_copy=true  \
            seed=${seed}
        
        echo python train.py \
            --config-name=train_prior.yaml \
            task=${task} \
            exp_name=${exp_name} \
            variant_name=beefy_block_${block} \
            algo=diffuser_actor \
            algo/encoder=hybrid  \
            algo.chunk_size=${block} \
            +algo.policy.beefy=true \
            train_dataloader.num_workers=4 \
            training.n_epochs=100 \
            rollout.interval=20 \
            task.demos_per_env=10 \
            pace_copy=true  \
            seed=${seed}

        echo python train.py \
            --config-name=train_prior.yaml \
            task=${task} \
            exp_name=${exp_name} \
            variant_name=beefy_relative_block_${block} \
            algo=diffuser_actor \
            algo/encoder=hybrid  \
            algo.chunk_size=${block} \
            +algo.policy.beefy=true \
            algo.policy.relative=true \
            train_dataloader.num_workers=4 \
            training.n_epochs=100 \
            rollout.interval=20 \
            task.demos_per_env=10 \
            pace_copy=true  \
            seed=${seed}
        
        echo python train.py \
            --config-name=train_prior.yaml \
            task=${task} \
            exp_name=${exp_name} \
            variant_name=relative_block_${block} \
            algo=diffuser_actor \
            algo/encoder=hybrid  \
            algo.chunk_size=${block} \
            algo.policy.relative=true \
            train_dataloader.num_workers=4 \
            training.n_epochs=100 \
            rollout.interval=20 \
            task.demos_per_env=10 \
            pace_copy=true  \
            seed=${seed}

        # echo python train.py \
        #     --config-name=train_prior.yaml \
        #     task=${task} \
        #     exp_name=${exp_name} \
        #     variant_name=faithful_block_${block} \
        #     algo=diffuser_actor \
        #     algo/encoder=hybrid  \
        #     algo.chunk_size=${block} \
        #     +algo.policy.beefy=true \
        #     algo.policy.fps_subsampling_factor=3 \
        #     algo.policy.relative=true \
        #     algo.policy.diffusion_timesteps=25 \
        #     algo.lr=0.0003 \
        #     algo.weight_decay=0.0005 \
        #     train_dataloader.num_workers=4 \
        #     training.n_epochs=100 \
        #     rollout.interval=20 \
            task.demos_per_env=10 \
        #     pace_copy=true  \
        #     seed=${seed}
    done
done

# python train.py \
#     --config-name=train_prior.yaml \
#     task=libero_90_hybrid \
#     exp_name=final_backup \
#     variant_name=block_8 \
#     algo=diffuser_actor \
#     algo/encoder=hybrid  \
#     algo.chunk_size=8 \
#     train_dataloader.num_workers4 \
#     training.n_epochs=100 \
#     rollout.interval=20 \
            task.demos_per_env=10 \
#     seed=0





