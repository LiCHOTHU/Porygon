exp_name="corl-push-2"
task="libero_90_hybrid"
blocks=(16)
# seeds=(0 1 2 3 4)
seeds=(0 1 2)

export HYDRA_FULL_ERROR=1

# algo.policy.temporal_agg=true \
# algo/aug=color_jitter \
# ~task.shape_meta.observation.lowdim \
# sbatch slurm/run_rtx6000.sbatch 

for block in ${blocks[@]}; do
    for seed in ${seeds[@]}; do
        sbatch slurm/run_l40s.sbatch python train.py \
            --config-name=train_prior.yaml \
            task=${task} \
            exp_name=${exp_name} \
            variant_name=3dda \
            algo=diffuser_actor \
            algo/encoder=hybrid  \
            algo.chunk_size=${block} \
            train_dataloader.num_workers=4 \
            training.n_epochs=100 \
            rollout.interval=100 \
            pace_copy=true  \
            seed=${seed}
        
        # sbatch slurm/run_l40s.sbatch python train.py \
        #     --config-name=train_prior.yaml \
        #     task=${task} \
        #     exp_name=${exp_name} \
        #     variant_name=beefy_block_${block} \
        #     algo=diffuser_actor \
        #     algo/encoder=hybrid  \
        #     algo.chunk_size=${block} \
        #     +algo.policy.beefy=true \
        #     train_dataloader.num_workers=10 \
        #     training.n_epochs=100 \
        #     rollout.interval=100 \
        #     pace_copy=true  \
        #     seed=${seed}

        # sbatch slurm/run_l40s.sbatch python train.py \
        #     --config-name=train_prior.yaml \
        #     task=${task} \
        #     exp_name=${exp_name} \
        #     variant_name=beefy_relative_block_${block} \
        #     algo=diffuser_actor \
        #     algo/encoder=hybrid  \
        #     algo.chunk_size=${block} \
        #     +algo.policy.beefy=true \
        #     algo.policy.relative=true \
        #     train_dataloader.num_workers=10 \
        #     training.n_epochs=100 \
        #     rollout.interval=100 \
        #     pace_copy=true  \
        #     seed=${seed}
        
        # sbatch slurm/run_l40s.sbatch python train.py \
        #     --config-name=train_prior.yaml \
        #     task=${task} \
        #     exp_name=${exp_name} \
        #     variant_name=relative_block_${block} \
        #     algo=diffuser_actor \
        #     algo/encoder=hybrid  \
        #     algo.chunk_size=${block} \
        #     algo.policy.relative=true \
        #     train_dataloader.num_workers=10 \
        #     training.n_epochs=100 \
        #     rollout.interval=100 \
        #     pace_copy=true  \
        #     seed=${seed}

        # sbatch slurm/run_l40s.sbatch python train.py \
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
        #     train_dataloader.num_workers=10 \
        #     training.n_epochs=100 \
        #     rollout.interval=100 \
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
#     train_dataloader.num_workers=8 \
#     training.n_epochs=100 \
#     rollout.interval=100 \
#     seed=0





# python train.py \
#     --config-name=train_debug.yaml \
#     task=libero_90_hybrid \
#     algo=diffuser_actor \
#     algo/encoder=hybrid  \
#     algo.chunk_size=16 \
#     train_dataloader.num_workers=4 \
#     training.n_epochs=100 \
#     rollout.interval=100 \
#     pace_copy=true  

