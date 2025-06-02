
blocks=(4 8 16)
seeds=(0 1)
for block in ${blocks[@]}; do
    for seed in ${seeds[@]}; do
        sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
            task=metaworld_pc_mt50 \
            algo=diffusion_policy_pc \
            exp_name=dp_initial \
            variant_name=block_${block} \
            training.use_tqdm=false \
            training.use_amp=true \
            training.save_all_checkpoints=true \
            train_dataloader.persistent_workers=true \
            train_dataloader.num_workers=6 \
            make_unique_experiment_dir=false \
            algo.chunk_size=${block} \
            training.n_epochs=1000 \
            rollout.interval=25 \
            seed=${seed}
    done
done



# python fix.py --config-name=train_prior.yaml \
#     task=metaworld_ml45_prise \
#     algo=diffusion_policy \
#     exp_name=dp_final \
#     variant_name=block_16 \
#     training.use_tqdm=false \
#     training.use_amp=true \
#     training.save_all_checkpoints=true \
#     train_dataloader.persistent_workers=true \
#     train_dataloader.num_workers=6 \
#     make_unique_experiment_dir=false \
#     algo.chunk_size=16 \
#     training.n_epochs=200 \
#     training.resume=true \
#     seed=1


# python fix.py --config-name=train_prior.yaml \
#     task=metaworld_ml45_prise \
#     algo=diffusion_policy \
#     exp_name=dp_final \
#     variant_name=block_16 \
#     training.use_tqdm=false \
#     training.use_amp=true \
#     training.save_all_checkpoints=true \
#     train_dataloader.persistent_workers=true \
#     train_dataloader.num_workers=6 \
#     make_unique_experiment_dir=false \
#     algo.chunk_size=16 \
#     training.n_epochs=200 \
#     training.resume=true \
#     seed=2

