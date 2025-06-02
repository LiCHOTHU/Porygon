exp_name="debugin_diffuser_actor"
task="libero_10_hybrid"
blocks=(8)
seeds=(0)

export HYDRA_FULL_ERROR=1

# algo.policy.temporal_agg=true \
# algo/aug=color_jitter \
# ~task.shape_meta.observation.lowdim \
# sbatch slurm/run_rtx6000.sbatch 

for block in ${blocks[@]}; do
    for seed in ${seeds[@]}; do
        python train.py \
            --config-name=train_prior.yaml \
            task=${task} \
            exp_name=${exp_name} \
            variant_name=debug_${block} \
            algo=diffuser_actor \
            algo/encoder=hybrid  \
            algo.chunk_size=${block} \
            training.use_tqdm=true \
            training.use_amp=false \
            training.save_all_checkpoints=true \
            train_dataloader.persistent_workers=true \
            train_dataloader.num_workers=6 \
            train_dataloader.batch_size=64 \
            make_unique_experiment_dir=true \
            training.n_epochs=2000 \
            training.resume=true \
            logging.mode=disabled \
            task.demos_per_env=1 \
            rollout.interval=1 \
            seed=${seed}
    done
done