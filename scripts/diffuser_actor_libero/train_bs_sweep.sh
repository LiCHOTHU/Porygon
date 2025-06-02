python train.py \
    --config-name=train_debug.yaml \
    task=libero_90_hybrid \
    algo=diffuser_actor \
    algo/encoder=hybrid  \
    algo.chunk_size=8 \
    train_dataloader.num_workers=4 \
    training.n_epochs=100 \
    rollout.interval=100

exp_name="debugin_diffuser_actor"
task="libero_90_hybrid"
blocks=(4 8 16)
seeds=(0)

export HYDRA_FULL_ERROR=1

# algo.policy.temporal_agg=true \
# algo/aug=color_jitter \
# ~task.shape_meta.observation.lowdim \
# sbatch slurm/run_rtx6000.sbatch 

for block in ${blocks[@]}; do
    for seed in ${seeds[@]}; do
        # RGB baseline
        python train.py \
            --config-name=train_prior.yaml \
            task=${task} \
            exp_name=${exp_name} \
            variant_name=debug_${block} \
            algo=diffuser_actor \
            algo/encoder=hybrid  \
            algo.chunk_size=${block} \
            train_dataloader.num_workers=8 \
            training.n_epochs=100 \
            rollout.interval=100 \
            seed=${seed}
    done
done