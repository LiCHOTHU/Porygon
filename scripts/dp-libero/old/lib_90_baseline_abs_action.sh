python train.py \
    --config-name=train_debug.yaml \
    exp_name=abs_action_debug \
    task=libero_10_rgb \
    algo=diffusion_policy \
    algo.chunk_size=8 \
    algo/aug=image \
    train_dataloader.num_workers=6 \
    training.n_epochs=100 \
    training.save_interval=20 \
    rollout.interval=10 \
    
    seed=0

exp_name="diffusion_policy_2"
blocks=(4 8 16)
seeds=(0 1)

for block in ${blocks[@]}; do
    for seed in ${seeds[@]}; do
        # RGB baseline
        sbatch slurm/run_rtx6000.sbatch python train.py \
            --config-name=train_prior.yaml \
            exp_name=${exp_name} \
            variant_name=rgb_block_${block} \
            task=libero_90_rgb \
            algo=diffusion_policy \
            algo.chunk_size=${block} \
            algo/aug=image \
            train_dataloader.num_workers=6 \
            training.n_epochs=100 \
            training.save_interval=20 \
            rollout.interval=100 \
            seed=${seed}
    done
done

