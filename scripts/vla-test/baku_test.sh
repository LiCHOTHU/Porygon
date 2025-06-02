task="libero_90_hybrid"
exp_name="head_sweep_3"


sbatch slurm/run_l40s.sbatch python train.py \
    --config-name=train_prior.yaml \
    task=${task} \
    exp_name=${exp_name} \
    variant_name=baku_eecf \
    pace_copy=true \
    algo=baku \
    algo/encoder=hybrid  \
    algo/aug=image \
    algo.chunk_size=10 \
    +algo.policy.eecf=true \
    algo.encoder.do_hand_crop=true \
    +algo.encoder.tight_crop=true \
    training.save_interval=20 \
    train_dataloader.num_workers=4 \
    rollout.interval=200 \
    task.demos_per_env=50 \
    training.n_epochs=101
