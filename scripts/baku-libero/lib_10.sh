exp_name="lib_10_testing"
seeds=(0)

for seed in ${seeds[@]}; do

    # no lang
    sbatch slurm/run_l40s.sbatch python train.py \
        --config-name=train_prior.yaml \
        pace_copy=true  \
        task=libero_10_hybrid \
        exp_name=${exp_name} \
        variant_name=baseline_2 \
        algo=baku \
        algo/encoder=hybrid  \
        algo/aug=image \
        algo.chunk_size=10 \
        +algo.policy.eecf=true \
        algo.encoder.do_hand_crop=true \
        +algo.encoder.tight_crop=true \
        training.save_interval=20 \
        train_dataloader.num_workers=8 \
        rollout.interval=20 \
        task.demos_per_env=50 \
        training.n_epochs=101 \
        seed=${seed}

done


# python train.py \
#     --config-name=train_debug.yaml \
#     pace_copy=true  \
#     task=libero_10_hybrid \
#     algo=baku \
#     algo/encoder=hybrid  \
#     algo/aug=image \
#     algo.chunk_size=10 \
#     +algo.policy.eecf=true \
#     algo.encoder.do_hand_crop=true \
#     +algo.encoder.tight_crop=true \
#     training.save_interval=20 \
#     train_dataloader.num_workers=8 \
#     rollout.interval=20 \
#     task.demos_per_env=50 \
#     training.n_epochs=101
