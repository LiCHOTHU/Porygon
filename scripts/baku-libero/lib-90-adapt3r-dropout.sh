exp_name="corl-push-2"
seeds=(0)
dropouts=(0.05 0.1 0.2)
algo="baku"

for seed in ${seeds[@]}; do

    for dropout in ${dropouts[@]}; do
        # adapt3r baseline
        sbatch slurm/run_l40s.sbatch python train.py \
            --config-name=train_prior.yaml \
            pace_copy=true  \
            task=libero_90_hybrid \
            exp_name=${exp_name} \
            variant_name=adapt3r_d_${dropout} \
            algo=${algo} \
            algo/encoder=hybrid  \
            algo.chunk_size=10 \
            +algo.encoder.dropout=${dropout} \
            training.save_interval=10 \
            train_dataloader.num_workers=4 \
            rollout.interval=50 \
            task.demos_per_env=50 \
            training.n_epochs=51 \
            seed=${seed}
    done

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

