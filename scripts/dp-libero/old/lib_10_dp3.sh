# python train.py \
#     --config-name=train_prior.yaml \
#     task=libero_10_rgbd \
#     algo=diffusion_policy \
#     algo.chunk_size=16 \
#     algo/aug=image \
#     training.use_tqdm=true \
#     training.use_amp=false \
#     training.save_all_checkpoints=false \
#     train_dataloader.persistent_workers=false \
#     train_dataloader.num_workers=6 \
#     make_unique_experiment_dir=true \
#     training.n_epochs=2000 \
#     rollout.interval=50 \
#     logging.mode=disabled



blocks=(8 16)
seeds=(0 1)
tasks=("libero_10_dp3")

for block in ${blocks[@]}; do
    for seed in ${seeds[@]}; do
        for task in ${tasks[@]}; do
            sbatch slurm/run_rtx6000.sbatch python train.py \
                --config-name=train_prior.yaml \
                task=${task} \
                algo=diffusion_policy \
                algo.chunk_size=${block} \
                algo/aug=identity \
                exp_name=diff_po_1 \
                variant_name=dp3_block_${block} \
                training.use_tqdm=false \
                training.use_amp=false \
                training.save_all_checkpoints=true \
                train_dataloader.persistent_workers=true \
                train_dataloader.num_workers=6 \
                train_dataloader.batch_size=64 \
                make_unique_experiment_dir=false \
                training.n_epochs=2000 \
                training.resume=true \
                rollout.interval=50 \
                seed=${seed}
        done
    done
done


