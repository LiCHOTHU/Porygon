# python train.py \
#     --config-name=train_prior.yaml \
#     task=metaworld_mt50_rgb \
#     algo=baku \
#     algo.chunk_size=15 \
#     algo.policy.temporal_agg=true \
#     algo/aug=image \
#     algo.encoder.image_encoder_factory.pretrained=true \
#     +algo.encoder.language_fusion=true \
#     training.use_tqdm=true \
#     training.use_amp=false \
#     train_dataloader.num_workers=6 \
#     train_dataloader.batch_size=64 \
#     training.n_epochs=2000 \
#     training.resume=true \
#     rollout.interval=20 \
#     training.save_interval=5 \
#     task.demos_per_env=10 \
#     logging.mode=disabled


exp_name="baku_baseline"
blocks=(10 15)
seeds=(0 1)

for block in ${blocks[@]}; do
    for seed in ${seeds[@]}; do
        # RGB baseline
        echo sbatch slurm/run_rtx6000.sbatch python train.py \
            --config-name=train_prior.yaml \
            exp_name=${exp_name} \
            variant_name=rgb_block_${block} \
            task=metaworld_mt50_rgb \
            algo=baku \
            algo.chunk_size=${block} \
            algo.policy.temporal_agg=true \
            algo/aug=image \
            algo.encoder.image_encoder_factory.pretrained=true \
            +algo.encoder.language_fusion=true \
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
            training.save_interval=5 \
            task.demos_per_env=10 \
            seed=${seed}

        # RGBD baseline
        echo sbatch slurm/run_rtx6000.sbatch python train.py \
            --config-name=train_prior.yaml \
            exp_name=${exp_name} \
            variant_name=rgbd_block_${block} \
            task=metaworld_mt50_rgbd \
            algo=baku \
            algo.chunk_size=${block} \
            algo.policy.temporal_agg=true \
            algo/aug=image \
            algo.encoder.image_encoder_factory.pretrained=true \
            +algo.encoder.language_fusion=true \
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
            training.save_interval=5 \
            task.demos_per_env=10 \
            seed=${seed}
        
        # # DP3 baseline
        # sbatch slurm/run_rtx6000.sbatch python train.py \
        #     --config-name=train_prior.yaml \
        #     task=metaworld_mt50_hybrid \
        #     exp_name=${exp_name} \
        #     variant_name=dp3_no_crop_block_${block} \
        #     algo=baku \
        #     algo/encoder=hybrid  \
        #     algo.chunk_size=15 \
        #     algo.policy.temporal_agg=true \
        #     algo/aug=identity \
        #     algo.encoder.hand_frame=false \
        #     algo.encoder.do_crop=false \
        #     +algo.encoder.do_image=false \
        #     +algo.encoder.do_lang=false \
        #     +algo.encoder.xyz_proj_type=none \
        #     algo.embed_dim=512 \
        #     +algo.encoder.downsample_mode=pos \
        #     training.use_tqdm=false \
        #     training.use_amp=false \
        #     training.save_all_checkpoints=true \
        #     train_dataloader.persistent_workers=true \
        #     train_dataloader.num_workers=6 \
        #     train_dataloader.batch_size=64 \
        #     make_unique_experiment_dir=false \
        #     training.n_epochs=2000 \
        #     training.resume=true \
        #     training.save_interval=5 \
        #     rollout.interval=20 \
        #     task.demos_per_env=25 \
        #     seed=${seed}
    done
done
