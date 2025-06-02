python train.py \
    --config-name=train_prior.yaml \
    task=libero_10_rgb \
    algo=diffusion_policy \
    algo.chunk_size=16 \
    algo.policy.temporal_agg=true \
    algo/aug=image \
    algo.encoder.image_encoder_factory.pretrained=true \
    +algo.encoder.language_fusion=true \
    train_dataloader.num_workers=6 \
    train_dataloader.batch_size=64 \
    training.n_epochs=2000 \
    training.resume=true \
    rollout.interval=20 \
    logging.mode=disabled


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
            task=libero_10_rgb \
            algo=diffusion_policy \
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
            rollout.interval=20 \
            seed=${seed}

        # RGBD baseline
        sbatch slurm/run_rtx6000.sbatch python train.py \
            --config-name=train_prior.yaml \
            exp_name=${exp_name} \
            variant_name=rgbd_block_${block} \
            task=libero_10_rgbd \
            algo=diffusion_policy \
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
            rollout.interval=20 \
            seed=${seed}
        
        # DP3 baseline
        sbatch slurm/run_rtx6000.sbatch python train.py \
            --config-name=train_prior.yaml \
            exp_name=${exp_name} \
            variant_name=dp3_block_${block} \
            task=libero_10_dp3 \
            algo=diffusion_policy \
            algo.chunk_size=${block} \
            algo.policy.temporal_agg=true \
            algo/aug=identity \
            training.use_tqdm=false \
            training.use_amp=false \
            training.save_all_checkpoints=true \
            train_dataloader.persistent_workers=true \
            train_dataloader.num_workers=6 \
            train_dataloader.batch_size=64 \
            make_unique_experiment_dir=false \
            training.n_epochs=2000 \
            training.resume=true \
            rollout.interval=20 \
            seed=${seed}
        
        # Hybrid
        sbatch slurm/run_rtx6000.sbatch python train.py \
            --config-name=train_prior.yaml \
            exp_name=${exp_name} \
            variant_name=hybrid_clip_block_${block} \
            task=libero_10_hybrid \
            algo=diffusion_policy \
            algo/encoder=hybrid  \
            algo.chunk_size=${block} \
            algo.policy.temporal_agg=true \
            algo/aug=color_jitter \
            ~task.shape_meta.observation.lowdim \
            +task.shape_meta.observation.lowdim={robot0_gripper_qpos:2} \
            algo.encoder.backbone_type=clip \
            algo.encoder.finetune=false \
            algo.encoder.hand_frame=true \
            training.use_tqdm=false \
            training.use_amp=false \
            training.save_all_checkpoints=true \
            train_dataloader.persistent_workers=true \
            train_dataloader.num_workers=6 \
            train_dataloader.batch_size=64 \
            make_unique_experiment_dir=false \
            training.n_epochs=2000 \
            training.resume=true \
            rollout.interval=20 \
            seed=${seed}
    done
done

