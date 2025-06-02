exp_name="diffusion_policy_2"
blocks=(4 8 16)
seeds=(0 1)

for block in ${blocks[@]}; do
    for seed in ${seeds[@]}; do
        # RGB baseline
        sbatch slurm/run_rtx6000.sbatch python train.py \
            --config-name=train_prior.yaml \
            exp_name=${exp_name} \
            variant_name=25_demo_rgb_block_${block} \
            task=libero_90_rgb \
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
            training.save_interval=5 \
            rollout.interval=20 \
            task.demos_per_env=25 \
            seed=${seed}

        # RGBD baseline
        sbatch slurm/run_rtx6000.sbatch python train.py \
            --config-name=train_prior.yaml \
            exp_name=${exp_name} \
            variant_name=25_demo_rgbd_block_${block} \
            task=libero_90_rgbd \
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
            training.save_interval=5 \
            rollout.interval=20 \
            task.demos_per_env=25 \
            seed=${seed}
        
        # DP3 baseline
        sbatch slurm/run_rtx6000.sbatch python train.py \
            --config-name=train_prior.yaml \
            exp_name=${exp_name} \
            variant_name=25_demo_dp3_block_${block} \
            task=libero_90_dp3 \
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
            training.save_interval=5 \
            rollout.interval=20 \
            task.demos_per_env=25 \
            seed=${seed}
        
        # Hybrid
        sbatch slurm/run_rtx6000.sbatch python train.py \
            --config-name=train_prior.yaml \
            task=libero_90_hybrid \
            exp_name=${exp_name} \
            variant_name=25_demo_hybrid_clip_block_${block} \
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
            training.save_interval=5 \
            rollout.interval=20 \
            task.demos_per_env=25 \
            seed=${seed}
    done
done

