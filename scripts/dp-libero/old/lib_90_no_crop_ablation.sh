
exp_name="diffusion_policy_no_crop_ablation"
blocks=(16)
seeds=(0 1)
downsample_modes=("feat")

for block in ${blocks[@]}; do
    for seed in ${seeds[@]}; do
        for downsample_mode in ${downsample_modes[@]}; do
            # no image
            sbatch slurm/run_rtx6000.sbatch python train.py \
                --config-name=train_prior.yaml \
                task=libero_90_hybrid \
                exp_name=${exp_name} \
                variant_name=no_image_block_${block} \
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
                algo.encoder.do_crop=false \
                algo.embed_dim=512 \
                +algo.encoder.downsample_mode=feat \
                +algo.encoder.pca_compression=true \
                +algo.encoder.pointcloud_extractor_factory.reduction=attention \
                +algo.encoder.do_image=false \
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

            # no lang
            sbatch slurm/run_rtx6000.sbatch python train.py \
                --config-name=train_prior.yaml \
                task=libero_90_hybrid \
                exp_name=${exp_name} \
                variant_name=no_lang_block_${block} \
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
                algo.encoder.do_crop=false \
                algo.embed_dim=512 \
                +algo.encoder.downsample_mode=feat \
                +algo.encoder.pca_compression=true \
                +algo.encoder.pointcloud_extractor_factory.reduction=attention \
                +algo.encoder.do_lang=false \
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

            # no eecf
            sbatch slurm/run_rtx6000.sbatch python train.py \
                --config-name=train_prior.yaml \
                task=libero_90_hybrid \
                exp_name=${exp_name} \
                variant_name=no_eecf_block_${block} \
                algo=diffusion_policy \
                algo/encoder=hybrid  \
                algo.chunk_size=${block} \
                algo.policy.temporal_agg=true \
                algo/aug=color_jitter \
                ~task.shape_meta.observation.lowdim \
                +task.shape_meta.observation.lowdim={robot0_gripper_qpos:2} \
                algo.encoder.backbone_type=clip \
                algo.encoder.finetune=false \
                algo.encoder.hand_frame=false \
                algo.encoder.do_crop=false \
                algo.embed_dim=512 \
                +algo.encoder.downsample_mode=feat \
                +algo.encoder.pca_compression=true \
                +algo.encoder.pointcloud_extractor_factory.reduction=attention \
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

            # learned lift
            sbatch slurm/run_rtx6000.sbatch python train.py \
                --config-name=train_prior.yaml \
                task=libero_90_hybrid \
                exp_name=${exp_name} \
                variant_name=learned_lift_block_${block} \
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
                algo.encoder.do_crop=false \
                algo.embed_dim=512 \
                +algo.encoder.downsample_mode=feat \
                +algo.encoder.pca_compression=true \
                +algo.encoder.pointcloud_extractor_factory.reduction=attention \
                +algo.encoder.xyz_proj_type=learned \
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

            # pos downsample
            sbatch slurm/run_rtx6000.sbatch python train.py \
                --config-name=train_prior.yaml \
                task=libero_90_hybrid \
                exp_name=${exp_name} \
                variant_name=pos_ds_block_${block} \
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
                algo.encoder.do_crop=false \
                algo.embed_dim=512 \
                +algo.encoder.downsample_mode=pos \
                +algo.encoder.pointcloud_extractor_factory.reduction=attention \
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

            # no attn
            sbatch slurm/run_rtx6000.sbatch python train.py \
                --config-name=train_prior.yaml \
                task=libero_90_hybrid \
                exp_name=${exp_name} \
                variant_name=no_attn_block_${block} \
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
                algo.encoder.do_crop=false \
                algo.embed_dim=512 \
                +algo.encoder.downsample_mode=feat \
                +algo.encoder.pca_compression=true \
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
done
      