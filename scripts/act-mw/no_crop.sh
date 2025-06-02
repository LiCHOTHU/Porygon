# python train.py \
#     --config-name=train_prior.yaml \
#     task=libero_10_hybrid \
#     algo=act \
#     algo/encoder=hybrid  \
#     algo.chunk_size=15 \
#     algo.policy.temporal_agg=true \
#     algo/aug=color_jitter \
#     ~task.shape_meta.observation.lowdim \
#     +task.shape_meta.observation.lowdim={robot0_gripper_qpos:2} \
#     algo.encoder.backbone_type=clip \
#     algo.encoder.finetune=false \
#     algo.encoder.hand_frame=true \
#     algo.encoder.do_crop=false \
#     +algo.encoder.downsample_mode=none \
#     +algo.encoder.pointcloud_extractor_factory.reduction=attention \
#     algo.embed_dim=512 \
#     training.use_tqdm=true \
#     training.use_amp=false \
#     training.save_all_checkpoints=false \
#     train_dataloader.persistent_workers=false \
#     train_dataloader.num_workers=6 \
#     train_dataloader.batch_size=64 \
#     make_unique_experiment_dir=true \
#     training.n_epochs=2000 \
#     training.resume=true \
#     training.save_interval=5 \
#     rollout.interval=20 \
#     task.demos_per_env=25 \
#     logging.mode=disabled

exp_name="act_no_crop_1"
blocks=(10 15)
seeds=(0 1)
downsample_modes=("feat")

for block in ${blocks[@]}; do
    for seed in ${seeds[@]}; do
        for downsample_mode in ${downsample_modes[@]}; do
            # DP3 baseline
            # DP3 baseline
            sbatch slurm/run_rtx6000.sbatch python train.py \
                --config-name=train_prior.yaml \
                task=libero_90_hybrid \
                exp_name=${exp_name} \
                variant_name=dp3_no_crop_block_${block} \
                algo=act \
                algo/encoder=hybrid  \
                algo.chunk_size=15 \
                algo.policy.temporal_agg=true \
                algo/aug=identity \
                algo.encoder.hand_frame=false \
                algo.encoder.do_crop=false \
                +algo.encoder.do_image=false \
                +algo.encoder.do_lang=false \
                +algo.encoder.xyz_proj_type=none \
                algo.embed_dim=512 \
                +algo.encoder.downsample_mode=pos \
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

            # attention reduction
            sbatch slurm/run_rtx6000.sbatch python train.py \
                --config-name=train_prior.yaml \
                task=libero_90_hybrid \
                exp_name=${exp_name} \
                variant_name=attention_reduction_block_${block} \
                algo=act \
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
                +algo.encoder.downsample_mode=none \
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
            
            # attention reduction finetune
            sbatch slurm/run_rtx6000.sbatch python train.py \
                --config-name=train_prior.yaml \
                task=libero_90_hybrid \
                exp_name=${exp_name} \
                variant_name=attention_reduction_finetune_block_${block} \
                algo=act \
                algo/encoder=hybrid  \
                algo.chunk_size=${block} \
                algo.policy.temporal_agg=true \
                algo/aug=color_jitter \
                ~task.shape_meta.observation.lowdim \
                +task.shape_meta.observation.lowdim={robot0_gripper_qpos:2} \
                algo.encoder.backbone_type=clip \
                algo.encoder.finetune=true \
                algo.encoder.hand_frame=true \
                algo.encoder.do_crop=false \
                algo.embed_dim=512 \
                +algo.encoder.downsample_mode=none \
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
            
            

            # baseline
            sbatch slurm/run_rtx6000.sbatch python train.py \
                --config-name=train_prior.yaml \
                task=libero_90_hybrid \
                exp_name=${exp_name} \
                variant_name=baseline_block_${block} \
                algo=act \
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
                +algo.encoder.downsample_mode=${downsample_mode} \
                +algo.encoder.pca_compression=true \
                algo.embed_dim=512 \
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
            

            # baseline finetune
            sbatch slurm/run_rtx6000.sbatch python train.py \
                --config-name=train_prior.yaml \
                task=libero_90_hybrid \
                exp_name=${exp_name} \
                variant_name=finetune_block_${block} \
                algo=act \
                algo/encoder=hybrid  \
                algo.chunk_size=${block} \
                algo.policy.temporal_agg=true \
                algo/aug=color_jitter \
                ~task.shape_meta.observation.lowdim \
                +task.shape_meta.observation.lowdim={robot0_gripper_qpos:2} \
                algo.encoder.backbone_type=clip \
                algo.encoder.finetune=true \
                algo.encoder.hand_frame=true \
                algo.encoder.do_crop=false \
                +algo.encoder.downsample_mode=${downsample_mode} \
                +algo.encoder.pca_compression=true \
                algo.embed_dim=512 \
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
            
            # # learned reduction
            # sbatch slurm/run_rtx6000.sbatch python train.py \
            #     --config-name=train_prior.yaml \
            #     task=libero_90_hybrid \
            #     exp_name=${exp_name} \
            #     variant_name=learned_reduction_block_${block} \
            #     algo=act \
            #     algo/encoder=hybrid  \
            #     algo.chunk_size=${block} \
            #     algo.policy.temporal_agg=true \
            #     algo/aug=color_jitter \
            #     ~task.shape_meta.observation.lowdim \
            #     +task.shape_meta.observation.lowdim={robot0_gripper_qpos:2} \
            #     algo.encoder.backbone_type=clip \
            #     algo.encoder.finetune=false \
            #     algo.encoder.hand_frame=true \
            #     algo.encoder.do_crop=false \
            #     algo.embed_dim=512 \
            #     +algo.encoder.downsample_mode=none \
            #     +algo.encoder.pointcloud_extractor_factory.reduction=learned \
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

            # # learned reduction finetune
            # sbatch slurm/run_rtx6000.sbatch python train.py \
            #     --config-name=train_prior.yaml \
            #     task=libero_90_hybrid \
            #     exp_name=${exp_name} \
            #     variant_name=learned_reduction_finetune_block_${block} \
            #     algo=act \
            #     algo/encoder=hybrid  \
            #     algo.chunk_size=${block} \
            #     algo.policy.temporal_agg=true \
            #     algo/aug=color_jitter \
            #     ~task.shape_meta.observation.lowdim \
            #     +task.shape_meta.observation.lowdim={robot0_gripper_qpos:2} \
            #     algo.encoder.backbone_type=clip \
            #     algo.encoder.finetune=true \
            #     algo.encoder.hand_frame=true \
            #     algo.encoder.do_crop=false \
            #     algo.embed_dim=512 \
            #     +algo.encoder.downsample_mode=none \
            #     +algo.encoder.pointcloud_extractor_factory.reduction=learned \
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
            

            # # # noimage
            # # sbatch slurm/run_rtx6000.sbatch python train.py \
            # #     --config-name=train_prior.yaml \
            # #     task=libero_90_hybrid \
            # #     exp_name=${exp_name} \
            # #     variant_name=no_image_block_${block} \
            # #     algo=act \
            # #     algo/encoder=hybrid  \
            # #     algo.chunk_size=${block} \
            # #     algo.policy.temporal_agg=true \
            # #     algo/aug=color_jitter \
            # #     ~task.shape_meta.observation.lowdim \
            # #     +task.shape_meta.observation.lowdim={robot0_gripper_qpos:2} \
            # #     algo.encoder.backbone_type=clip \
            # #     algo.encoder.finetune=false \
            # #     algo.encoder.hand_frame=true \
            # #     algo.encoder.do_crop=false \
            # #     +algo.encoder.downsample_mode=${downsample_mode} \
            # # +algo.encoder.pca_compression=true \
            # #     algo.embed_dim=512 \
            # #     +algo.encoder.do_image=false \
            # #     training.use_tqdm=false \
            # #     training.use_amp=false \
            # #     training.save_all_checkpoints=true \
            # #     train_dataloader.persistent_workers=true \
            # #     train_dataloader.num_workers=6 \
            # #     train_dataloader.batch_size=64 \
            # #     make_unique_experiment_dir=false \
            # #     training.n_epochs=2000 \
            # #     training.resume=true \
            # #     training.save_interval=5 \
            # #     rollout.interval=20 \
            # #     task.demos_per_env=25 \
            # #     seed=${seed}
            
            # # # no lang
            # # sbatch slurm/run_rtx6000.sbatch python train.py \
            # #     --config-name=train_prior.yaml \
            # #     task=libero_90_hybrid \
            # #     exp_name=${exp_name} \
            # #     variant_name=no_lang_block_${block} \
            # #     algo=act \
            # #     algo/encoder=hybrid  \
            # #     algo.chunk_size=${block} \
            # #     algo.policy.temporal_agg=true \
            # #     algo/aug=color_jitter \
            # #     ~task.shape_meta.observation.lowdim \
            # #     +task.shape_meta.observation.lowdim={robot0_gripper_qpos:2} \
            # #     algo.encoder.backbone_type=clip \
            # #     algo.encoder.finetune=false \
            # #     algo.encoder.hand_frame=true \
            # #     algo.encoder.do_crop=false \
            # #     +algo.encoder.downsample_mode=${downsample_mode} \
            # # +algo.encoder.pca_compression=true \
            # #     algo.embed_dim=512 \
            # #     +algo.encoder.do_lang=false \
            # #     training.use_tqdm=false \
            # #     training.use_amp=false \
            # #     training.save_all_checkpoints=true \
            # #     train_dataloader.persistent_workers=true \
            # #     train_dataloader.num_workers=6 \
            # #     train_dataloader.batch_size=64 \
            # #     make_unique_experiment_dir=false \
            # #     training.n_epochs=2000 \
            # #     training.resume=true \
            # #     training.save_interval=5 \
            # #     rollout.interval=20 \
            # #     task.demos_per_env=25 \
            # #     seed=${seed}
            
            # # # no lang finetune
            # # sbatch slurm/run_rtx6000.sbatch python train.py \
            # #     --config-name=train_prior.yaml \
            # #     task=libero_90_hybrid \
            # #     exp_name=${exp_name} \
            # #     variant_name=no_lang_finetune_block_${block} \
            # #     algo=act \
            # #     algo/encoder=hybrid  \
            # #     algo.chunk_size=${block} \
            # #     algo.policy.temporal_agg=true \
            # #     algo/aug=color_jitter \
            # #     ~task.shape_meta.observation.lowdim \
            # #     +task.shape_meta.observation.lowdim={robot0_gripper_qpos:2} \
            # #     algo.encoder.backbone_type=clip \
            # #     algo.encoder.finetune=true \
            # #     algo.encoder.hand_frame=true \
            # #     algo.encoder.do_crop=false \
            # #     +algo.encoder.downsample_mode=${downsample_mode} \
            # # +algo.encoder.pca_compression=true \
            # #     algo.embed_dim=512 \
            # #     +algo.encoder.do_lang=false \
            # #     training.use_tqdm=false \
            # #     training.use_amp=false \
            # #     training.save_all_checkpoints=true \
            # #     train_dataloader.persistent_workers=true \
            # #     train_dataloader.num_workers=6 \
            # #     train_dataloader.batch_size=64 \
            # #     make_unique_experiment_dir=false \
            # #     training.n_epochs=2000 \
            # #     training.resume=true \
            # #     training.save_interval=5 \
            # #     rollout.interval=20 \
            # #     task.demos_per_env=25 \
            # #     seed=${seed}

            # # # only pos
            # # sbatch slurm/run_rtx6000.sbatch python train.py \
            # #     --config-name=train_prior.yaml \
            # #     task=libero_90_hybrid \
            # #     exp_name=${exp_name} \
            # #     variant_name=no_image_no_lang_block_${block} \
            # #     algo=act \
            # #     algo/encoder=hybrid  \
            # #     algo.chunk_size=${block} \
            # #     algo.policy.temporal_agg=true \
            # #     algo/aug=color_jitter \
            # #     ~task.shape_meta.observation.lowdim \
            # #     +task.shape_meta.observation.lowdim={robot0_gripper_qpos:2} \
            # #     algo.encoder.backbone_type=clip \
            # #     algo.encoder.finetune=false \
            # #     algo.encoder.hand_frame=true \
            # #     algo.encoder.do_crop=false \
            # #     +algo.encoder.downsample_mode=${downsample_mode} \
            # # +algo.encoder.pca_compression=true \
            # #     algo.embed_dim=512 \
            # #     +algo.encoder.do_lang=false \
            # #     +algo.encoder.do_image=false \
            # #     training.use_tqdm=false \
            # #     training.use_amp=false \
            # #     training.save_all_checkpoints=true \
            # #     train_dataloader.persistent_workers=true \
            # #     train_dataloader.num_workers=6 \
            # #     train_dataloader.batch_size=64 \
            # #     make_unique_experiment_dir=false \
            # #     training.n_epochs=2000 \
            # #     training.resume=true \
            # #     training.save_interval=5 \
            # #     rollout.interval=20 \
            # #     task.demos_per_env=25 \
            # #     seed=${seed}
            
        done
    done
done

