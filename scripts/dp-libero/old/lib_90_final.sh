# python train.py \
#     --config-name=train_prior.yaml \
#     task=libero_90_hybrid \
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
#     algo.embed_dim=512 \
#     +algo.encoder.downsample_mode=feat \
#     +algo.encoder.fps_library=dgl \
#     +algo.encoder.pointcloud_extractor_factory.reduction=attention \
#     training.use_amp=false \
#     train_dataloader.num_workers=6 \
#     train_dataloader.batch_size=64 \
#     training.n_epochs=2000 \
#     training.resume=true \
#     training.save_interval=5 \
#     rollout.interval=20 \
#     task.demos_per_env=25 \
#     logging.mode=disabled


exp_name="diffusion_policy_no_crop_3"
blocks=(16)
seeds=(0 1)
downsample_modes=("feat")

for block in ${blocks[@]}; do
    for seed in ${seeds[@]}; do
        for downsample_mode in ${downsample_modes[@]}; do
            # attention reduction
            sbatch slurm/run_v100.sbatch python train.py \
                --config-name=train_prior.yaml \
                task=libero_90_hybrid \
                exp_name=${exp_name} \
                variant_name=flagship_block_${block} \
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
                +algo.encoder.fps_library=dgl \
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
            
        done
    done
done

