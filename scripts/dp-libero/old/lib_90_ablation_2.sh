# python train.py \
#     --config-name=train_prior.yaml \
#     task=libero_10_hybrid \
#     algo=diffusion_policy \
#     algo/encoder=hybrid  \
#     algo.policy.temporal_agg=true \
#     algo/aug=color_jitter \
#     ~task.shape_meta.observation.lowdim \
#     +task.shape_meta.observation.lowdim={robot0_gripper_qpos:2} \
#     algo.encoder.backbone_type=clip \
#     algo.encoder.finetune=false \
#     algo.encoder.hand_frame=true \
#     +algo.encoder.do_image=false \
#     training.use_tqdm=true \
#     training.use_amp=false \
#     training.save_all_checkpoints=false \
#     train_dataloader.persistent_workers=false \
#     train_dataloader.num_workers=6 \
#     train_dataloader.batch_size=64 \
#     make_unique_experiment_dir=true \
#     training.n_epochs=2000 \
#     training.resume=true \
#     rollout.interval=20 \
#     task.demos_per_env=25 \
#     logging.mode=disabled

exp_name="diffusion_policy_ablation_2"
blocks=(16)
seeds=(0 1)
embed_dims=(256)

for block in ${blocks[@]}; do
    for seed in ${seeds[@]}; do
        for embed_dim in ${embed_dims[@]}; do
            # noimage
            sbatch slurm/run_rtx6000.sbatch python train.py \
                --config-name=train_prior.yaml \
                task=libero_90_hybrid \
                exp_name=${exp_name} \
                variant_name=no_image_block_${block}_ed_${embed_dim} \
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
                algo.embed_dim=${embed_dim} \
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
                variant_name=no_lang_block_${block}_ed_${embed_dim} \
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
                algo.embed_dim=${embed_dim} \
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

            # only pos
            sbatch slurm/run_rtx6000.sbatch python train.py \
                --config-name=train_prior.yaml \
                task=libero_90_hybrid \
                exp_name=${exp_name} \
                variant_name=no_image_no_lang_block_${block}_ed_${embed_dim} \
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
                algo.embed_dim=${embed_dim} \
                +algo.encoder.do_lang=false \
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
            
            # no pos
            # if this does well im gonna kms
            sbatch slurm/run_rtx6000.sbatch python train.py \
                --config-name=train_prior.yaml \
                task=libero_90_hybrid \
                exp_name=${exp_name} \
                variant_name=no_pos_block_${block}_ed_${embed_dim} \
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
                algo.embed_dim=${embed_dim} \
                +algo.encoder.do_pos=false \
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

            # world frame
            sbatch slurm/run_rtx6000.sbatch python train.py \
                --config-name=train_prior.yaml \
                task=libero_90_hybrid \
                exp_name=${exp_name} \
                variant_name=world_frame_block_${block}_ed_${embed_dim} \
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
                algo.embed_dim=${embed_dim} \
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

