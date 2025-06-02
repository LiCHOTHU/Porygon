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
tasks=("libero_90_hybrid")

for block in ${blocks[@]}; do
    for seed in ${seeds[@]}; do
        for task in ${tasks[@]}; do
            sbatch slurm/run_rtx6000.sbatch python train.py \
                --config-name=train_prior.yaml \
                task=${task} \
                exp_name=diff_po_1 \
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
                rollout.interval=20 \
                task.demos_per_env=25 \
                seed=${seed}
        done
    done
done


