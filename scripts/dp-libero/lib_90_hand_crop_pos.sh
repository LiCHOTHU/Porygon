# python train.py \
#     --config-name=train_debug.yaml \
#     task=libero_10_hybrid \
#     algo=baku \
#     algo/encoder=hybrid  \
#     algo.chunk_size=10 \
#     algo.encoder.backbone_type=clip \
#     algo.encoder.finetune=false \
#     algo.encoder.num_points=1024 \
#     +algo.policy.eecf=true \
#     algo/aug=color_jitter \
#     ~task.shape_meta.observation.lowdim \
#     +task.shape_meta.observation.lowdim={robot0_gripper_qpos:2} \
#     rollout.interval=20 \
#     task.demos_per_env=50 

exp_name="final"
blocks=(8)
# seeds=(0 1)
seeds=(2 3 4)
backbones=("clip")
finetunes=(false)
downsamples=(512)


for block in ${blocks[@]}; do
    for seed in ${seeds[@]}; do
        for backbone in ${backbones[@]}; do
            for finetune in ${finetunes[@]}; do
                for downsample in ${downsamples[@]}; do
                    # ds to 1024
                    sbatch slurm/run_l40s.sbatch python train.py \
                        --config-name=train_prior.yaml \
                        task=libero_90_hybrid \
                        exp_name=${exp_name} \
                        variant_name=proprio_tight_hand_crop_ds_${downsample}_${backbone}_ft_${finetune}_block_${block} \
                        pace_copy=true pace_tmp_dir=$TMPDIR \
                        algo=diffusion_policy \
                        algo/encoder=hybrid  \
                        algo/aug=image \
                        algo.chunk_size=${block} \
                        algo.encoder.backbone_type=${backbone} \
                        algo.encoder.finetune=${finetune} \
                        algo.encoder.num_points=${downsample} \
                        +algo.policy.eecf=true \
                        algo.encoder.do_crop=true \
                        algo.encoder.do_hand_crop=true \
                        +algo.encoder.tight_crop=true \
                        training.save_interval=10 \
                        train_dataloader.num_workers=4 \
                        rollout.interval=100 \
                        task.demos_per_env=50 \
                        training.n_epochs=101 \
                        seed=${seed}
                done
            done
        done
    done
done


