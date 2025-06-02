algo=diffusion_policy
exp_name=${algo}_no_crop_3
blocks=(16)
seeds=(0 1)
downsample_modes=("feat")

for block in ${blocks[@]}; do
    for seed in ${seeds[@]}; do
        for downsample_mode in ${downsample_modes[@]}; do
            # DP3 baseline
            sbatch slurm/run_v100.sbatch python train.py \
                --config-name=train_prior.yaml \
                task=libero_90_hybrid \
                exp_name=${exp_name} \
                variant_name=dp3_crop_block_${block} \
                algo=${algo} \
                algo/encoder=hybrid  \
                algo.chunk_size=${block} \
                algo.policy.temporal_agg=true \
                algo/aug=identity \
                algo.encoder.hand_frame=false \
                algo.encoder.do_crop=true \
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
            
            
        done
    done
done

