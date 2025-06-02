exp_name="corl-push"
seeds=(0 1 2 3 4)

for seed in ${seeds[@]}; do

    # no lang
    sbatch slurm/run_l40s.sbatch python train.py \
        --config-name=train_prior.yaml \
        pace_copy=true  \
        task=libero_90_hybrid \
        exp_name=${exp_name} \
        variant_name=no_lang \
        algo=baku \
        algo/encoder=hybrid  \
        algo/aug=image \
        algo.chunk_size=10 \
        +algo.policy.eecf=true \
        algo.encoder.do_hand_crop=true \
        +algo.encoder.tight_crop=true \
        algo.encoder.do_lang=false \
        training.save_interval=20 \
        train_dataloader.num_workers=8 \
        rollout.interval=20 \
        task.demos_per_env=50 \
        training.n_epochs=101 \
        seed=${seed}

    # # no features
    # sbatch slurm/run_l40s.sbatch python train.py \
    #     --config-name=train_prior.yaml \
    #     pace_copy=true  \
    #     task=libero_90_hybrid \
    #     exp_name=${exp_name} \
    #     variant_name=no_feats \
    #     algo=baku \
    #     algo/encoder=hybrid  \
    #     algo/aug=image \
    #     algo.chunk_size=10 \
    #     +algo.policy.eecf=true \
    #     algo.encoder.do_hand_crop=true \
    #     +algo.encoder.tight_crop=true \
    #     algo.encoder.do_image=false \
    #     training.save_interval=20 \
    #     train_dataloader.num_workers=8 \
    #     rollout.interval=20 \
    #     task.demos_per_env=50 \
    #     training.n_epochs=101 \
    #     seed=${seed}

    # # no features + RGB
    # sbatch slurm/run_l40s.sbatch python train.py \
    #     --config-name=train_prior.yaml \
    #     pace_copy=true  \
    #     task=libero_90_hybrid \
    #     exp_name=${exp_name} \
    #     variant_name=no_feats_yes_rgb \
    #     algo=baku \
    #     algo/encoder=hybrid  \
    #     algo/aug=image \
    #     algo.chunk_size=10 \
    #     +algo.policy.eecf=true \
    #     algo.encoder.do_hand_crop=true \
    #     +algo.encoder.tight_crop=true \
    #     algo.encoder.do_image=false \
    #     +algo.encoder.do_rgb=true \
    #     training.save_interval=20 \
    #     train_dataloader.num_workers=8 \
    #     rollout.interval=20 \
    #     task.demos_per_env=50 \
    #     training.n_epochs=101 \
    #     seed=${seed}

    # # no EECF
    # sbatch slurm/run_l40s.sbatch python train.py \
    #     --config-name=train_prior.yaml \
    #     pace_copy=true  \
    #     task=libero_90_hybrid \
    #     exp_name=${exp_name} \
    #     variant_name=no_eecf \
    #     algo=baku \
    #     algo/encoder=hybrid  \
    #     algo/aug=image \
    #     algo.chunk_size=10 \
    #     +algo.policy.eecf=false \
    #     algo.encoder.hand_frame=false \
    #     algo.encoder.do_hand_crop=true \
    #     +algo.encoder.tight_crop=true \
    #     training.save_interval=20 \
    #     train_dataloader.num_workers=8 \
    #     rollout.interval=20 \
    #     task.demos_per_env=50 \
    #     training.n_epochs=101 \
    #     seed=${seed}

    # # no hand crop
    # sbatch slurm/run_l40s.sbatch python train.py \
    #     --config-name=train_prior.yaml \
    #     pace_copy=true  \
    #     task=libero_90_hybrid \
    #     exp_name=${exp_name} \
    #     variant_name=no_hand_crop \
    #     algo=baku \
    #     algo/encoder=hybrid  \
    #     algo/aug=image \
    #     algo.chunk_size=10 \
    #     +algo.policy.eecf=true \
    #     algo.encoder.do_hand_crop=false \
    #     +algo.encoder.tight_crop=true \
    #     training.save_interval=20 \
    #     train_dataloader.num_workers=8 \
    #     rollout.interval=20 \
    #     task.demos_per_env=50 \
    #     training.n_epochs=101 \
    #     seed=${seed}

    # # no nerf pos emb
    # sbatch slurm/run_l40s.sbatch python train.py \
    #     --config-name=train_prior.yaml \
    #     pace_copy=true  \
    #     task=libero_90_hybrid \
    #     exp_name=${exp_name} \
    #     variant_name=no_nerf_pos_emb \
    #     algo=baku \
    #     algo/encoder=hybrid  \
    #     algo/aug=image \
    #     algo.chunk_size=10 \
    #     +algo.policy.eecf=true \
    #     algo.encoder.do_hand_crop=true \
    #     +algo.encoder.tight_crop=true \
    #     algo.encoder.xyz_proj_type=none \
    #     training.save_interval=20 \
    #     train_dataloader.num_workers=8 \
    #     rollout.interval=20 \
    #     task.demos_per_env=50 \
    #     training.n_epochs=101 \
    #     seed=${seed}

    # # pos based fps
    # sbatch slurm/run_l40s.sbatch python train.py \
    #     --config-name=train_prior.yaml \
    #     pace_copy=true  \
    #     task=libero_90_hybrid \
    #     exp_name=${exp_name} \
    #     variant_name=pos_based_fps \
    #     algo=baku \
    #     algo/encoder=hybrid  \
    #     algo/aug=image \
    #     algo.chunk_size=10 \
    #     +algo.policy.eecf=true \
    #     algo.encoder.do_hand_crop=true \
    #     +algo.encoder.tight_crop=true \
    #     algo.encoder.downsample_mode=pos \
    #     training.save_interval=20 \
    #     train_dataloader.num_workers=8 \
    #     rollout.interval=20 \
    #     task.demos_per_env=50 \
    #     training.n_epochs=101 \
    #     seed=${seed}
done


