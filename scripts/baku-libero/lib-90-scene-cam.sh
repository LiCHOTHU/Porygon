exp_name="corl-push"
seeds=(0 1 2 3 4)
seeds=(3)
algo="baku"

for seed in ${seeds[@]}; do

    # # Adapt3R
    # sbatch slurm/run_l40s.sbatch python train.py \
    #     --config-name=train_prior.yaml \
    #     pace_copy=true  \
    #     task=libero_90_hybrid_scene_cam \
    #     exp_name=${exp_name} \
    #     variant_name=adapt3r_scene_only \
    #     algo=baku \
    #     algo/encoder=hybrid  \
    #     algo.chunk_size=10 \
    #     algo.encoder.do_hand_crop=false \
    #     algo.encoder.hand_frame=false \
    #     training.save_interval=20 \
    #     train_dataloader.num_workers=4 \
    #     rollout.interval=100 \
    #     task.demos_per_env=50 \
    #     training.n_epochs=101 \
    #     seed=${seed}

    # # RGB
    # sbatch slurm/run_l40s.sbatch python train.py \
    #     --config-name=train_prior.yaml \
    #     pace_copy=true  \
    #     task=libero_90_rgb_scene_cam \
    #     exp_name=${exp_name} \
    #     variant_name=rgb_scene_only \
    #     algo=${algo} \
    #     algo.chunk_size=15 \
    #     training.save_interval=20 \
    #     train_dataloader.num_workers=4 \
    #     rollout.interval=100 \
    #     task.demos_per_env=50 \
    #     training.n_epochs=101 \
    #     seed=${seed}

    # # RGBD
    # sbatch slurm/run_l40s.sbatch python train.py \
    #     --config-name=train_prior.yaml \
    #     pace_copy=true  \
    #     task=libero_90_rgbd_scene_cam \
    #     exp_name=${exp_name} \
    #     variant_name=rgbd_scene_only \
    #     algo=${algo} \
    #     algo.chunk_size=15 \
    #     training.save_interval=20 \
    #     train_dataloader.num_workers=4 \
    #     rollout.interval=100 \
    #     task.demos_per_env=50 \
    #     training.n_epochs=101 \
    #     seed=${seed}

    # # DP3
    # sbatch slurm/run_l40s.sbatch python train.py \
    #     --config-name=train_prior.yaml \
    #     pace_copy=true  \
    #     task=libero_90_hybrid_scene_cam \
    #     exp_name=${exp_name} \
    #     variant_name=dp3_scene_only \
    #     algo=${algo} \
    #     algo/encoder=hybrid_dp3 \
    #     algo.chunk_size=15 \
    #     training.save_interval=20 \
    #     train_dataloader.num_workers=4 \
    #     rollout.interval=100 \
    #     task.demos_per_env=50 \
    #     training.n_epochs=101 \
    #     seed=${seed}

    # 3DDA
    sbatch slurm/run_l40s.sbatch python train.py \
        --config-name=train_prior.yaml \
        task=libero_90_hybrid_scene_cam \
        exp_name=${exp_name} \
        variant_name=3dda_scene_only \
        algo=diffuser_actor \
        algo/encoder=hybrid  \
        algo.chunk_size=16 \
        train_dataloader.num_workers=4 \
        training.n_epochs=100 \
        rollout.interval=100 \
        pace_copy=true  \
        seed=${seed}

done


# python train.py \
#     --config-name=train_debug.yaml \
#     task=libero_90_hybrid_scene_cam \
#     algo=diffuser_actor \
#     algo/encoder=hybrid  \
#     algo.chunk_size=16 \
#     train_dataloader.num_workers=4 \
#     training.n_epochs=100 \
#     rollout.interval=100 \
#     pace_copy=true 


# python train.py \
#     --config-name=train_debug.yaml \
#     pace_copy=true  \
#     task=libero_90_hybrid_scene_cam \
#     algo=baku \
#     algo/encoder=hybrid_dp3 \
#     algo.chunk_size=15 \
#     training.save_interval=20 \
#     train_dataloader.num_workers=4 \
#     rollout.interval=100 \
#     task.demos_per_env=50 \
#     training.n_epochs=101


# python train.py \
#     --config-name=train_debug.yaml \
#     pace_copy=true  \
#     task=libero_90_hybrid_scene_cam \
#     algo=baku \
#     algo/encoder=hybrid  \
#     algo.chunk_size=10 \
#     algo.encoder.do_hand_crop=false \
#     algo.encoder.hand_frame=false \
#     training.save_interval=20 \
#     train_dataloader.num_workers=4 \
#     rollout.interval=100 \
#     task.demos_per_env=50 \
#     training.n_epochs=101
