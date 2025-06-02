python train.py --config-name=train_debug.yaml \
    task=libero_10_hybrid \
    algo=dit_head_2 \
    algo.chunk_size=15 \
    algo.policy.hidden_dim=256 \
    algo.policy.num_layers=4 \
    training.save_interval=10 \
    train_dataloader.num_workers=4 \
    rollout.interval=10 \
    task.demos_per_env=50 \
    training.n_epochs=101



python export_videos.py \
    task=libero_90_hybrid \
    algo=da_style_head \
    checkpoint_path=data/checkpoints/da_head_rot_aug.pth \
    +task.env_factory.camera_pose_variations=large

python train.py --config-name=train_debug.yaml \
    task=libero_10_hybrid \
    algo=dit_head \
    algo.chunk_size=15 \
    algo.policy.hidden_dim=256 \
    algo.policy.num_layers=4 \
    algo.policy.num_keys=1 \
    algo.obs_eecf=true \
    algo.act_eecf=false \
    algo.abs_action=false \
    training.save_interval=10 \
    train_dataloader.num_workers=4 \
    rollout.interval=1 \
    task.demos_per_env=50 \
    training.n_epochs=101 \
    training.cut=10

python train.py --config-name=train_debug.yaml \
    task=libero_10_hybrid \
    algo=dit_head_2 \
    algo.chunk_size=15 \
    algo.policy.hidden_dim=256 \
    algo.policy.num_layers=4 \
    algo.obs_eecf=true \
    algo.act_eecf=false \
    algo.abs_action=false \
    training.save_interval=10 \
    train_dataloader.num_workers=4 \
    rollout.interval=1 \
    task.demos_per_env=50 \
    training.n_epochs=101 \
    training.cut=10

python train.py --config-name=train_debug.yaml \
    task=libero_10_hybrid \
    algo=dit_head_2 \
    algo.chunk_size=15 \
    algo.policy.hidden_dim=256 \
    algo.policy.num_layers=4 \
    algo.obs_eecf=true \
    algo.act_eecf=false \
    algo.abs_action=true \
    training.save_interval=10 \
    train_dataloader.num_workers=4 \
    rollout.interval=10 \
    task.demos_per_env=50 \
    training.n_epochs=101

python train.py --config-name=train_debug.yaml \
    task=libero_10_hybrid \
    algo=da_style_head \
    algo.chunk_size=15 \
    algo.obs_eecf=false \
    algo.act_eecf=false \
    training.save_interval=10 \
    train_dataloader.num_workers=4 \
    rollout.interval=10 \
    task.demos_per_env=50 \
    training.n_epochs=101

python train.py \
        --config-name=train_prior.yaml \
        pace_copy=true  \
        task=libero_90_hybrid_wrist_cam \
        exp_name=${exp_name} \
        variant_name=adapt3r_no_scene_cam_no_proprio \
        ~task.shape_meta.observation.lowdim \
        +task.shape_meta.observation.lowdim={robot0_gripper_qpos:2} \
        algo=baku \
        algo/encoder=hybrid  \
        algo/aug=image \
        algo.chunk_size=10 \
        +algo.policy.eecf=true \
        algo.encoder.do_hand_crop=true \
        +algo.encoder.tight_crop=true \
        training.save_interval=20 \
        train_dataloader.num_workers=4 \
        rollout.interval=200 \
        task.demos_per_env=50 \
        training.n_epochs=101 \
        seed=${seed}

python export_videos.py \
    task=libero_90_hybrid \
    algo=da_style_head \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/vast/quest_v0/experiments/libero/libero_90/head_sweep_3/da_style_head_no_eecf/stage_1/ \
    rollout.max_episode_length=100


python train.py \
    --config-name=train_debug.yaml \
    task=libero_10_hybrid \
    algo=da_style_head \
    algo.chunk_size=15 \
    algo.obs_eecf=false \
    algo.act_eecf=false \
    training.save_interval=10 \
    train_dataloader.num_workers=4 \
    rollout.interval=10 \
    task.demos_per_env=50 \
    training.n_epochs=101
