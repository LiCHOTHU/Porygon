python export_videos.py \
    task=mimicgen_rgb_base \
    task.task_name=coffee \
    algo=diffusion_policy \
    +overrides.temporal_agg=false \
    +overrides.action_horizon=3 \
    task.robot=UR5e \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready/mimicgen/coffee/diffusion_policy/mimicgen_corl/rgb/0/stage_1/checkpoint.pth


python export_videos.py \
    task=mimicgen_hybrid_base \
    task.task_name=square \
    algo=diffusion_policy \
    +overrides.temporal_agg=false \
    +overrides.action_horizon=2 \
    task.robot=UR5e \
    seed=0 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready/mimicgen/square/diffusion_policy/mimicgen_corl/adapt3r_good/0/stage_1/checkpoint.pth


python export_videos.py \
    task=mimicgen_hybrid_base \
    task.task_name=coffee \
    algo=diffusion_policy \
    +overrides.temporal_agg=false \
    +overrides.action_horizon=2 \
    +overrides.encoder={use_old_hand_frame_crop:true} \
    seed=0 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready/mimicgen/coffee/diffusion_policy/mimicgen_corl/adapt3r_good/0/stage_1/checkpoint.pth \
    rollout.max_episode_length=100
