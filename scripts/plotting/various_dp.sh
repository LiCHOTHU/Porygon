algo="diffusion_policy"


python scripts/make_bar_plot.py \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/dp3_no_joint_demos_50_block_8 \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/dp3_no_joint_demos_50_block_16 \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/idp3_no_joint_demos_50_block_8 \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/idp3_no_joint_demos_50_block_16 \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/rgb_no_joint_demos_50_block_8 \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/rgb_no_joint_demos_50_block_16 \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/rgbd_no_joint_demos_50_block_8 \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/rgbd_no_joint_demos_50_block_16 \
    --labels  \
        dp3_10 \
        dp3_15 \
        idp3_10 \
        idp3_15 \
        rgb_10 \
        rgb_15 \
        rgbd_10 \
        rgbd_15 \
    --fname ${algo}-final --xtick-rotation 90 

python scripts/make_bar_plot.py \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/dp3_no_joint_demos_50_block_8_UR5e \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/dp3_no_joint_demos_50_block_16_UR5e \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/idp3_no_joint_demos_50_block_8_UR5e \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/idp3_no_joint_demos_50_block_16_UR5e \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/rgb_no_joint_demos_50_block_8_UR5e \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/rgb_no_joint_demos_50_block_16_UR5e \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/rgbd_no_joint_demos_50_block_8_UR5e \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/rgbd_no_joint_demos_50_block_16_UR5e \
    --labels  \
        dp3_10 \
        dp3_15 \
        idp3_10 \
        idp3_15 \
        rgb_10 \
        rgb_15 \
        rgbd_10 \
        rgbd_15 \
    --fname ${algo}-final-UR5e --xtick-rotation 90 

python scripts/make_bar_plot.py \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/dp3_no_joint_demos_50_block_8_Kinova3 \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/dp3_no_joint_demos_50_block_16_Kinova3 \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/idp3_no_joint_demos_50_block_8_Kinova3 \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/idp3_no_joint_demos_50_block_16_Kinova3 \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/rgb_no_joint_demos_50_block_8_Kinova3 \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/rgb_no_joint_demos_50_block_16_Kinova3 \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/rgbd_no_joint_demos_50_block_8_Kinova3 \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/rgbd_no_joint_demos_50_block_16_Kinova3 \
    --labels  \
        dp3_10 \
        dp3_15 \
        idp3_10 \
        idp3_15 \
        rgb_10 \
        rgb_15 \
        rgbd_10 \
        rgbd_15 \
    --fname ${algo}-final-Kinova3 --xtick-rotation 90 

python scripts/make_bar_plot.py \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/dp3_no_joint_demos_50_block_8_Sawyer \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/dp3_no_joint_demos_50_block_16_Sawyer \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/idp3_no_joint_demos_50_block_8_Sawyer \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/idp3_no_joint_demos_50_block_16_Sawyer \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/rgb_no_joint_demos_50_block_8_Sawyer \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/rgb_no_joint_demos_50_block_16_Sawyer \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/rgbd_no_joint_demos_50_block_8_Sawyer \
        ~/shared/upce_models/ice_backup/experiments/evaluate/libero/libero_90/${algo}/baselines/rgbd_no_joint_demos_50_block_16_Sawyer \
    --labels  \
        dp3_10 \
        dp3_15 \
        idp3_10 \
        idp3_15 \
        rgb_10 \
        rgb_15 \
        rgbd_10 \
        rgbd_15 \
    --fname ${algo}-final-Sawyer --xtick-rotation 90 