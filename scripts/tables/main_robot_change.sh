
# Robot change
python scripts/make_table.py \
    --prefix /storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready/evaluate \
    --data-dirs \
        libero/libero_90/act/robot_change/rgb_UR5e \
        libero/libero_90/act/robot_change/rgb_Kinova3 \
        libero/libero_90/act/robot_change/rgb_IIWA \
        libero/libero_90/act/robot_change/rgbd_UR5e \
        libero/libero_90/act/robot_change/rgbd_Kinova3 \
        libero/libero_90/act/robot_change/rgbd_IIWA \
        libero/libero_90/act/robot_change/dp3_UR5e \
        libero/libero_90/act/robot_change/dp3_Kinova3 \
        libero/libero_90/act/robot_change/dp3_IIWA \
        libero/libero_90/act/robot_change/idp3_UR5e \
        libero/libero_90/act/robot_change/idp3_Kinova3 \
        libero/libero_90/act/robot_change/idp3_IIWA \
        libero/libero_90/act/robot_change/UR5e_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/act/robot_change/Kinova3_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/act/robot_change/IIWA_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/diffusion_policy/robot_change/rgb_UR5e \
        libero/libero_90/diffusion_policy/robot_change/rgb_Kinova3 \
        libero/libero_90/diffusion_policy/robot_change/rgb_IIWA \
        libero/libero_90/diffusion_policy/robot_change/rgbd_UR5e \
        libero/libero_90/diffusion_policy/robot_change/rgbd_Kinova3 \
        libero/libero_90/diffusion_policy/robot_change/rgbd_IIWA \
        libero/libero_90/diffusion_policy/robot_change/dp3_UR5e \
        libero/libero_90/diffusion_policy/robot_change/dp3_Kinova3 \
        libero/libero_90/diffusion_policy/robot_change/dp3_IIWA \
        libero/libero_90/diffusion_policy/robot_change/idp3_UR5e \
        libero/libero_90/diffusion_policy/robot_change/idp3_Kinova3 \
        libero/libero_90/diffusion_policy/robot_change/idp3_IIWA \
        libero/libero_90/diffuser_actor/robot_change/UR5e_block_16 \
        libero/libero_90/diffuser_actor/robot_change/Kinova3_block_16 \
        libero/libero_90/diffuser_actor/robot_change/IIWA_block_16 \
        libero/libero_90/diffusion_policy/robot_change/UR5e_proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        libero/libero_90/diffusion_policy/robot_change/Kinova3_proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        libero/libero_90/diffusion_policy/robot_change/IIWA_proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        libero/libero_90/baku/robot_change/rgb_UR5e \
        libero/libero_90/baku/robot_change/rgb_Kinova3 \
        libero/libero_90/baku/robot_change/rgb_IIWA \
        libero/libero_90/baku/robot_change/rgbd_UR5e \
        libero/libero_90/baku/robot_change/rgbd_Kinova3 \
        libero/libero_90/baku/robot_change/rgbd_IIWA \
        libero/libero_90/baku/robot_change/dp3_UR5e \
        libero/libero_90/baku/robot_change/dp3_Kinova3 \
        libero/libero_90/baku/robot_change/dp3_IIWA \
        libero/libero_90/baku/robot_change/idp3_UR5e \
        libero/libero_90/baku/robot_change/idp3_Kinova3 \
        libero/libero_90/baku/robot_change/idp3_IIWA \
        libero/libero_90/baku/robot_change/UR5e_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/robot_change/Kinova3_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/robot_change/IIWA_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
    --row-names \
        "ACT + RGB" \
        "ACT + RGBD" \
        "ACT + DP3" \
        "ACT + iDP3" \
        "ACT + \algabbr" \
        "DP + RGB" \
        "DP + RGBD" \
        "DP + DP3" \
        "DP + iDP3" \
        "3D Diffuser-Actor" \
        "DP + \algabbr" \
        "BAKU + RGB" \
        "BAKU + RGBD" \
        "BAKU + DP3" \
        "BAKU + iDP3" \
        "BAKU + \algabbr" \
    --column-names \
        Algorithm UR5e Kinova3 IIWA \
    --column-config lccc \
    --chunk-sizes 5 6 5 \
    --midrules 4 10 \
    --h 16 --w 3

