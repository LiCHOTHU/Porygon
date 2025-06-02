
# Camera change
python scripts/make_table.py \
    --prefix /storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready/evaluate \
    --data-dirs \
        libero/libero_90/act/camera_change/rgb_small \
        libero/libero_90/act/camera_change/rgb_medium \
        libero/libero_90/act/camera_change/rgb_large \
        metaworld/MT50/act/camera_ready/cam_change_rgb_no_joint_demos_10_block_15 \
        libero/libero_90/act/camera_change/rgbd_small \
        libero/libero_90/act/camera_change/rgbd_medium \
        libero/libero_90/act/camera_change/rgbd_large \
        metaworld/MT50/act/camera_ready/cam_change_rgbd_no_joint_demos_10_block_15 \
        libero/libero_90/act/camera_change/dp3_small \
        libero/libero_90/act/camera_change/dp3_medium \
        libero/libero_90/act/camera_change/dp3_large \
        metaworld/MT50/act/camera_ready/cam_change_dp3_no_joint_demos_10_block_15 \
        libero/libero_90/act/camera_change/idp3_small \
        libero/libero_90/act/camera_change/idp3_medium \
        libero/libero_90/act/camera_change/idp3_large \
        metaworld/MT50/act/camera_ready/cam_change_idp3_no_joint_demos_10_block_15 \
        libero/libero_90/act/camera_change/small_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/act/camera_change/medium_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/act/camera_change/large_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        metaworld/MT50/act/camera_ready/cam_change_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/diffusion_policy/camera_change/rgb_small \
        libero/libero_90/diffusion_policy/camera_change/rgb_medium \
        libero/libero_90/diffusion_policy/camera_change/rgb_large \
        metaworld/MT50/diffusion_policy/camera_ready/cam_change_rgb_no_joint_demos_10_block_16 \
        libero/libero_90/diffusion_policy/camera_change/rgbd_small \
        libero/libero_90/diffusion_policy/camera_change/rgbd_medium \
        libero/libero_90/diffusion_policy/camera_change/rgbd_large \
        metaworld/MT50/diffusion_policy/camera_ready/cam_change_rgbd_no_joint_demos_10_block_16 \
        libero/libero_90/diffusion_policy/camera_change/dp3_small \
        libero/libero_90/diffusion_policy/camera_change/dp3_medium \
        libero/libero_90/diffusion_policy/camera_change/dp3_large \
        metaworld/MT50/diffusion_policy/camera_ready/cam_change_dp3_no_joint_demos_10_block_16 \
        libero/libero_90/diffusion_policy/camera_change/idp3_small \
        libero/libero_90/diffusion_policy/camera_change/idp3_medium \
        libero/libero_90/diffusion_policy/camera_change/idp3_large \
        metaworld/MT50/diffusion_policy/camera_ready/cam_change_idp3_no_joint_demos_10_block_16 \
        libero/libero_90/diffuser_actor/camera_change/small_block_16 \
        libero/libero_90/diffuser_actor/camera_change/medium_block_16 \
        libero/libero_90/diffuser_actor/camera_change/large_block_16 \
        - \
        libero/libero_90/diffusion_policy/camera_change/small_proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        libero/libero_90/diffusion_policy/camera_change/medium_proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        libero/libero_90/diffusion_policy/camera_change/large_proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        metaworld/MT50/diffusion_policy/camera_ready/cam_change_proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        libero/libero_90/baku/camera_change/rgb_small \
        libero/libero_90/baku/camera_change/rgb_medium \
        libero/libero_90/baku/camera_change/rgb_large \
        metaworld/MT50/baku/camera_ready/cam_change_rgb_no_joint_demos_10_block_15 \
        libero/libero_90/baku/camera_change/rgbd_small \
        libero/libero_90/baku/camera_change/rgbd_medium \
        libero/libero_90/baku/camera_change/rgbd_large \
        metaworld/MT50/baku/camera_ready/cam_change_rgbd_no_joint_demos_10_block_15 \
        libero/libero_90/baku/camera_change/dp3_small \
        libero/libero_90/baku/camera_change/dp3_medium \
        libero/libero_90/baku/camera_change/dp3_large \
        metaworld/MT50/baku/camera_ready/cam_change_dp3_no_joint_demos_10_block_15 \
        libero/libero_90/baku/camera_change/idp3_small \
        libero/libero_90/baku/camera_change/idp3_medium \
        libero/libero_90/baku/camera_change/idp3_large \
        metaworld/MT50/baku/camera_ready/cam_change_idp3_no_joint_demos_10_block_15 \
        libero/libero_90/baku/camera_change/small_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/camera_change/medium_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/camera_change/large_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        metaworld/MT50/baku/camera_ready/cam_change_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
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
        Algorithm Small Medium Large MetaWorld \
    --column-config lcccc \
    --chunk-sizes 5 6 5 \
    --midrules 4 10 \
    --h 16 --w 4

