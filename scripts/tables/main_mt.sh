
# In dist multitask
python scripts/make_table.py \
    --prefix /storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready/evaluate \
    --data-dirs \
        libero/libero_90/act/multitask/rgb \
        metaworld/MT50/act/camera_ready/rgb_no_joint_demos_10_block_15 \
        libero/libero_90/act/multitask/rgbd \
        metaworld/MT50/act/camera_ready/rgbd_no_joint_demos_10_block_15 \
        libero/libero_90/act/multitask/dp3 \
        metaworld/MT50/act/camera_ready/dp3_no_joint_demos_10_block_15 \
        libero/libero_90/act/multitask/idp3 \
        metaworld/MT50/act/camera_ready/idp3_no_joint_demos_10_block_15 \
        libero/libero_90/act/multitask/proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        metaworld/MT50/act/camera_ready/proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/diffusion_policy/multitask/rgb \
        metaworld/MT50/diffusion_policy/camera_ready/rgb_no_joint_demos_10_block_16 \
        libero/libero_90/diffusion_policy/multitask/rgbd \
        metaworld/MT50/diffusion_policy/camera_ready/rgbd_no_joint_demos_10_block_16 \
        libero/libero_90/diffusion_policy/multitask/dp3 \
        metaworld/MT50/diffusion_policy/camera_ready/dp3_no_joint_demos_10_block_16 \
        libero/libero_90/diffusion_policy/multitask/idp3 \
        metaworld/MT50/diffusion_policy/camera_ready/idp3_no_joint_demos_10_block_16 \
        libero/libero_90/diffuser_actor/multitask/block_16 \
        - \
        libero/libero_90/diffusion_policy/multitask/proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        metaworld/MT50/diffusion_policy/camera_ready/proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        libero/libero_90/baku/multitask/rgb \
        metaworld/MT50/baku/camera_ready/rgb_no_joint_demos_10_block_15 \
        libero/libero_90/baku/multitask/rgbd \
        metaworld/MT50/baku/camera_ready/rgbd_no_joint_demos_10_block_15 \
        libero/libero_90/baku/multitask/dp3 \
        metaworld/MT50/baku/camera_ready/dp3_no_joint_demos_10_block_15 \
        libero/libero_90/baku/multitask/idp3 \
        metaworld/MT50/baku/camera_ready/idp3_no_joint_demos_10_block_15 \
        libero/libero_90/baku/multitask/proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        metaworld/MT50/baku/camera_ready/proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
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
        Algorithm LIBERO-90 "MetaWorld" \
    --column-config lcccc \
    --chunk-sizes 5 6 5 \
    --midrules 4 10 \
    --h 16 --w 2


