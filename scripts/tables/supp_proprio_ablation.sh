
# Camera change
python scripts/make_table.py \
    --prefix /storage/home/hcoda1/1/awilcox31/shared/upce_models/final_evals \
    --data-dirs \
        libero/libero_90/act/multitask/tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/act/robot_change/UR5e_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/act/robot_change/Kinova3_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/act/robot_change/IIWA_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/act/camera_change/small_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/act/camera_change/medium_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/act/camera_change/large_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/act/multitask/proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/act/robot_change/UR5e_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/act/robot_change/Kinova3_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/act/robot_change/IIWA_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/act/camera_change/small_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/act/camera_change/medium_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/act/camera_change/large_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/diffusion_policy/multitask/tight_hand_crop_ds_512_clip_ft_false_block_8 \
        libero/libero_90/diffusion_policy/robot_change/UR5e_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        libero/libero_90/diffusion_policy/robot_change/Kinova3_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        libero/libero_90/diffusion_policy/robot_change/IIWA_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        libero/libero_90/diffusion_policy/camera_change/small_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        libero/libero_90/diffusion_policy/camera_change/medium_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        libero/libero_90/diffusion_policy/camera_change/large_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        libero/libero_90/diffusion_policy/multitask/proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        libero/libero_90/diffusion_policy/robot_change/UR5e_proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        libero/libero_90/diffusion_policy/robot_change/Kinova3_proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        libero/libero_90/diffusion_policy/robot_change/IIWA_proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        libero/libero_90/diffusion_policy/camera_change/small_proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        libero/libero_90/diffusion_policy/camera_change/medium_proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        libero/libero_90/diffusion_policy/camera_change/large_proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        libero/libero_90/baku/multitask/tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/robot_change/UR5e_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/robot_change/Kinova3_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/robot_change/IIWA_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/camera_change/small_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/camera_change/medium_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/camera_change/large_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/multitask/proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/robot_change/UR5e_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/robot_change/Kinova3_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/robot_change/IIWA_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/camera_change/small_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/camera_change/medium_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/camera_change/large_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
    --row-names \
        "ACT + \algabbr without" \
        "ACT + \algabbr with" \
        "DP + \algabbr without" \
        "DP + \algabbr with" \
        "BAKU + \algabbr without" \
        "BAKU + \algabbr with" \
    --column-names \
        Algorithm Orig. UR5e Kinova3 IIWA Small Medium Large \
    --column-config "l|c|ccc|ccc" \
    --chunk-sizes 2 2 2 \
    --midrules 1 3 \
    --h 6 --w 7

