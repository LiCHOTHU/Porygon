# In dist multitask
python scripts/make_table.py \
    --prefix /storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready/evaluate \
    --data-dirs \
        libero/libero_90/baku/multitask/pos_based_fps \
        libero/libero_90/baku/multitask/proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/robot_change/UR5e_pos_based_fps \
        libero/libero_90/baku/robot_change/UR5e_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/robot_change/Kinova3_pos_based_fps \
        libero/libero_90/baku/robot_change/Kinova3_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/robot_change/IIWA_pos_based_fps \
        libero/libero_90/baku/robot_change/IIWA_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/camera_change/small_pos_based_fps \
        libero/libero_90/baku/camera_change/small_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/camera_change/medium_pos_based_fps \
        libero/libero_90/baku/camera_change/medium_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/camera_change/large_pos_based_fps \
        libero/libero_90/baku/camera_change/large_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
    --row-names \
        Orig. UR5e Kinova3 IIWA Small Medium Large \
    --column-names \
        "Position-Based FPS" "Feature-Based FPS" \
    --column-config lcccc \
    --midrules 0 3 \
    --max-axis 1 \
    --h 7 --w 2


