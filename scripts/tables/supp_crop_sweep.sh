# In dist multitask
python scripts/make_table.py \
    --prefix /storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready/evaluate \
    --data-dirs \
        libero/libero_90/baku/multitask/proprio_no_world_crop \
        libero/libero_90/baku/multitask/proprio_loose_world_crop \
        libero/libero_90/baku/multitask/no_hand_crop \
        libero/libero_90/baku/multitask/proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/robot_change/UR5e_proprio_no_world_crop \
        libero/libero_90/baku/robot_change/UR5e_proprio_loose_world_crop \
        libero/libero_90/baku/robot_change/UR5e_no_hand_crop \
        libero/libero_90/baku/robot_change/UR5e_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/robot_change/Kinova3_proprio_no_world_crop \
        libero/libero_90/baku/robot_change/Kinova3_proprio_loose_world_crop \
        libero/libero_90/baku/robot_change/Kinova3_no_hand_crop \
        libero/libero_90/baku/robot_change/Kinova3_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/robot_change/IIWA_proprio_no_world_crop \
        libero/libero_90/baku/robot_change/IIWA_proprio_loose_world_crop \
        libero/libero_90/baku/robot_change/IIWA_no_hand_crop \
        libero/libero_90/baku/robot_change/IIWA_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/camera_change/small_proprio_no_world_crop \
        libero/libero_90/baku/camera_change/small_proprio_loose_world_crop \
        libero/libero_90/baku/camera_change/small_no_hand_crop \
        libero/libero_90/baku/camera_change/small_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/camera_change/medium_proprio_no_world_crop \
        libero/libero_90/baku/camera_change/medium_proprio_loose_world_crop \
        libero/libero_90/baku/camera_change/medium_no_hand_crop \
        libero/libero_90/baku/camera_change/medium_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/camera_change/large_proprio_no_world_crop \
        libero/libero_90/baku/camera_change/large_proprio_loose_world_crop \
        libero/libero_90/baku/camera_change/large_no_hand_crop \
        libero/libero_90/baku/camera_change/large_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
    --row-names \
        Orig. UR5e Kinova3 IIWA Small Medium Large \
    --column-names \
        None Loose "Tight, No EE" Tight \
    --column-config lcccc \
    --midrules 0 3 \
    --max-axis 1 \
    --h 7 --w 4


