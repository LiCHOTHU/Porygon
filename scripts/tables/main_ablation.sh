
# Camera change
python scripts/make_table.py \
    --prefix /storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready/evaluate \
    --data-dirs \
        libero/libero_90/baku/multitask/no_eecf \
        libero/libero_90/baku/robot_change/UR5e_no_eecf \
        libero/libero_90/baku/robot_change/Kinova3_no_eecf \
        libero/libero_90/baku/robot_change/IIWA_no_eecf \
        libero/libero_90/baku/camera_change/small_no_eecf \
        libero/libero_90/baku/camera_change/medium_no_eecf \
        libero/libero_90/baku/camera_change/large_no_eecf \
        libero/libero_90/baku/multitask/no_feats \
        libero/libero_90/baku/robot_change/UR5e_no_feats \
        libero/libero_90/baku/robot_change/Kinova3_no_feats \
        libero/libero_90/baku/robot_change/IIWA_no_feats \
        libero/libero_90/baku/camera_change/small_no_feats \
        libero/libero_90/baku/camera_change/medium_no_feats \
        libero/libero_90/baku/camera_change/large_no_feats \
        libero/libero_90/baku/multitask/no_feats_yes_rgb \
        libero/libero_90/baku/robot_change/UR5e_no_feats_yes_rgb \
        libero/libero_90/baku/robot_change/Kinova3_no_feats_yes_rgb \
        libero/libero_90/baku/robot_change/IIWA_no_feats_yes_rgb \
        libero/libero_90/baku/camera_change/small_no_feats_yes_rgb \
        libero/libero_90/baku/camera_change/medium_no_feats_yes_rgb \
        libero/libero_90/baku/camera_change/large_no_feats_yes_rgb \
        libero/libero_90/baku/multitask/no_hand_crop \
        libero/libero_90/baku/robot_change/UR5e_no_hand_crop \
        libero/libero_90/baku/robot_change/Kinova3_no_hand_crop \
        libero/libero_90/baku/robot_change/IIWA_no_hand_crop \
        libero/libero_90/baku/camera_change/small_no_hand_crop \
        libero/libero_90/baku/camera_change/medium_no_hand_crop \
        libero/libero_90/baku/camera_change/large_no_hand_crop \
        libero/libero_90/baku/multitask/no_lang \
        libero/libero_90/baku/robot_change/UR5e_no_lang \
        libero/libero_90/baku/robot_change/Kinova3_no_lang \
        libero/libero_90/baku/robot_change/IIWA_no_lang \
        libero/libero_90/baku/camera_change/small_no_lang \
        libero/libero_90/baku/camera_change/medium_no_lang \
        libero/libero_90/baku/camera_change/large_no_lang \
        libero/libero_90/baku/multitask/no_nerf_pos_emb \
        libero/libero_90/baku/robot_change/UR5e_no_nerf_pos_emb \
        libero/libero_90/baku/robot_change/Kinova3_no_nerf_pos_emb \
        libero/libero_90/baku/robot_change/IIWA_no_nerf_pos_emb \
        libero/libero_90/baku/camera_change/small_no_nerf_pos_emb \
        libero/libero_90/baku/camera_change/medium_no_nerf_pos_emb \
        libero/libero_90/baku/camera_change/large_no_nerf_pos_emb \
        libero/libero_90/baku/multitask/pos_based_fps \
        libero/libero_90/baku/robot_change/UR5e_pos_based_fps \
        libero/libero_90/baku/robot_change/Kinova3_pos_based_fps \
        libero/libero_90/baku/robot_change/IIWA_pos_based_fps \
        libero/libero_90/baku/camera_change/small_pos_based_fps \
        libero/libero_90/baku/camera_change/medium_pos_based_fps \
        libero/libero_90/baku/camera_change/large_pos_based_fps \
        libero/libero_90/baku/multitask/dp3_extractor \
        libero/libero_90/baku/robot_change/UR5e_dp3_extractor \
        libero/libero_90/baku/robot_change/Kinova3_dp3_extractor \
        libero/libero_90/baku/robot_change/IIWA_dp3_extractor \
        libero/libero_90/baku/camera_change/small_dp3_extractor \
        libero/libero_90/baku/camera_change/medium_dp3_extractor \
        libero/libero_90/baku/camera_change/large_dp3_extractor \
        libero/libero_90/baku/multitask/proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/robot_change/UR5e_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/robot_change/Kinova3_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/robot_change/IIWA_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/camera_change/small_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/camera_change/medium_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        libero/libero_90/baku/camera_change/large_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
    --row-names \
        "No EECF" \
        "No Image Features" \
        "RGB Point Cloud" \
        "No EE Crop" \
        "No Lang. Features" \
        "No Positional Encoding" \
        "Position-Based FPS" \
        "No Attention" \
        "Ours" \
    --column-names \
        "" Orig. UR5e Kinova3 IIWA Small Medium Large \
    --column-config "l|c|ccc|ccc" \
    --h 9 --w 7

