
# In dist multitask
python scripts/make_line_plot.py \
    --prefix /storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready/evaluate/libero/libero_90 \
    --data-dirs \
        baku/multitask/rgb \
        baku/camera_change/cs_0.2_rad_rgb \
        baku/camera_change/cs_0.4_rad_rgb \
        baku/camera_change/cs_0.6_rad_rgb \
        baku/camera_change/cs_0.8_rad_rgb \
        baku/camera_change/cs_1.0_rad_rgb \
        baku/camera_change/cs_1.2_rad_rgb \
        baku/camera_change/cs_1.4_rad_rgb \
        baku/camera_change/cs_1.6_rad_rgb \
        baku/camera_change/cs_1.8_rad_rgb \
        baku/camera_change/cs_2.0_rad_rgb \
        baku/multitask/rgbd \
        baku/camera_change/cs_0.2_rad_rgbd \
        baku/camera_change/cs_0.4_rad_rgbd \
        baku/camera_change/cs_0.6_rad_rgbd \
        baku/camera_change/cs_0.8_rad_rgbd \
        baku/camera_change/cs_1.0_rad_rgbd \
        baku/camera_change/cs_1.2_rad_rgbd \
        baku/camera_change/cs_1.4_rad_rgbd \
        baku/camera_change/cs_1.6_rad_rgbd \
        baku/camera_change/cs_1.8_rad_rgbd \
        baku/camera_change/cs_2.0_rad_rgbd \
        baku/multitask/dp3 \
        baku/camera_change/cs_0.2_rad_dp3 \
        baku/camera_change/cs_0.4_rad_dp3 \
        baku/camera_change/cs_0.6_rad_dp3 \
        baku/camera_change/cs_0.8_rad_dp3 \
        baku/camera_change/cs_1.0_rad_dp3 \
        baku/camera_change/cs_1.2_rad_dp3 \
        baku/camera_change/cs_1.4_rad_dp3 \
        baku/camera_change/cs_1.6_rad_dp3 \
        baku/camera_change/cs_1.8_rad_dp3 \
        baku/camera_change/cs_2.0_rad_dp3 \
        diffuser_actor/multitask/block_16 \
        diffuser_actor/camera_change/0.2_block_16 \
        diffuser_actor/camera_change/0.4_block_16 \
        diffuser_actor/camera_change/0.6_block_16 \
        diffuser_actor/camera_change/0.8_block_16 \
        diffuser_actor/camera_change/1.0_block_16 \
        diffuser_actor/camera_change/1.2_block_16 \
        diffuser_actor/camera_change/1.4_block_16 \
        diffuser_actor/camera_change/1.6_block_16 \
        diffuser_actor/camera_change/1.8_block_16 \
        diffuser_actor/camera_change/2.0_block_16 \
        baku/multitask/proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        baku/camera_change_final/0.2_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        baku/camera_change_final/0.4_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        baku/camera_change_final/0.6_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        baku/camera_change_final/0.8_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        baku/camera_change_final/1.0_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        baku/camera_change_final/1.2_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        baku/camera_change_final/1.4_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        baku/camera_change_final/1.6_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        baku/camera_change_final/1.8_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        baku/camera_change_final/2.0_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
    --line-labels RGB RGBD DP3 3DDA Adapt3R \
    --x-labels 0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 \
    --figsize 7 3 \
    --xlabel "$\theta$" --ylabel "Success Rate" \
    --legend-outfile zplots/cs-sweep-legend.pdf \
    --title LIBERO-90 \
    --font-size-xtick 14  --font-size-ytick 12  --font-size-annot 12 --font-size-label 14 --font-size-title 16 \
    --fname zplots/libero-cs-sweep.pdf
    # --colors \
    #     lightcoral lightgreen skyblue \
    #     lightcoral lightgreen skyblue \
    #     lightcoral lightgreen skyblue \
    #     lightcoral lightgreen skyblue \
    #     lightcoral lightgreen skyblue \
    # --legend-labels "Small Change" "Medium Change" "Large Change" \
    # --legend-colors lightcoral lightgreen skyblue \
    # --group-width 0.95
        # darkred forestgreen steelblue



