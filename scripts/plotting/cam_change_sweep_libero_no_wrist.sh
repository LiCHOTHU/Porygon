
# In dist multitask
python scripts/make_line_plot.py \
    --prefix /storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready/evaluate/libero/libero_90 \
    --data-dirs \
        baku/multitask/rgb_scene_only \
        baku/camera_change/cs_0.2_rad_rgb_scene_only \
        baku/camera_change/cs_0.4_rad_rgb_scene_only \
        baku/camera_change/cs_0.6_rad_rgb_scene_only \
        baku/camera_change/cs_0.8_rad_rgb_scene_only \
        baku/camera_change/cs_1.0_rad_rgb_scene_only \
        baku/camera_change/cs_1.2_rad_rgb_scene_only \
        baku/camera_change/cs_1.4_rad_rgb_scene_only \
        baku/camera_change/cs_1.6_rad_rgb_scene_only \
        baku/camera_change/cs_1.8_rad_rgb_scene_only \
        baku/camera_change/cs_2.0_rad_rgb_scene_only \
        baku/multitask/rgbd_scene_only \
        baku/camera_change/cs_0.2_rad_rgbd_scene_only \
        baku/camera_change/cs_0.4_rad_rgbd_scene_only \
        baku/camera_change/cs_0.6_rad_rgbd_scene_only \
        baku/camera_change/cs_0.8_rad_rgbd_scene_only \
        baku/camera_change/cs_1.0_rad_rgbd_scene_only \
        baku/camera_change/cs_1.2_rad_rgbd_scene_only \
        baku/camera_change/cs_1.4_rad_rgbd_scene_only \
        baku/camera_change/cs_1.6_rad_rgbd_scene_only \
        baku/camera_change/cs_1.8_rad_rgbd_scene_only \
        baku/camera_change/cs_2.0_rad_rgbd_scene_only \
        baku/multitask/dp3_scene_only \
        baku/camera_change/cs_0.2_rad_dp3_scene_only \
        baku/camera_change/cs_0.4_rad_dp3_scene_only \
        baku/camera_change/cs_0.6_rad_dp3_scene_only \
        baku/camera_change/cs_0.8_rad_dp3_scene_only \
        baku/camera_change/cs_1.0_rad_dp3_scene_only \
        baku/camera_change/cs_1.2_rad_dp3_scene_only \
        baku/camera_change/cs_1.4_rad_dp3_scene_only \
        baku/camera_change/cs_1.6_rad_dp3_scene_only \
        baku/camera_change/cs_1.8_rad_dp3_scene_only \
        baku/camera_change/cs_2.0_rad_dp3_scene_only \
        baku/multitask/adapt3r_scene_only \
        baku/camera_change/cs_0.2_rad_adapt3r_scene_only \
        baku/camera_change/cs_0.4_rad_adapt3r_scene_only \
        baku/camera_change/cs_0.6_rad_adapt3r_scene_only \
        baku/camera_change/cs_0.8_rad_adapt3r_scene_only \
        baku/camera_change/cs_1.0_rad_adapt3r_scene_only \
        baku/camera_change/cs_1.2_rad_adapt3r_scene_only \
        baku/camera_change/cs_1.4_rad_adapt3r_scene_only \
        baku/camera_change/cs_1.6_rad_adapt3r_scene_only \
        baku/camera_change/cs_1.8_rad_adapt3r_scene_only \
        baku/camera_change/cs_2.0_rad_adapt3r_scene_only \
    --line-labels RGB RGBD DP3 Adapt3R \
    --x-labels 0 0.4 0.8 1.2 1.6 2.0 \
    --xticks 0 2 4 6 8 10 \
    --figsize 5 3 \
    --xlabel "$\theta$" --ylabel "Success Rate" \
    --legend-outfile zplots/cs-sweep-legend-scene-only.pdf \
    --title LIBERO-90 \
    --font-size-xtick 14  --font-size-ytick 12  --font-size-annot 12 --font-size-label 14 --font-size-title 16 \
    --fname zplots/libero-cs-sweep-scene-only.pdf
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



