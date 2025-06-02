
# In dist multitask
python scripts/make_fancy_bar_plot.py \
    --prefix /storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready/evaluate/libero/libero_90 \
    --data-dirs \
        act/robot_change/UR5e_rgb_scene_only \
        diffusion_policy/robot_change/UR5e_rgb_scene_only \
        baku/robot_change/UR5e_rgb_scene_only \
        act/robot_change/UR5e_rgbd_scene_only \
        diffusion_policy/robot_change/UR5e_rgbd_scene_only \
        baku/robot_change/UR5e_rgbd_scene_only \
        act/robot_change/UR5e_dp3_scene_only \
        diffusion_policy/robot_change/UR5e_dp3_scene_only \
        baku/robot_change/UR5e_dp3_scene_only \
        act/robot_change/UR5e_adapt3r_scene_only \
        diffusion_policy/robot_change/UR5e_adapt3r_scene_only \
        baku/robot_change/UR5e_adapt3r_scene_only \
        act/robot_change/Kinova3_rgb_scene_only \
        diffusion_policy/robot_change/Kinova3_rgb_scene_only \
        baku/robot_change/Kinova3_rgb_scene_only \
        act/robot_change/Kinova3_rgbd_scene_only \
        diffusion_policy/robot_change/Kinova3_rgbd_scene_only \
        baku/robot_change/Kinova3_rgbd_scene_only \
        act/robot_change/Kinova3_dp3_scene_only \
        diffusion_policy/robot_change/Kinova3_dp3_scene_only \
        baku/robot_change/Kinova3_dp3_scene_only \
        act/robot_change/Kinova3_adapt3r_scene_only \
        diffusion_policy/robot_change/Kinova3_adapt3r_scene_only \
        baku/robot_change/Kinova3_adapt3r_scene_only \
        act/robot_change/IIWA_rgb_scene_only \
        diffusion_policy/robot_change/IIWA_rgb_scene_only \
        baku/robot_change/IIWA_rgb_scene_only \
        act/robot_change/IIWA_rgbd_scene_only \
        diffusion_policy/robot_change/IIWA_rgbd_scene_only \
        baku/robot_change/IIWA_rgbd_scene_only \
        act/robot_change/IIWA_dp3_scene_only \
        diffusion_policy/robot_change/IIWA_dp3_scene_only \
        baku/robot_change/IIWA_dp3_scene_only \
        act/robot_change/IIWA_adapt3r_scene_only \
        diffusion_policy/robot_change/IIWA_adapt3r_scene_only \
        baku/robot_change/IIWA_adapt3r_scene_only \
    --labels UR5e Kinova3 "Kuka IIWA" \
    --title LIBERO-90 \
    --group-size 4 \
    --bar-sizes 3 3 3 3  3 3 3 3  3 3 3 3  \
    --figsize 4 3 \
    --colors \
        '#aec7e8' '#ffbb78' '#98df8a' '#f7b6d2' \
        '#aec7e8' '#ffbb78' '#98df8a' '#f7b6d2' \
        '#aec7e8' '#ffbb78' '#98df8a' '#f7b6d2' \
    --line-colors \
        '#1f77b4' '#ff7f0e' '#2ca02c' '#d62728' \
        '#1f77b4' '#ff7f0e' '#2ca02c' '#d62728' \
        '#1f77b4' '#ff7f0e' '#2ca02c' '#d62728' \
    --font-size-xtick 14  --font-size-ytick 12  --font-size-annot 6 --font-size-label 14 --font-size-title 16 \
    --legend-labels RGB RGBD DP3 Adapt3R \
    --legend-colors '#aec7e8' '#ffbb78' '#98df8a' '#f7b6d2' \
    --legend-fname zplots/robot_change_legend-scene-only.pdf \
    --group-width 0.8  --gap 0.01 \
    --fname zplots/libero-xe-scene-only.pdf \
    # --show \
        # darkred forestgreen steelblue



