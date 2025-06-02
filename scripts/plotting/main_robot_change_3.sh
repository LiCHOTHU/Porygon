
# In dist multitask
python scripts/make_fancy_bar_plot.py \
    --prefix /home/albert/quest/scripts/plotting/evaluate_old/libero/libero_90 \
    --data-dirs \
        act/robot_change/rgb_UR5e \
        diffusion_policy/robot_change/rgb_UR5e \
        baku/robot_change/rgb_UR5e \
        act/robot_change/rgbd_UR5e \
        diffusion_policy/robot_change/rgbd_UR5e \
        baku/robot_change/rgbd_UR5e \
        act/robot_change/dp3_UR5e \
        diffusion_policy/robot_change/dp3_UR5e \
        baku/robot_change/dp3_UR5e \
        diffuser_actor/robot_change/UR5e_block_16 \
        act/robot_change/UR5e_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        diffusion_policy/robot_change/UR5e_proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        baku/robot_change/UR5e_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        act/robot_change/rgb_Kinova3 \
        diffusion_policy/robot_change/rgb_Kinova3 \
        baku/robot_change/rgb_Kinova3 \
        act/robot_change/rgbd_Kinova3 \
        diffusion_policy/robot_change/rgbd_Kinova3 \
        baku/robot_change/rgbd_Kinova3 \
        act/robot_change/dp3_Kinova3 \
        diffusion_policy/robot_change/dp3_Kinova3 \
        baku/robot_change/dp3_Kinova3 \
        diffuser_actor/robot_change/Kinova3_block_16 \
        act/robot_change/Kinova3_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        diffusion_policy/robot_change/Kinova3_proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        baku/robot_change/Kinova3_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        act/robot_change/rgb_IIWA \
        diffusion_policy/robot_change/rgb_IIWA \
        baku/robot_change/rgb_IIWA \
        act/robot_change/rgbd_IIWA \
        diffusion_policy/robot_change/rgbd_IIWA \
        baku/robot_change/rgbd_IIWA \
        act/robot_change/dp3_IIWA \
        diffusion_policy/robot_change/dp3_IIWA \
        baku/robot_change/dp3_IIWA \
        diffuser_actor/robot_change/IIWA_block_16 \
        act/robot_change/IIWA_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        diffusion_policy/robot_change/IIWA_proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        baku/robot_change/IIWA_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
    --labels UR5e Kinova3 "Kuka IIWA" \
    --title LIBERO-90 \
    --group-size 5 \
    --bar-sizes 3 3 3 1 3  3 3 3 1 3  3 3 3 1 3  \
    --figsize 7 3 \
    --colors \
        '#aec7e8' '#ffbb78' '#98df8a' '#f7b6d2' '#d5b9f5' \
        '#aec7e8' '#ffbb78' '#98df8a' '#f7b6d2' '#d5b9f5' \
        '#aec7e8' '#ffbb78' '#98df8a' '#f7b6d2' '#d5b9f5' \
    --line-colors \
        '#1f77b4' '#ff7f0e' '#2ca02c' '#d62728' '#9467bd' \
        '#1f77b4' '#ff7f0e' '#2ca02c' '#d62728' '#9467bd' \
        '#1f77b4' '#ff7f0e' '#2ca02c' '#d62728' '#9467bd' \
    --font-size-xtick 14  --font-size-ytick 12  --font-size-annot 10 --font-size-label 14 --font-size-title 16 \
    --ylim 90 \
    --legend-labels RGB RGBD DP3 3DDA Adapt3R \
    --legend-colors '#aec7e8' '#ffbb78' '#98df8a' '#f7b6d2' '#d5b9f5' \
    --legend-fname zplots/robot_change_legend.pdf \
    --group-width 0.8 \
    --fname zplots/main_robot_change.pdf \
    # --show \
        # darkred forestgreen steelblue



