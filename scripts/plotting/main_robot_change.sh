
# In dist multitask
python scripts/make_fancy_bar_plot.py \
    --prefix /home/albert/quest/scripts/plotting/evaluate_old/libero/libero_90 \
    --data-dirs \
        act/robot_change/rgb_UR5e \
        diffusion_policy/robot_change/rgb_UR5e \
        baku/robot_change/rgb_UR5e \
        act/robot_change/rgb_Kinova3 \
        diffusion_policy/robot_change/rgb_Kinova3 \
        baku/robot_change/rgb_Kinova3 \
        act/robot_change/rgb_IIWA \
        diffusion_policy/robot_change/rgb_IIWA \
        baku/robot_change/rgb_IIWA \
        act/robot_change/rgbd_UR5e \
        diffusion_policy/robot_change/rgbd_UR5e \
        baku/robot_change/rgbd_UR5e \
        act/robot_change/rgbd_Kinova3 \
        diffusion_policy/robot_change/rgbd_Kinova3 \
        baku/robot_change/rgbd_Kinova3 \
        act/robot_change/rgbd_IIWA \
        diffusion_policy/robot_change/rgbd_IIWA \
        baku/robot_change/rgbd_IIWA \
        act/robot_change/dp3_UR5e \
        diffusion_policy/robot_change/dp3_UR5e \
        baku/robot_change/dp3_UR5e \
        act/robot_change/dp3_Kinova3 \
        diffusion_policy/robot_change/dp3_Kinova3 \
        baku/robot_change/dp3_Kinova3 \
        act/robot_change/dp3_IIWA \
        diffusion_policy/robot_change/dp3_IIWA \
        baku/robot_change/dp3_IIWA \
        diffuser_actor/robot_change/UR5e_block_16 \
        diffuser_actor/robot_change/Kinova3_block_16 \
        diffuser_actor/robot_change/IIWA_block_16 \
        act/robot_change/UR5e_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        diffusion_policy/robot_change/UR5e_proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        baku/robot_change/UR5e_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        act/robot_change/Kinova3_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        diffusion_policy/robot_change/Kinova3_proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        baku/robot_change/Kinova3_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        act/robot_change/IIWA_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        diffusion_policy/robot_change/IIWA_proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        baku/robot_change/IIWA_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
    --labels RGB RGBD DP3 3DDA Adapt3R \
    --group-size 3 \
    --bar-sizes 3 3 3  3 3 3  3 3 3  1 1 1  3 3 3 \
    --figsize 7 3 \
    --colors \
        skyblue gainsboro darkorange \
        skyblue gainsboro darkorange \
        skyblue gainsboro darkorange \
        skyblue gainsboro darkorange \
        skyblue gainsboro darkorange \
    --font-size-xtick 14  --font-size-ytick 12  --font-size-annot 12 --font-size-label 14 \
    --legend-labels "UR5e" "Kinova3" "Kuka IIWA" \
    --legend-colors skyblue gainsboro darkorange \
    --group-width 0.95 \
    --fname zplots/main_robot_change.pdf \
    # --show \
        # darkred forestgreen steelblue



