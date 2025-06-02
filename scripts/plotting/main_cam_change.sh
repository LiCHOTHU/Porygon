
# In dist multitask
python scripts/make_fancy_bar_plot.py \
    --prefix /home/albert/quest/scripts/plotting/evaluate_old/libero/libero_90 \
    --data-dirs \
        act/camera_change/rgb_small \
        diffusion_policy/camera_change/rgb_small \
        baku/camera_change/rgb_small \
        act/camera_change/rgb_medium \
        diffusion_policy/camera_change/rgb_medium \
        baku/camera_change/rgb_medium \
        act/camera_change/rgb_large \
        diffusion_policy/camera_change/rgb_large \
        baku/camera_change/rgb_large \
        act/camera_change/rgbd_small \
        diffusion_policy/camera_change/rgbd_small \
        baku/camera_change/rgbd_small \
        act/camera_change/rgbd_medium \
        diffusion_policy/camera_change/rgbd_medium \
        baku/camera_change/rgbd_medium \
        act/camera_change/rgbd_large \
        diffusion_policy/camera_change/rgbd_large \
        baku/camera_change/rgbd_large \
        act/camera_change/dp3_small \
        diffusion_policy/camera_change/dp3_small \
        baku/camera_change/dp3_small \
        act/camera_change/dp3_medium \
        diffusion_policy/camera_change/dp3_medium \
        baku/camera_change/dp3_medium \
        act/camera_change/dp3_large \
        diffusion_policy/camera_change/dp3_large \
        baku/camera_change/dp3_large \
        diffuser_actor/camera_change/small_block_16 \
        diffuser_actor/camera_change/medium_block_16 \
        diffuser_actor/camera_change/large_block_16 \
        act/camera_change/small_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        diffusion_policy/camera_change/small_proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        baku/camera_change/small_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        act/camera_change/medium_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        diffusion_policy/camera_change/medium_proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        baku/camera_change/medium_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        act/camera_change/large_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        diffusion_policy/camera_change/large_proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        baku/camera_change/large_proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
    --labels RGB RGBD DP3 3DDA Adapt3R \
    --group-size 3 \
    --bar-sizes 3 3 3  3 3 3  3 3 3  1 1 1  3 3 3 \
    --figsize 7 3 \
    --colors \
        lightcoral lightgreen skyblue \
        lightcoral lightgreen skyblue \
        lightcoral lightgreen skyblue \
        lightcoral lightgreen skyblue \
        lightcoral lightgreen skyblue \
    --line-colors \
        lightcoral lightgreen skyblue \
        lightcoral lightgreen skyblue \
        lightcoral lightgreen skyblue \
        lightcoral lightgreen skyblue \
        lightcoral lightgreen skyblue \
    --linewidth 0 \
    --font-size-xtick 14  --font-size-ytick 12  --font-size-annot 12 --font-size-label 14 \
    --legend-labels "Small Change" "Medium Change" "Large Change" \
    --legend-colors lightcoral lightgreen skyblue \
    --group-width 0.95 \
    --fname zplots/main_cam_change.pdf \
    --show
        # darkred forestgreen steelblue



