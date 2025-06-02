
# In dist multitask
python scripts/make_fancy_bar_plot.py \
    --prefix /home/albert/quest/scripts/plotting/evaluate_old/libero/libero_90 \
    --data-dirs \
        act/multitask/rgb \
        diffusion_policy/multitask/rgb \
        baku/multitask/rgb \
        act/multitask/rgbd \
        diffusion_policy/multitask/rgbd \
        baku/multitask/rgbd \
        act/multitask/dp3 \
        diffusion_policy/multitask/dp3 \
        baku/multitask/dp3 \
        diffuser_actor/multitask/block_16 \
        act/multitask/proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
        diffusion_policy/multitask/proprio_tight_hand_crop_ds_512_clip_ft_false_block_8 \
        baku/multitask/proprio_tight_hand_crop_ds_512_clip_ft_false_block_10 \
    --labels RGB RGBD DP3 3DDA Ours \
    --bar-sizes 3 3 3 1 3 \
    --figsize 4 3 \
    --font-size-xtick 14  --font-size-ytick 12  --font-size-annot 12 --font-size-label 14 \
    --ylabel "Success Rate (%)" \
    --colors lightgrey lightgrey lightgrey lightgrey steelblue \
    --fname zplots/main_mt.pdf \
    # --show \



