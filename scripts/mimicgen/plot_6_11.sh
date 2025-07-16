
# In dist multitask
python scripts/make_fancy_bar_plot.py \
    --prefix /storage/home/hcoda1/1/awilcox31/vast/imitation/experiments/evaluate/mimicgen/square_d1/fixed_sweep_actions_space \
    --data-dirs \
        mt_adapt3r_ft \
        mt_adapt3r_ft_abs \
        mt_adapt3r_ft_abs_eecf \
        mt_adapt3r_ft_eecf \
        mt_adapt3r_ft_no_hf \
        mt_adapt3r_ft_no_hf_abs \
        mt_rgb \
        mt_rgb_abs \
    --labels \
        "Delta" \
        "ABS" \
        "ABS + action EECF" \
        "Delta + action EECF" \
        "No PC EECF" \
        "No PC EECF + ABS" \
        "RGB + Delta" \
        "RGB + ABS" \
    --font-size-xtick 14  --font-size-ytick 12  --font-size-annot 12 --font-size-label 14 \
    --xlabel-rotation 45 \
    --ylabel "Success Rate (%)" \
    --fname zplots/6_11_actions_space.pdf \
    # --show \



