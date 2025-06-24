
# In dist multitask
python scripts/make_fancy_bar_plot.py \
    --prefix /storage/home/hcoda1/1/awilcox31/vast/imitation/experiments/evaluate/dexmimicgen/two_arm_coffee/cp_sweep/ \
    --data-dirs \
        mt_adapt3r_bm_concat_ft_abs_eecfn_epoch_0050 \
        mt_adapt3r_bm_concat_ft_abs_eecfn_epoch_0100 \
        mt_adapt3r_bm_concat_ft_abs_eecfn_epoch_0150 \
        mt_adapt3r_bm_concat_ft_abs_eecfn_epoch_0200 \
        mt_adapt3r_bm_concat_ft_abs_eecfn \
        mt_adapt3r_bm_concat_ft_absn_epoch_0050 \
        mt_adapt3r_bm_concat_ft_absn_epoch_0100 \
        mt_adapt3r_bm_concat_ft_absn_epoch_0150 \
        mt_adapt3r_bm_concat_ft_absn_epoch_0200 \
        mt_adapt3r_bm_concat_ft_absn \
        mt_adapt3r_bm_separate_ft_abs_eecfn \
        mt_adapt3r_bm_separate_ft_abs_eecfn_epoch_0050 \
        mt_adapt3r_bm_separate_ft_abs_eecfn_epoch_0100 \
        mt_adapt3r_bm_separate_ft_abs_eecfn_epoch_0150 \
        mt_adapt3r_bm_separate_ft_abs_eecfn_epoch_0200 \
        mt_adapt3r_bm_separate_ft_absn \
        mt_adapt3r_bm_separate_ft_absn_epoch_0050 \
        mt_adapt3r_bm_separate_ft_absn_epoch_0100 \
        mt_adapt3r_bm_separate_ft_absn_epoch_0150 \
        mt_adapt3r_bm_separate_ft_absn_epoch_0200 \
        mt_adapt3r_ft_no_hf_absn \
        mt_dp3_absn \
        mt_rgb_absn \
    --bar-sizes 5 5 5 5 1 1 1  \
    --agg max \
    --labels \
        "concat eecf" \
        "concat" \
        "separate eecf" \
        "separate" \
        "no hf" \
        "dp3" \
        "rgb" \
    --title "3rd Person" \
    --font-size-xtick 14  --font-size-ytick 12  --font-size-annot 12 --font-size-label 14 \
    --xlabel-rotation 45 \
    --do-annotations \
    --ylabel "Success Rate (%)" \
    --fname zplots/6_18_dmg_3rd.pdf \
    # --show \

python scripts/make_fancy_bar_plot.py \
    --prefix /storage/home/hcoda1/1/awilcox31/vast/imitation/experiments/evaluate/dexmimicgen/two_arm_coffee/cp_sweep/ \
    --data-dirs \
        mt_adapt3r_bm_concat_ft_abs_eecfn_ego_epoch_0050 \
        mt_adapt3r_bm_concat_ft_abs_eecfn_ego_epoch_0100 \
        mt_adapt3r_bm_concat_ft_abs_eecfn_ego_epoch_0150 \
        mt_adapt3r_bm_concat_ft_abs_eecfn_ego_epoch_0200 \
        mt_adapt3r_bm_concat_ft_abs_eecfn_ego \
        mt_adapt3r_bm_concat_ft_absn_ego \
        mt_adapt3r_bm_concat_ft_absn_ego_epoch_0050 \
        mt_adapt3r_bm_concat_ft_absn_ego_epoch_0100 \
        mt_adapt3r_bm_concat_ft_absn_ego_epoch_0150 \
        mt_adapt3r_bm_concat_ft_absn_ego_epoch_0200 \
        mt_adapt3r_bm_separate_ft_abs_eecfn_ego \
        mt_adapt3r_bm_separate_ft_abs_eecfn_ego_epoch_0050 \
        mt_adapt3r_bm_separate_ft_abs_eecfn_ego_epoch_0100 \
        mt_adapt3r_bm_separate_ft_abs_eecfn_ego_epoch_0150 \
        mt_adapt3r_bm_separate_ft_abs_eecfn_ego_epoch_0200 \
        mt_adapt3r_bm_separate_ft_absn_ego \
        mt_adapt3r_bm_separate_ft_absn_ego_epoch_0050 \
        mt_adapt3r_bm_separate_ft_absn_ego_epoch_0100 \
        mt_adapt3r_bm_separate_ft_absn_ego_epoch_0150 \
        mt_adapt3r_bm_separate_ft_absn_ego_epoch_0200 \
        mt_adapt3r_ft_no_hf_absn_ego \
        mt_dp3_absn_ego \
        mt_rgb_absn_ego \
    --bar-sizes 5 5 5 5 1 1 1 \
    --agg max \
    --labels \
        "concat eecf" \
        "concat" \
        "separate eecf" \
        "separate" \
        "no hf" \
        "dp3" \
        "rgb" \
    --title "1st Person" \
    --font-size-xtick 14  --font-size-ytick 12  --font-size-annot 12 --font-size-label 14 \
    --xlabel-rotation 45 \
    --do-annotations \
    --ylabel "Success Rate (%)" \
    --fname zplots/6_18_dmg_ego.pdf \

