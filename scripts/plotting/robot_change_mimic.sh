# In dist multitask
benchmarks=(coffee square threading)
titles=(
    "MG Coffee"
    "MG Square"
    "MG Threading"
)

length=${#benchmarks[@]}

for (( i = 0; i < $length; i++ )) ; do
benchmark=${benchmarks[i]}
title=${titles[i]}
# In dist multitask
python scripts/make_fancy_bar_plot.py \
    --prefix /storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready/mimicgen_final_evals/${benchmark} \
    --data-dirs \
        UR5e_rgb \
        UR5e_rgbd \
        UR5e_dp3 \
        UR5e_3dda \
        UR5e_adapt3r \
        Kinova3_rgb \
        Kinova3_rgbd \
        Kinova3_dp3 \
        Kinova3_3dda \
        Kinova3_adapt3r \
        IIWA_rgb \
        IIWA_rgbd \
        IIWA_dp3 \
        IIWA_3dda \
        IIWA_adapt3r \
    --labels UR5e Kinova3 "Kuka IIWA" \
    --title "${title}" \
    --group-size 5 \
    --figsize 4 3 \
    --colors \
        '#aec7e8' '#ffbb78' '#98df8a' '#f7b6d2' '#d5b9f5' \
        '#aec7e8' '#ffbb78' '#98df8a' '#f7b6d2' '#d5b9f5' \
        '#aec7e8' '#ffbb78' '#98df8a' '#f7b6d2' '#d5b9f5' \
    --line-colors \
        '#1f77b4' '#ff7f0e' '#2ca02c' '#d62728' '#9467bd' \
        '#1f77b4' '#ff7f0e' '#2ca02c' '#d62728' '#9467bd' \
        '#1f77b4' '#ff7f0e' '#2ca02c' '#d62728' '#9467bd' \
    --font-size-xtick 14  --font-size-ytick 12  --font-size-annot 6 --font-size-label 14 --font-size-title 16 \
    --legend-labels RGB RGBD DP3 3DDA Adapt3R \
    --legend-colors '#aec7e8' '#ffbb78' '#98df8a' '#f7b6d2' '#d5b9f5' \
    --legend-fname zplots/legend.pdf \
    --group-width 0.8  --gap 0.01 \
    --fname zplots/${benchmark}-xe.pdf \
    # --show \
        # darkred forestgreen steelblue
done


