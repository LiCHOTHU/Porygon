
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
    python scripts/make_line_plot.py \
        --prefix /storage/home/hcoda1/1/awilcox31/shared-vast/camera_ready/mimicgen_final_evals/${benchmark} \
        --data-dirs \
            mt_rgb \
            cam_shift_0.2_rgb \
            cam_shift_0.4_rgb \
            cam_shift_0.6_rgb \
            cam_shift_0.8_rgb \
            cam_shift_1.0_rgb \
            mt_rgbd \
            cam_shift_0.2_rgbd \
            cam_shift_0.4_rgbd \
            cam_shift_0.6_rgbd \
            cam_shift_0.8_rgbd \
            cam_shift_1.0_rgbd \
            mt_dp3 \
            cam_shift_0.2_dp3 \
            cam_shift_0.4_dp3 \
            cam_shift_0.6_dp3 \
            cam_shift_0.8_dp3 \
            cam_shift_1.0_dp3 \
            mt_3dda \
            cam_shift_0.2_3dda \
            cam_shift_0.4_3dda \
            cam_shift_0.6_3dda \
            cam_shift_0.8_3dda \
            cam_shift_1.0_3dda \
            mt_adapt3r \
            cam_shift_0.2_adapt3r \
            cam_shift_0.4_adapt3r \
            cam_shift_0.6_adapt3r \
            cam_shift_0.8_adapt3r \
            cam_shift_1.0_adapt3r \
        --line-labels RGB RGBD DP3 3DDA Adapt3R \
        --x-labels 0 0.2 0.4 0.6 0.8 1.0 \
        --figsize 3 3 \
        --xlabel "$\theta$"  \
        --title "${title}" \
        --font-size-xtick 14  --font-size-ytick 12  --font-size-annot 12 --font-size-label 14 --font-size-title 16 \
        --fname zplots/${benchmark}-cs-sweep.pdf
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
done



