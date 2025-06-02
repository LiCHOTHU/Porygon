import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base_dir = "real_data"
methods = ["rgb", "dp3", "3dda", "adapt3r"]
labels = ["RGB", "DP3", "3DDA", "Adapt3R"]
conditions = ["mt", "cam_change"]

avg_completions = {method: {} for method in methods}
ste_completions = {method: {} for method in methods}

for method in methods:
    for condition in conditions:
        file_path = os.path.join(base_dir, method, f"{condition}.csv")
        df = pd.read_csv(file_path, index_col=0)
        df.dropna(how="all", inplace=True)
        try_cols = [col for col in df.columns if col.startswith("Try")]
        try_values = df[try_cols].astype(float)
        try_values_arr = np.array(try_values) * 100 / 3
        avg_completions[method][condition] = try_values_arr.mean()
        ste_completions[method][condition] = try_values_arr.std() / np.sqrt(try_values_arr.size)

# Data for bars
mt_means = [avg_completions[m]["mt"] for m in methods]
cam_means = [avg_completions[m]["cam_change"] for m in methods]
mt_stds = [ste_completions[m]["mt"] for m in methods]
cam_stds = [ste_completions[m]["cam_change"] for m in methods]

# Set up positions
n_methods = len(methods)
x = np.arange(n_methods)
width = 0.5
error_kw = {
    "capsize": 2,
    "elinewidth": 1,
    "ecolor": "black"
}

fig, ax = plt.subplots(figsize=(6, 4))

# Plot original (grey) bars
bars_mt = ax.bar(x, mt_means, width,
                 color='lightgrey', edgecolor='grey', linewidth=1.0,
                 yerr=mt_stds, error_kw=error_kw, label='Original Viewpoint')

# Plot unseen (red) bars shifted upward
bars_cam = ax.bar(x, cam_means, width,
                #   bottom=mt_means,  # this shifts them up
                  color='darkred', edgecolor='black', linewidth=1.0,
                  yerr=cam_stds, error_kw=error_kw, label='Unseen Viewpoint')

# Horizontal gridlines
ax.yaxis.grid(True, linestyle='-', alpha=0.5)
ax.set_axisbelow(True)

# Set y limits
ax.set_ylim(0, 100)

# Add labels
for bar_mt, bar_cam, mt_std, cam_std in zip(bars_mt, bars_cam, mt_stds, cam_stds):
    height_mt = bar_mt.get_height()
    height_cam = bar_cam.get_height()
    ax.annotate(f'{height_mt:.1f}',
                xy=(bar_mt.get_x() + bar_mt.get_width() / 2, height_mt),
                xytext=(0, 0),
                textcoords="offset points",
                ha='center', va='center', fontsize=10, color='black')
    ax.annotate(f'{height_cam:.1f}',
                xy=(bar_cam.get_x() + bar_cam.get_width() / 2, bar_mt.get_height() + height_cam / 2),
                xytext=(0, 0),
                textcoords="offset points",
                ha='center', va='center', fontsize=10, color='white')

# Final formatting
ax.set_ylabel('Average Completion (%)', fontsize=14)
ax.set_title('Performance by Viewpoint', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=14)
ax.legend(fontsize=12)

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/real_res_stacked_separate.pdf')
plt.show()
