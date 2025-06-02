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

# Data for grouped bars
mt_means = [avg_completions[m]["mt"] for m in methods]
cam_means = [avg_completions[m]["cam_change"] for m in methods]
mt_stds = [ste_completions[m]["mt"] for m in methods]
cam_stds = [ste_completions[m]["cam_change"] for m in methods]

# Set up positions
n_methods = len(methods)
x = np.arange(n_methods)
width = 0.35
error_kw = {
    "capsize": 2,
    "elinewidth": 1,
    "ecolor": "black"
}

fig, ax = plt.subplots(figsize=(6, 4))

# New colors
mt_colors = ['#aec7e8'] * n_methods  # Light blue
cam_colors = ['#ffbb78'] * n_methods  # Light coral/orange
mt_edgecolors = ['#1f77b4'] * n_methods  # Darker blue
cam_edgecolors = ['#ff7f0e'] * n_methods  # Darker orange

# Plot bars
bars_mt = ax.bar(
    x - width/2 - 0.02, 
    mt_means, 
    width, 
    label='Original Viewpoint',
    color=mt_colors, 
    edgecolor=mt_edgecolors, 
    linewidth=1.5, 
    error_kw=error_kw)
bars_cam = ax.bar(
    x + width/2 + 0.02, 
    cam_means, 
    width, 
    label='Unseen Viewpoint',
    color=cam_colors, 
    edgecolor=cam_edgecolors, 
    linewidth=2.0, 
    error_kw=error_kw)

# Arrows and annotations
for i, mt_mean, cam_mean in zip(range(4), mt_means, cam_means):
    head_length = 2
    x_ = i + width / 2 - 0.08
    y = mt_mean
    diff = cam_mean - mt_mean
    dy = cam_mean - mt_mean + head_length + 1
    plt.arrow(x_, y, 0, dy, head_width=0.1, head_length=head_length, ec='black', overhang=1)
    xytext = (0, 3)
    ax.annotate(f'{diff:2.1f}', xy=(x_ + 0.08, y + diff / 2), xytext=xytext,
                        textcoords="offset points",
                        ha='left', va='center', fontsize=12)

# Horizontal gridlines
ax.yaxis.grid(True, linestyle='-', alpha=0.5)
ax.set_axisbelow(True)

# Axis limits
max_val = max([m + s for m, s in zip(mt_means + cam_means, mt_stds + cam_stds)])
ax.set_ylim(0, max_val * 1.2)

# Labels on bars
i = 0
for data in [(bars_mt, mt_stds), (bars_cam, cam_stds)]:
    for bar, std in zip(*data):
        height = bar.get_height()
        if i == 0:
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11)
        else:
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, -5),
                        textcoords="offset points",
                        ha='center', va='top', fontsize=11, color='black')
    i += 1

# Axis labels and ticks
ax.set_ylabel('Average Completion (%)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=14)
ax.legend(fontsize=12)

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/real_res.pdf')
plt.show()
