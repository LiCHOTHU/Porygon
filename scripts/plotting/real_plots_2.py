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

        # Drop rows that are entirely empty
        df.dropna(how="all", inplace=True)

        # Select only 'Try' columns (assumes they start with 'Try')
        try_cols = [col for col in df.columns if col.startswith("Try")]
        try_values = df[try_cols].astype(float)

        try_values_arr = np.array(try_values) * 100 / 3

        # breakpoint()

        # avg = try_values.mean(axis=1).mean() * 100 / 3
        # std = try_values.mean(axis=1).std() * 100 / 3

        avg_completions[method][condition] = try_values_arr.mean()
        ste_completions[method][condition] = try_values_arr.std() / np.sqrt(try_values_arr.size)

x = np.arange(len(methods)) * 0.8
width = 0.6
error_kw = {
    "capsize": 2,
    "elinewidth": 1,
    "ecolor": "black"
}

mt_means = [avg_completions[m]["mt"] for m in methods]
cam_means = [avg_completions[m]["cam_change"] for m in methods]
mt_stds = [ste_completions[m]["mt"] for m in methods]
cam_stds = [ste_completions[m]["cam_change"] for m in methods]
# colors = ['lightblue', 'lightblue', 'lightblue', 'steelblue']
colors = ['lightgrey', 'lightgrey', 'lightgrey', 'steelblue']

def plot_bar_with_labels(title, means, stds, color, edge_color, linewidth, fname=None, show=True):
    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(x, means, width, yerr=stds, color=color,
                #   edgecolor=edge_color, linewidth=linewidth)
                  edgecolor=edge_color, linewidth=linewidth, error_kw=error_kw)
    
    # Add horizontal gridlines
    ax.yaxis.grid(True, linestyle='-', alpha=0.5)
    ax.set_axisbelow(True)

    # Set axis limits
    max_val = max([m + s for m, s in zip(means, stds)])
    ax.set_ylim(0, max_val * 1.17)  # 10% padding

    # Labels
    for bar, std in zip(bars, stds):
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height + std),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)

    ax.set_ylabel('Average Completion (%)', fontsize=14)
    ax.set_title(title, fontsize=16)
    # ax.set_yticks([0, 20, 40, 60, 80, 100], fontsize=12)
    ax.set_yticklabels([0, 20, 40, 60, 80, 100], fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    plt.tight_layout()
    if fname is not None:
        os.makedirs(os.path.basename(os.path.dirname(fname)), exist_ok=True)
        plt.savefig(fname)
    if show:
        plt.show()
    else:
        plt.close()

# Plot MT
plot_bar_with_labels(
    title='Original Viewpoint',
    means=mt_means,
    stds=mt_stds,
    color=colors,
    # color='light',
    edge_color='black',
    linewidth=1.0,
    fname='outputs/real_res_multitask.pdf',
    show=True
)

# Plot Cam Change
plot_bar_with_labels(
    title='Unseen Viewpoint',
    means=cam_means,
    stds=cam_stds,
    color=colors,
    edge_color='black',
    linewidth=1.0,
    fname='outputs/real_res_cam_change.pdf',
    show=True
)