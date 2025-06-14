from argparse import ArgumentParser
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

def walk_dirs(folders, prefix=None):
    data = []
    for folder in folders:
        data_len_begin = len(data)
        if prefix is not None:
            folder = os.path.join(prefix, folder)
        for root, dirs, files in os.walk(folder):
            if 'data.json' in files:
                with open(os.path.join(root, 'data.json'), 'r') as f:
                    data_dict = json.load(f)
                data.append(data_dict['rollout']['overall_success_rate'] * 100)
        if len(data) == data_len_begin:
            print(f'Warning: {folder} had no data')
    mean_success_rate = np.mean(data)
    std_error = np.std(data) / np.sqrt(len(data)) if data else 0
    return mean_success_rate, std_error


def plot_grouped_bar(data, args):
    data = np.array(data).reshape(-1, args.group_size, 2)  # shape: [num_groups, group_size, (mean, err)]
    num_groups, bars_per_group = data.shape[:2]

    means = data[:, :, 0]
    errors = data[:, :, 1]

    # x = np.arange(num_groups)
    bar_width = args.group_width / bars_per_group
    error_kw = {
        "capsize": 2,
        "elinewidth": 1,
        "ecolor": "black"
    }

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    for i in range(num_groups):
        if args.colors is not None:
            bar_colors = args.colors[i * bars_per_group: (i + 1) * bars_per_group]
        else:
            bar_colors = None

        if args.line_colors is not None:
            line_colors = args.line_colors[i * bars_per_group: (i + 1) * bars_per_group]
        else:
            line_colors = 'black'

        x = (np.arange(bars_per_group) - (bars_per_group - 1) / 2) * bar_width + i
        bars = ax.bar(
            x,
            means[i],
            width=bar_width - args.gap,
            yerr=errors[i],
            color=bar_colors,
            linewidth=args.linewidth,
            edgecolor=line_colors,
            error_kw=error_kw,
            zorder=2,
        )

        if args.do_annotations:
            for bar, error in zip(bars, errors[i]):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + error,
                    f'{height:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=args.font_size_annot
                )

    
    ax.set_ylabel(args.ylabel, fontsize=args.font_size_label)
    ax.set_xlabel(args.xlabel, fontsize=args.font_size_label)
    ax.set_title(args.title, fontsize=args.font_size_title)
    ax.set_xticks(np.arange(num_groups))
    ax.set_xticklabels(args.labels, fontsize=args.font_size_xtick, rotation=args.xlabel_rotation)
    if args.ylim is not None:
        ax.set_ylim(0, args.ylim)
    ax.tick_params(axis='y', labelsize=args.font_size_ytick)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    plt.tight_layout()
    if args.fname:
        plt.savefig(args.fname)
    # if args.legend_colors:

    if args.legend_fname:
        assert len(args.legend_colors) == len(args.legend_labels)
        legend_fig, legend_ax = plt.subplots(figsize=(4, 0.000001))
        legend_ax.axis('off')
        
        patches = []
        for color, label in zip(args.legend_colors, args.legend_labels):
            patches.append(mpatches.Patch(color=color, label=label))

        legend_ax.legend(
            handles=patches, 
            fontsize=args.font_size_legend,
            loc='center',
            frameon=False,
            ncol=len(patches),
        )
        legend_fig.savefig(args.legend_fname, bbox_inches='tight')


    if args.show:
        plt.show()
    else:
        plt.close()



def main():
    parser = ArgumentParser()
    parser.add_argument('--data-dirs', nargs='*', required=True)
    parser.add_argument('--bar-sizes', nargs='*', type=int)
    parser.add_argument('--labels', nargs='*')
    parser.add_argument('--colors', nargs='*')
    parser.add_argument('--line-colors', nargs='*')
    parser.add_argument('--group-size', type=int, default=1)
    parser.add_argument('--prefix')

    # Customization defaults
    parser.add_argument('--xlabel', type=str)
    parser.add_argument('--ylabel', type=str)
    parser.add_argument('--title', type=str)
    parser.add_argument('--ylim', type=float)
    parser.add_argument('--figsize', nargs=2, type=float, default=[10.0, 6.0])
    parser.add_argument('--font-size-label', type=int, default=12)
    parser.add_argument('--font-size-xtick', type=int, default=10)
    parser.add_argument('--font-size-ytick', type=int, default=10)
    parser.add_argument('--font-size-annot', type=int, default=10)
    parser.add_argument('--font-size-title', type=int, default=14)
    parser.add_argument('--font-size-legend', type=int, default=14)
    parser.add_argument('--legend-colors', nargs='*', default=[])
    parser.add_argument('--legend-labels', nargs='*', default=[])
    parser.add_argument('--linewidth', default=1.0, type=float)
    parser.add_argument('--group-width', default=0.8, type=float)
    parser.add_argument('--do-annotations', action='store_true')
    parser.add_argument('--gap', default=0, type=float)
    parser.add_argument('--fname')
    parser.add_argument('--legend-fname')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--xlabel-rotation', type=float, default=0, help='Rotation angle for x-axis labels in degrees')

    args = parser.parse_args()

    prefix = args.prefix
    data_dirs = args.data_dirs
    bar_sizes = args.bar_sizes

    if bar_sizes is None:
        bar_sizes = [1] * len(data_dirs)
    assert sum(bar_sizes) == len(data_dirs)

    data = []
    for bar_size in bar_sizes:
        bar_data_dirs = data_dirs[:bar_size]
        data_dirs = data_dirs[bar_size:]
        mean_success_rate, std_error = walk_dirs(bar_data_dirs, prefix=prefix)
        data.append((mean_success_rate, std_error))
    

    plot_grouped_bar(data, args)


if __name__ == '__main__':
    main()
