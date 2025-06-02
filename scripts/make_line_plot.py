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
        if prefix is not None:
            folder = os.path.join(prefix, folder)
        for root, dirs, files in os.walk(folder):
            if 'data.json' in files:
                with open(os.path.join(root, 'data.json'), 'r') as f:
                    data_dict = json.load(f)
                data.append(data_dict['rollout']['overall_success_rate'] * 100)
    mean_success_rate = np.mean(data)
    std_error = np.std(data) / np.sqrt(len(data)) if data else 0
    return mean_success_rate, std_error




def main():
    parser = ArgumentParser()
    parser.add_argument('--data-dirs', nargs='*', required=True)
    parser.add_argument('--x-labels', nargs='*')
    parser.add_argument('--line-labels', nargs='*')
    parser.add_argument('--colors', nargs='*')
    parser.add_argument('--prefix')

    # Customization defaults
    parser.add_argument('--xlabel', type=str)
    parser.add_argument('--xticks', nargs='*', type=float)
    parser.add_argument('--ylabel', type=str)
    parser.add_argument('--title', type=str)
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
    parser.add_argument('--legend-outfile', type=str, default=None, help="Optional path to save the legend separately")
    parser.add_argument('--fname')
    parser.add_argument('--show', action='store_true')

    args = parser.parse_args()

    prefix = args.prefix
    data_dirs = args.data_dirs

    data = []
    for data_dir in data_dirs:
        mean_success_rate, std_error = walk_dirs([data_dir], prefix=prefix)
        data.append((mean_success_rate, std_error))
    
    
    x_labels = args.x_labels
    line_labels = args.line_labels
    colors = args.colors if args.colors else [None] * len(line_labels)

    data = np.array(data)
    mean_values = data[:, 0]
    error_values = data[:, 1]

    # Reshape into (num_lines, num_x_labels)
    num_lines = len(line_labels)
    num_x = len(data) // num_lines

    mean_values = mean_values.reshape(num_lines, -1)
    error_values = error_values.reshape(num_lines, -1)

    # Plot
    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    x = np.arange(num_x)

    lines = []
    for i in range(num_lines):
        line = ax.errorbar(
            x,
            mean_values[i],
            yerr=error_values[i],
            label=line_labels[i],
            color=colors[i] if colors[i] is not None else None,
            linewidth=args.linewidth,
            marker='o',
            capsize=4,
        )
        lines.append(line)

    ax.set_xlabel(args.xlabel, fontsize=args.font_size_label)
    ax.set_ylabel(args.ylabel, fontsize=args.font_size_label)
    ax.set_title(args.title, fontsize=args.font_size_title)
    if args.xticks is None:
        ax.set_xticks(x)
    else:
        assert len(args.xticks) == len(x_labels)
        ax.set_xticks(args.xticks)
    ax.set_xticklabels(x_labels, fontsize=args.font_size_xtick, rotation=0)
    ax.tick_params(axis='y', labelsize=args.font_size_ytick)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    fig.tight_layout()

    if args.fname:
        plt.savefig(args.fname, bbox_inches='tight')
    
    # ax.legend(fontsize=args.font_size_legend)
    if args.legend_outfile:
        legend_fig, legend_ax = plt.subplots(figsize=(4, 0.000001))
        legend_ax.axis('off')
        legend = legend_ax.legend(
            handles=[l[0] for l in lines],
            labels=line_labels,
            loc='center',
            frameon=False,
            fontsize=args.font_size_legend,
            ncol=len(line_labels),  # all labels in one row
        )
        legend_fig.savefig(args.legend_outfile, bbox_inches='tight')


    if args.show or not args.fname:
        plt.show()


if __name__ == '__main__':
    main()
