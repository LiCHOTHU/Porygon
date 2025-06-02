from argparse import ArgumentParser
import os
import json
import numpy as np

# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
# rcParams.update({'figure.autolayout': True})


def main():
    parser = ArgumentParser()
    parser.add_argument('data_dirs', nargs='*')
    parser.add_argument('--filter')
    args = parser.parse_args()

    mean_success_rate = []
    std_errors = []
    prefix = os.path.commonprefix(args.data_dirs)
    # labels = [data_dir[len(prefix):] for data_dir in args.data_dirs]
    for data_dir in args.data_dirs:
        data = []
        for root, dirs, files in os.walk(data_dir):
            if 'data.json' in files:
                if args.filter is not None and args.filter not in root:
                    # print('skipping', os.path.join())
                    continue
                with open(os.path.join(root, 'data.json'), 'r') as f:
                    data_dict = json.load(f)
                data.append(data_dict['rollout']['overall_success_rate'])
        mean = np.mean(data)
        std_error = np.std(data) / np.sqrt(len(data))
        mean_success_rate.append(mean)
        std_errors.append(std_error)

        print(f'{data_dir[len(prefix):]}: {mean:1.3f} +/- {std_error:1.3f}; len: {len(data)}')

    # Create plot
    # fig, ax = plt.subplots(figsize=(8, 6))

    # # Plot bars with error bars
    # bars = ax.bar(labels, mean_success_rate, color=colors, yerr=std_error, capsize=5)

    # # Add text annotations
    # for i, bar in enumerate(bars):
    #     height = bar.get_height()
    #     ax.annotate(f'{height:.1f}',
    #                 xy=(bar.get_x() + bar.get_width() / 2, height + std_error[i]/2 + 1.5),
    #                 xytext=(0, 3),  # 3 points vertical offset
    #                 textcoords="offset points",
    #                 ha='center', va='bottom',
    #                 fontsize=args.font_size_annotation)  # Increased font size

    # # Labeling
    # ax.set_xlabel(args.xlabel, fontsize=args.font_size_xlabel)
    # ax.set_ylabel(args.ylabel, fontsize=args.font_size_ylabel)
    # ax.set_title(args.title, fontsize=args.font_size_title)
    # ax.set_ylim(args.ylim_low, args.ylim_high)

    # # Customize the plot
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    # ax.set_axisbelow(True)

    # # Increase font size of x-tick labels
    # plt.xticks(rotation=0, fontsize=args.font_size_xticks)
    # plt.yticks(fontsize=args.font_size_yticks)

    # # Show the plot
    # plt.tight_layout()
    # plt.savefig(f'plots/{args.fname}.pdf')
    # plt.close()


if __name__ == '__main__':
    main()