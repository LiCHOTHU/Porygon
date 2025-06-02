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
    parser.add_argument('--labels', nargs='*')
    # parser.add_argument('--bold_best', action='store_true')
    parser.add_argument('--include')
    parser.add_argument('--exclude')
    args = parser.parse_args()

    assert len(args.data_dirs) == len(args.labels)


    # mean_success_rate = {}
    # std_errors = {}


    # prefix = os.path.commonprefix(args.data_dirs)

    # labels = [data_dir[len(prefix):] for data_dir in args.data_dirs]
    medians, stds = [], []
    for data_dir in args.data_dirs:
        algo_data = {}
        for root, dirs, files in os.walk(data_dir):
            if 'data.json' in files:
                if args.include is not None and args.include not in root or args.exclude is not None and args.exclude in root:
                    # print('skipping', os.path.join())
                    continue
                with open(os.path.join(root, 'data.json'), 'r') as f:
                    data_dict = json.load(f)
                rollout_success_rate = data_dict['rollout_success_rate']
                for env_name in rollout_success_rate:
                    algo_env_data = algo_data.get(env_name, [])
                    algo_env_data.append(rollout_success_rate[env_name])
                    algo_data[env_name] = algo_env_data
        
        algo_medians = {env_name: np.median(algo_data[env_name]) for env_name in algo_data}
        medians.append(algo_medians)
        algo_std_errors = {env_name: np.std(algo_data[env_name]) / np.sqrt(len(algo_data[env_name])) for env_name in algo_data}
        stds.append(algo_std_errors)

    env_names = list(medians[0].keys())
    env_names.sort()
    
    """
    \hline
    \textbf{Task ID} & \textbf{QueST} & \textbf{Diffusion Policy} & \textbf{VQ-BeT} & \textbf{ResNet-T} & \textbf{ACT} & \textbf{PRISE} \\
    \hline
    \endfirsthead

    \hline
    \textbf{Task ID} & \textbf{QueST} & \textbf{Diffusion Policy} & \textbf{VQ-BeT} & \textbf{ResNet-T} & \textbf{ACT} & \textbf{PRISE} \\
    \hline
    \endhead

    \hline
    \endfoot

    \hline
    \endlastfoot
    """


    print('\n\n\n\n\n')

    table = "\\begin{longtable}{|c|" + \
        'c|' * len(args.data_dirs) + \
        '}\n'
    
    table += '\n\\hline\n\\textbf{Task ID}'
    for label in args.labels:
        table = table + ' & \\textbf{' + label + '}'
    table += ' \\\\\n\\hline\n\\endfirsthead\n\n'

    table += '\n\\hline\n\\textbf{Task ID}'
    for label in args.labels:
        table = table + ' & \\textbf{' + label + '}'
    table += ' \\\\\n\\hline\n\\endhead\n\n'

    table += '\\hline\n\\endfoot\n\n'
    table += '\\hline\n\\endlastfoot\n\n'

    for env_name in env_names:
        table += env_name
        sota = -1
        for median_dict, std_dict in zip(medians, stds):
            median = median_dict[env_name]
            std = std_dict[env_name]

            if median > sota:
                sota = median - std

        for median_dict, std_dict in zip(medians, stds):
            median = median_dict[env_name]
            std = std_dict[env_name]

            # breakpoint()
            if median >= sota:
                # table += '& $\\mathbf{' + f'{median:1.2f} \\pm {std:1.2f}' + '}$ '
                table += '& $\\mathbf{' + f'{median:1.2f}' + '}$ '
            else:
                table += f'& ${median:1.2f}$ '
                # table += f'& ${median:1.2f} \\pm {std:1.2f}$ '
        table += '\\\\\n'

    table += '\hline\n'
    table += '\\caption{todo}\n'
    table += '\\end{longtable}'

    print(table)
    
    # print(env_names)
    
        # data.append(algo_data)
        # mean = np.median(data)
        # std_error = np.std(data) / np.sqrt(len(data))
        # mean_success_rate.append(mean)
        # std_errors.append(std_error)

        # print(f'{data_dir[len(prefix):]}: {mean:1.3f} +/- {std_error:1.3f}')

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