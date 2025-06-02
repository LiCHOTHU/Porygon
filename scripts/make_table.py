from argparse import ArgumentParser
import os
import json
import numpy as np


def walk_dir(folder):
    data = []
    for root, dirs, files in os.walk(folder):
        if 'data.json' in files:
            with open(os.path.join(root, 'data.json'), 'r') as f:
                data_dict = json.load(f)
            data.append(data_dict['rollout']['overall_success_rate'])
    mean_success_rate = np.mean(data)
    if len(data):
        std_error = np.std(data) / np.sqrt(len(data))
    else:
        std_error = 0
    return mean_success_rate, std_error

def make_header(column_names, column_config=None):
    header = "\\begin{tabular}\n{"
    if column_config is not None:
        header += column_config
    else:
        header += 'c' * len(column_names)
    header += "}\n\\toprule\n"
    
    header += ' & '.join(column_names) 
    header += ' \\\\\n\\midrule\n'
    return header
    
        

def main():
    parser = ArgumentParser()
    parser.add_argument('--data-dirs', nargs='*', required=True)
    parser.add_argument('--column-names', nargs='*', required=True)
    parser.add_argument('--row-names', nargs='*', required=True)
    parser.add_argument('--chunk-sizes', nargs='*', type=int)
    parser.add_argument('--midrules', nargs='*', type=int)
    parser.add_argument('--h', type=int, required=True)
    parser.add_argument('--w', type=int, required=True)
    parser.add_argument('--prefix')
    parser.add_argument('--column-config')
    parser.add_argument('--do-std-error', action='store_true')
    parser.add_argument('--max-axis', type=int, default=0)
    args = parser.parse_args()
    
    column_names = args.column_names
    row_names = args.row_names
    prefix = args.prefix
    max_axis = args.max_axis
    midrules = args.midrules
    
    data = []
    for data_dir in args.data_dirs:
        if prefix is not None:
            data_dir = os.path.join(prefix, data_dir)
        mean_success_rate, std_error = walk_dir(data_dir)
        data.append((mean_success_rate, std_error))
        
    data = np.array(data).reshape(args.h, args.w, 2)
    # replace missing values with -1
    data[data != data] = -1
    
    H, W, _ = data.shape
    assert len(column_names) in (W, W+1)
    assert len(row_names) == H
    
    if len(column_names) == W:
        column_names = [''] + column_names
    
    table = make_header(column_names, args.column_config)
    
    means = data[..., 0]
    stds = data[..., 1]
    # argmaxes = np.argmax(means, axis=0)
    
    if max_axis == 0:
        chunk_sizes = [H] if args.chunk_sizes is None else args.chunk_sizes
        assert sum(chunk_sizes) == H
    elif max_axis == 1:
        chunk_sizes = [W] if args.chunk_sizes is None else args.chunk_sizes
        assert sum(chunk_sizes) == W
    
    indices = np.cumsum(chunk_sizes)[:-1]
    chunks = np.split(means, indices, axis=max_axis)
    argmax_mask = []
    for chunk in chunks:
        argmax_indices = np.argmax(chunk, axis=max_axis)
        mask = np.zeros_like(chunk, dtype=int)
        
        if max_axis == 0:  # Argmax along columns
            mask[argmax_indices, np.arange(chunk.shape[1])] = 1
        elif max_axis == 1:  # Argmax along rows
            mask[np.arange(chunk.shape[0]), argmax_indices] = 1
        argmax_mask.append(mask)
    argmax_mask = np.concatenate(argmax_mask, axis=max_axis)
    
    
    # i = 0
    # for chunk_size in chunk_sizes:
    for i in range(H):
        table += f'{row_names[i]} '
        for j in range(W):
            if means[i, j] == -1:
                table += '& - '
            else:
                if args.do_std_error:
                    if argmax_mask[i, j]:
                        table += '& $\\mathbf{' + f'{means[i,j]:1.3f} \\pm {stds[i,j]:1.3f}' + '}$ '
                    else:
                        table += f'& ${means[i,j]:1.3f} \\pm {stds[i,j]:1.3f}$ '
                else:
                    if argmax_mask[i, j]:
                        table += '& $\\mathbf{' + f'{means[i,j]:1.3f}' + '}$ '
                    else:
                        table += f'& ${means[i,j]:1.3f}$ '
        table += '\\\\\n'
        if midrules is not None and i in midrules:
            table += '\\midrule\n'

    table += '\\bottomrule\n\\end{tabular}'
            
            
        
    print(table)
    



if __name__ == '__main__':
    main()