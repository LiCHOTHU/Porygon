from argparse import ArgumentParser
import os
import shutil


def main():
    parser = ArgumentParser()
    parser.add_argument('--source')
    parser.add_argument('--dest')
    args = parser.parse_args()

    for root, dirs, files in os.walk(args.source):
        if 'data.json' in files:
            f = os.path.join(root, 'data.json')
            new_dir = os.path.join(args.dest, root[len(args.source):])
            os.makedirs(new_dir, exist_ok=True)
            shutil.copy(f, new_dir)
    

if __name__ == '__main__':
    main()