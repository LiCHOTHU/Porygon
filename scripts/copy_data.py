import os
import shutil
from argparse import ArgumentParser

from natsort import natsorted


def get_latest_checkpoint(checkpoint_dir):
    if os.path.isfile(checkpoint_dir):
        return checkpoint_dir

    onlyfiles = [
        f for f in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir, f))
    ]
    onlyfiles = natsorted(onlyfiles)
    best_file = onlyfiles[-1]
    return os.path.join(checkpoint_dir, best_file)


def main():
    parser = ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--target")
    args = parser.parse_args()

    root_dir = args.root
    target_dir = args.target

    for root, dirs, files in os.walk(root_dir):
        if len(files) > 0:
            if ".pth" in files[0]:
                # print(f'cleaning {root}')
                data_path = os.path.join(
                    root, f"data.json"
                )

                # Get the relative path and ensure it doesn't start with a slash
                rel_path = root[len(root_dir):]
                rel_path = rel_path.lstrip(os.sep)
                dest = os.path.join(target_dir, rel_path)
                os.makedirs(dest, exist_ok=True)
                dest_file = os.path.join(dest, "checkpoint.pth")
                if os.path.isfile(data_path):
                    print(f"copying {data_path} to {dest_file}")
                    shutil.copyfile(data_path, dest_file)
                else:
                    print(f"{data_path} does not exist, skipping")


if __name__ == "__main__":
    main()
