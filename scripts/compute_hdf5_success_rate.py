import os
import h5py
import argparse
from tqdm import tqdm

def compute_success_rate(hdf5_path: str) -> float:
    """
    Compute success rate across all demos in an HDF5 file.
    A demo is considered successful if it contains any nonzero reward.

    Args:
        hdf5_path: Path to HDF5 file containing demonstrations

    Returns:
        float: Success rate (between 0 and 1)
    """
    with h5py.File(hdf5_path, "r") as f:
        demos = list(f["data"].keys())
        num_successful = 0
        
        for demo in tqdm(demos, desc="Processing demos"):
            rewards = f[f"data/{demo}/rewards"][()]
            if (rewards != 0).any():
                num_successful += 1
                
        success_rate = num_successful / len(demos)
        return success_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hdf5_path",
        type=str,
        required=True,
        help="Path to input HDF5 dataset"
    )
    args = parser.parse_args()

    if not os.path.exists(args.hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found at {args.hdf5_path}")

    success_rate = compute_success_rate(args.hdf5_path)
    print(f"Success rate: {success_rate:.2%}")
