import tyro
import h5py
import numpy as np
import os
from tqdm import tqdm
from PIL import Image

def main(hdf5_file: str, dest: str, key: str, demo_idx: int = 0):
    """Extract images from HDF5 file and save them as individual image files.
    
    Args:
        hdf5_file: Path to the input HDF5 file containing image data
        dest: Directory where the extracted images will be saved
    """
    os.makedirs(dest, exist_ok=True)
    
    with h5py.File(hdf5_file) as data:
        demo = f'demo_{demo_idx}'
        # Create a subdirectory for each demonstration
        demo_dir = os.path.join(dest, demo)
        os.makedirs(demo_dir, exist_ok=True)
        
        # Get the image sequence and flip horizontally
        images = np.asarray(data['data'][demo]['obs'][key])
        # images = np.flip(images, axis=1)
        
        # Save each frame as an image
        for frame_idx, frame in enumerate(images):
            # Convert to PIL Image and save
            img = Image.fromarray(frame)
            save_path = os.path.join(demo_dir, f'frame_{frame_idx:04d}.png')
            img.save(save_path)

if __name__ == '__main__':
    tyro.cli(main) 