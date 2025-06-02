import tyro
import h5py
import numpy as np
import os
from tqdm import tqdm
from moviepy import ImageSequenceClip

def main(hdf5_file: str, dest: str):
    os.mkdir(dest)
    with h5py.File(hdf5_file) as data:
        for demo in tqdm(data['data']):
            video = np.asarray(data['data'][demo]['obs']['agentview_image'])
            video = np.flip(video, axis=1)
            save_path = os.path.join(dest, f'{demo}.mp4')
            clip = ImageSequenceClip(list(video), fps=24)
            clip.write_videofile(save_path, fps=24, logger=None)
        
if __name__ == '__main__':
    tyro.cli(main)