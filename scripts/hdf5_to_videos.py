import tyro
import h5py
import numpy as np
import os
from tqdm import tqdm
from moviepy import ImageSequenceClip
from PIL import Image
from natsort import natsorted

def main(hdf5_file: str, dest: str, num_videos: int = None):
    os.makedirs(dest, exist_ok=True)
    with h5py.File(hdf5_file) as data:
        demos = natsorted(list(data['data']))[:num_videos] if num_videos is not None else data['data']

        for demo in tqdm(demos):
            video = np.asarray(data['data'][demo]['obs']['agentview_image'])
            # Resize each frame to 224x224
            resized_frames = []
            for frame in video:
                img = Image.fromarray(frame)
                resized_img = img.resize((224, 224), Image.Resampling.LANCZOS)
                resized_frames.append(np.array(resized_img))
            video = np.array(resized_frames)
            # video = np.flip(video, axis=1)
            save_path = os.path.join(dest, f'{demo}.mp4')
            clip = ImageSequenceClip(list(video), fps=24)
            clip.write_videofile(save_path, fps=24, logger=None)
        
if __name__ == '__main__':
    tyro.cli(main)