import os
import time
import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
from argparse import ArgumentParser

# import torch
# import torch.nn as nn
import imitation.utils.utils as utils
# from pyinstrument import Profiler
# from moviepy.editor import ImageSequenceClip
# import json

OmegaConf.register_new_resolver("eval", eval, replace=True)


# @hydra.main(config_path="config", config_name='evaluate', version_base=None)
def main():
    parser = ArgumentParser()
    parser.add_argument('--root')
    args = parser.parse_args()
    # device = cfg.device
    # seed = cfg.seed
    # torch.manual_seed(seed)
    # train_cfg = cfg.training
    # OmegaConf.resolve(cfg)
    
    # create model

    root_dir = args.root

    delete_count = all_count = 0
    for root, dirs, files in os.walk(root_dir):
        if len(files) > 0:
            if '.pth' in files[0]:
                print(f'cleaning {root}')
                latest = utils.get_latest_checkpoint(root)
                for file in files:
                    path = os.path.join(root, file)
                    if path != latest:
                        delete_count += 1
                        os.remove(path)
                    all_count += 1
    print(delete_count, all_count)
                # print('----------------------------------------')
                # print(latest)
                # print('-------')
                # print(files)
                
            # extensions = set()
            # for file in files:
            #     extensions.add(file[file.index('.') + 1:])
                
            # # print(root, dirs, files)
            # # if len(extensions) == 1:
            # # if 'pth' in extensions:
            #     print(extensions)
            # breakpoint()
    exit(0)
    

    if cfg.checkpoint_path is None:
        # Basically if you don't provide a checkpoint path it will automatically find one corresponding
        # to the experiment/variant name you provide
        checkpoint_path, _ = utils.get_experiment_dir(cfg, evaluate=False, allow_overlap=True)
        checkpoint_path = utils.get_latest_checkpoint(checkpoint_path)
    else:
        checkpoint_path = utils.get_latest_checkpoint(cfg.checkpoint_path)
    state_dict = utils.load_state(checkpoint_path)
    
    if 'config' in state_dict:
        print('autoloading based on saved parameters')
        model = instantiate(state_dict['config']['algo']['policy'], 
                            shape_meta=cfg.task.shape_meta)
    else:
        model = instantiate(cfg.algo.policy,
                            shape_meta=cfg.task.shape_meta)
    model.to(device)
    model.eval()

    model.load_state_dict(state_dict['model'])

    env_runner = instantiate(cfg.task.env_runner)
    
    print(save_dir)

    def save_video_fn(video_chw, env_name, idx):
        video_dir = os.path.join(save_dir, 'videos', env_name)
        os.makedirs(video_dir, exist_ok=True)
        save_path = os.path.join(video_dir, f'{idx}.mp4')
        clip = ImageSequenceClip(list(video_chw.transpose(0, 2, 3, 1)), fps=24)
        clip.write_videofile(save_path, fps=24, verbose=False, logger=None)

    # policy = lambda obs, task_id: model.get_action(obs, task_id)
    if train_cfg.do_profile:
        profiler = Profiler()
        profiler.start()
    rollout_results = env_runner.run(model, n_video=cfg.rollout.n_video, do_tqdm=train_cfg.use_tqdm, save_video_fn=save_video_fn)
    if train_cfg.do_profile:
        profiler.stop()
        profiler.print()
    print(
        f"[info]     success rate: {rollout_results['rollout']['overall_success_rate']:1.3f} \
            | environments solved: {rollout_results['rollout']['environments_solved']}")

    # videos = {key.split('/')[1]: value.data for key, value in rollout_results.items() if 'rollout_videos' in key}
        


    # videos = rollout_results['rollout_videos']
    # del rollout_results['rollout_videos']
    with open(os.path.join(save_dir, 'data.json'), 'w') as f:
        json.dump(rollout_results, f)

    


if __name__ == "__main__":
    main()
