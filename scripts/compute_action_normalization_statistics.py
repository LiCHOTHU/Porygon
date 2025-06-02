import os
import time
import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import Path
import warnings

import torch
import torch.nn as nn
from pyinstrument import Profiler
from imitation.utils.logger import Logger
from imitation.utils.data_utils import copy_data_pace
import imitation.utils.point_cloud_utils as pcu
import pytorch3d.transforms as pt
import einops
import numpy as np
import gc

OmegaConf.register_new_resolver("eval", eval, replace=True)
os.environ["WANDB_INIT_TIMEOUT"] = "300"

# TODO: should probably remove for the final release
torch.set_printoptions(sci_mode=False)



@hydra.main(config_path="config", version_base=None)
def main(cfg):
    device = cfg.device
    seed = cfg.seed
    torch.manual_seed(seed)
    train_cfg = cfg.training

    if cfg.pace_copy:
        pace_tmp_dir = os.getenv('TMPDIR')
        copy_data_pace(cfg, pace_tmp_dir)
        dataset = instantiate(cfg.task.dataset,
                              data_prefix=os.path.join(pace_tmp_dir, 'data'))
    else:
        dataset = instantiate(cfg.task.dataset)
    train_dataloader = instantiate(
        cfg.train_dataloader, 
        dataset=dataset)
    
    all_actions = []
    for data in tqdm(train_dataloader, disable=not train_cfg.use_tqdm):
        actions = data['abs_actions']
        ee_mat_inv = data['obs']['hand_mat_inv'].squeeze(1)
        actions_pos, actions_rot, actions_gripper = torch.split(actions, [3, 3, 1], dim=-1)
        rot_mat = pt.axis_angle_to_matrix(actions_rot)
        action_mat = pcu.pos_rot_mat_to_mat(actions_pos, rot_mat)
        action_mat_eecf = torch.einsum('bij,bnjk->bnik', ee_mat_inv, action_mat)
        breakpoint()
        action_pos, action_rot_6d = pcu.matrix_to_pos_6d(action_mat_eecf)
        actions = torch.cat((action_pos, action_rot_6d, actions_gripper), dim=-1)
        
        actions = einops.rearrange(actions, 'b n d -> (b n) d').cpu().numpy()
        all_actions.append(actions)
    all_actions = np.concatenate(all_actions, axis=0)

    p1 = np.percentile(all_actions, 1, axis=0)
    p99 = np.percentile(all_actions, 99, axis=0)
    print('p1', p1)
    print('p99', p99)
    breakpoint()

if __name__ == "__main__":
    main()
    
    
    
# class Monkey():
#     def speak(self):
#         print('oo oo ah ah')
        
# def speak2(self):
#     print('i fucked your wife')
    
# monkey = Monkey()
# monkey.speak = speak2
