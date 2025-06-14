import os
import time
import hydra
import wandb
import logging
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
import torch.nn as nn
import imitation.utils.utils as utils
from pyinstrument import Profiler
from imitation.utils.logger import Logger
from imitation.dataset.utils import copy_data_pace
from pathlib import Path
import open3d as o3d
import numpy as np
from imitation.utils.point_cloud_utils import lift_point_cloud_batch
import imitation.envs.utils as eu

# Disable scientific notation for numpy and torch
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)


OmegaConf.register_new_resolver("eval", eval, replace=True)
os.environ["WANDB_INIT_TIMEOUT"] = "300"

def visualize_point_cloud(pcd_np, rgb=None):
    """Visualize point cloud using Open3D"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 1
    vis.get_render_option().background_color = np.array([0, 0, 0])
    vis.run()
    vis.destroy_window()

@hydra.main(config_path="../config", version_base=None)
def main(cfg):
    dataset = utils.make_dataset(cfg)
    train_dataloader = instantiate(
        cfg.train_dataloader, 
        dataset=dataset)
    
    for data in train_dataloader:
        data = utils.map_tensor_to_device(data, 'cuda')
        obs_data = data['obs']
        
        # Get shape metadata from dataset
        shape_meta = cfg.task.shape_meta
        
        # Lists to store point clouds and RGB values from all cameras
        all_pcds = []
        all_rgbs = []
        
        # Process each camera's point cloud
        for camera_name in eu.list_cameras(shape_meta):
            intrinsic_key = eu.camera_name_to_intrinsic_key(camera_name)
            extrinsic_key = eu.camera_name_to_extrinsic_key(camera_name)
            depth_key = eu.camera_name_to_depth_key(camera_name)
            
            # Get camera data
            depths = obs_data[depth_key].squeeze(2).to(torch.float32) / 1000  # Convert to meters
            intrinsics = obs_data[intrinsic_key]
            extrinsics = obs_data[extrinsic_key]
            
            # Reshape for batch processing
            B, T, H, W = depths.shape
            depths = depths.reshape(B, 1, H, W)
            intrinsics = intrinsics.reshape(B, 1, 3, 3)
            extrinsics = extrinsics.reshape(B, 1, 4, 4)
            
            # Lift depth to point cloud
            pcd = lift_point_cloud_batch(
                depths,        # [B, 1, H, W]
                intrinsics,    # [B, 1, 3, 3]
                extrinsics,    # [B, 1, 4, 4]
                keepdims=True
            )
            
            # Convert to numpy for visualization
            pcd_np = pcd[0, 0].cpu().numpy()  # Take first batch, first frame
            
            # Get RGB if available
            rgb = None
            if f"{camera_name}_rgb" in obs_data:
                rgb = obs_data[f"{camera_name}_rgb"][0, 0].cpu().numpy()
                rgb = rgb.reshape(-1, 3)  # Reshape to match point cloud
            
            all_pcds.append(pcd_np)
            all_rgbs.append(rgb)
        
        # Combine all point clouds
        combined_pcd = np.vstack(all_pcds)
        
        # Combine all RGB values if available
        combined_rgb = None
        if all(rgb is not None for rgb in all_rgbs):
            combined_rgb = np.vstack(all_rgbs)
        
        print(f"Visualizing combined point cloud from all cameras")
        visualize_point_cloud(combined_pcd, combined_rgb)
        
        # Only process first batch for testing
        break

if __name__ == "__main__":
    main()
