#!/usr/bin/env python3

import os
import h5py
import argparse
from tqdm import tqdm
import numpy as np

import torch
import imitation.utils.pytorch3d_transforms as p3d
import imitation.utils.point_cloud_utils as pcu

def pos_ori_to_mat(pos: np.ndarray, ori: np.ndarray):
    pos = torch.tensor(pos)
    ori = torch.tensor(ori)
    ori = p3d.axis_angle_to_matrix(ori)
    mat = pcu.pos_rot_mat_to_mat(pos, ori)
    return mat.numpy()

def pos_ori_to_inv_mat(pos: np.ndarray, ori: np.ndarray):
    mat = pos_ori_to_mat(pos, ori)
    inv_mat = torch.linalg.inv(torch.tensor(mat))
    return inv_mat.numpy()

def ori_to_rot_6d(actions: np.ndarray):
    actions = torch.tensor(actions)
    pos, rot, openness = actions.split([3, 3, 1], dim=-1)
    rot_6d = p3d.matrix_to_rotation_6d(p3d.axis_angle_to_matrix(rot))
    return torch.cat([pos, rot_6d, openness], dim=-1).numpy()

act_mapping = {
    "abs_actions": (("UR5_left_cartesian_actions",), ori_to_rot_6d),
}

obs_mapping = {
    "wrist_depth": "wrist_left_realsense_depth",
    "wrist_image": "wrist_left_realsense_rgb",
    "wrist_extrinsic": (("wrist_left_realsense_position", "wrist_left_realsense_orientation"), pos_ori_to_mat),
    "wrist_intrinsic": "wrist_left_realsense_intrinsics",
    "front_depth": "front_zed_depth",
    "front_image": "front_zed_rgb",
    "front_extrinsic": (("front_zed_position", "front_zed_orientation"), pos_ori_to_mat),
    "front_intrinsic": "front_zed_intrinsics",
    "hand_mat": (("UR5_left_ee_trans", "UR5_left_ee_orientation"), pos_ori_to_mat),
    "hand_mat_inv": (("UR5_left_ee_trans", "UR5_left_ee_orientation"), pos_ori_to_inv_mat),
    "eef_pos": "UR5_left_ee_trans",
    "eef_ori": "UR5_left_ee_orientation",
    "eef_openness": "UR5_left_gripper_opening",
}

def remap_observations(source_path: str, target_path: str, obs_mapping: dict):
    """
    Remap observation keys in an HDF5 file according to the provided mapping dictionary.
    
    Args:
        source_path (str): Path to source HDF5 file
        target_path (str): Path to save the remapped HDF5 file
        obs_mapping (dict): Dictionary mapping new observation keys to old ones
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    with h5py.File(source_path, 'r') as f_src, h5py.File(target_path, 'w') as f_dst:
        # Create data group in target file
        data_grp = f_dst.create_group('data')
        
        # Copy global attributes
        for attr in f_src['data'].attrs:
            data_grp.attrs[attr] = f_src['data'].attrs[attr]
        
        # Process each demo
        demos = list(f_src['data'].keys())
        for demo in tqdm(demos, desc="Processing demos"):
            # Create demo group
            demo_grp = data_grp.create_group(demo)
            
            # Copy all non-obs data and attributes
            for key in f_src[f'data/{demo}']:
                if key != 'obs' and key != 'actions':
                    f_src[f'data/{demo}/{key}'].copy(f_src[f'data/{demo}/{key}'], demo_grp)
            
            # Copy demo attributes
            for attr in f_src[f'data/{demo}'].attrs:
                demo_grp.attrs[attr] = f_src[f'data/{demo}'].attrs[attr]
            
            # Create act group
            act_grp = demo_grp.create_group('actions')
            
            # Remap actions
            for new_key, old_key in act_mapping.items():
                if type(old_key) == tuple:
                    input_keys, func = old_key
                    input_data = [f_src[f'data/{demo}/actions/{key}'][()] for key in input_keys]
                    output_data = func(*input_data)
                    act_grp.create_dataset(new_key, data=output_data)
                elif old_key in f_src[f'data/{demo}/actions']:
                    # Copy data with new key name
                    src_dataset = f_src[f'data/{demo}/actions/{old_key}']
                    dst_dataset = act_grp.create_dataset(new_key, data=src_dataset[()])
                    # Copy attributes if any
                    for attr in src_dataset.attrs:
                        dst_dataset.attrs[attr] = src_dataset.attrs[attr]
                else:
                    print(f"Warning: Key '{old_key}' not found in demo {demo}")
            
            # Create obs group
            obs_grp = demo_grp.create_group('obs')
            
            # Remap observations
            for new_key, old_key in obs_mapping.items():
                if type(old_key) == tuple:
                    input_keys, func = old_key
                    input_data = [f_src[f'data/{demo}/obs/{key}'][()] for key in input_keys]
                    output_data = func(*input_data)
                    obs_grp.create_dataset(new_key, data=output_data)
                elif old_key in f_src[f'data/{demo}/obs']:
                    # Copy data with new key name
                    src_dataset = f_src[f'data/{demo}/obs/{old_key}']
                    dst_dataset = obs_grp.create_dataset(new_key, data=src_dataset[()])
                    # Copy attributes if any
                    for attr in src_dataset.attrs:
                        dst_dataset.attrs[attr] = src_dataset.attrs[attr]
                else:
                    print(f"Warning: Key '{old_key}' not found in demo {demo}")
            
            # # Copy any remaining observation keys that weren't in the mapping
            # for key in f_src[f'data/{demo}/obs']:
            #     if key not in obs_mapping:
            #         src_dataset = f_src[f'data/{demo}/obs/{key}']
            #         dst_dataset = obs_grp.create_dataset(key, data=src_dataset[()])
            #         # Copy attributes if any
            #         for attr in src_dataset.attrs:
            #             dst_dataset.attrs[attr] = src_dataset.attrs[attr]
        
        # Copy any additional groups (like 'mask')
        for key in f_src:
            if key != 'data':
                f_src[key].copy(f_src[key], f_dst)

def main():
    parser = argparse.ArgumentParser(description='Remap observation keys in an HDF5 file')
    parser.add_argument('source', help='Path to source HDF5 file')
    parser.add_argument('target', help='Path to save the remapped HDF5 file')
    parser.add_argument('--allow-overwrite', action='store_true', 
                      help='Allow overwriting existing output file')
    
    args = parser.parse_args()
    
    # Check if target file exists
    if os.path.exists(args.target) and not args.allow_overwrite:
        print(f"Error: Output file {args.target} already exists. Use --allow-overwrite to overwrite.")
        return
    
    print(f"Processing {args.source}...")
    print(f"Output will be saved to {args.target}")
    
    remap_observations(args.source, args.target, obs_mapping)
    print("Done!")

if __name__ == '__main__':
    main() 