# reference implementation: https://github.com/mattcorsaro1/mj_pc
# with personal modifications


import math
import numpy as np
import open3d as o3d
import torch
import einops




"""
Generates numpy rotation matrix from quaternion

@param quat: w-x-y-z quaternion rotation tuple

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""
def quat2Mat(quat):
    if len(quat) != 4:
        print("Quaternion", quat, "invalid when generating transformation matrix.")
        raise ValueError

    # Note that the following code snippet can be used to generate the 3x3
    #    rotation matrix, we don't use it because this file should not depend
    #    on mujoco.
    '''
    from mujoco_py import functions
    res = np.zeros(9)
    functions.mju_quat2Mat(res, camera_quat)
    res = res.reshape(3,3)
    '''

    # This function is lifted directly from scipy source code
    #https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    rot_mat_arr = [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw), \
        2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw), \
        2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2]
    np_rot_mat = rotMatList2NPRotMat(rot_mat_arr)
    return np_rot_mat

"""
Generates numpy rotation matrix from rotation matrix as list len(9)

@param rot_mat_arr: rotation matrix in list len(9) (row 0, row 1, row 2)

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""
def rotMatList2NPRotMat(rot_mat_arr):
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat

"""
Generates numpy transformation matrix from position list len(3) and 
    numpy rotation matrix

@param pos:     list len(3) containing position
@param rot_mat: 3x3 rotation matrix as numpy array

@return t_mat:  4x4 transformation matrix as numpy array
"""
def posRotMat2Mat(pos, rot_mat):
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat

def posRotMat2Mat_batch(pos, rot_mat):
    t_mat = torch.eye(4, device=pos.device).repeat((pos.shape[0], 1, 1))
    t_mat[:, :3, :3] = rot_mat
    t_mat[:, :3, 3] = pos
    return t_mat

"""
Generates Open3D camera intrinsic matrix object from numpy camera intrinsic
    matrix and image width and height

@param cam_mat: 3x3 numpy array representing camera intrinsic matrix
@param width:   image width in pixels
@param height:  image height in pixels

@return t_mat:  4x4 transformation matrix as numpy array
"""
def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0,2]
    fx = cam_mat[0,0]
    cy = cam_mat[1,2]
    fy = cam_mat[1,1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

def depth2fgpcd_batch(depth, cam_params):
    # depth: (B, h, w)
    # fgpcd: (B, n, 3)
    # mask: (B, h, w)
    # import time

    # t = time.time()
    B, h, w = depth.shape

    # depth = torch.nan_to_num(depth, -1)
    # mask = depth > 0
    # mask = (depth <= 0.599/0.8)
    fgpcd = torch.zeros((B, h * w, 3), device=depth.device, dtype=torch.float16)
    fx, fy, cx, cy = cam_params
    pos_x, pos_y = torch.meshgrid(
        torch.arange(w, device=depth.device), 
        torch.arange(h, device=depth.device), indexing='ij')
    
    pos_x = einops.rearrange(pos_x, 'w h -> h w')
    pos_y = einops.rearrange(pos_y, 'w h -> h w')
    
    pos_x = pos_x.repeat((B, 1, 1))
    pos_y = pos_y.repeat((B, 1, 1))

    fx = fx.reshape((-1, 1, 1))
    fy = fy.reshape((-1, 1, 1))
    cx = cx.reshape((-1, 1, 1))
    cy = cy.reshape((-1, 1, 1))

    # pos_x = pos_x[mask]
    # pos_y = pos_y[mask]
    fgpcd[:, :, 0] = einops.rearrange((pos_x - cx) * depth / fx, 'b h w -> b (h w)')
    fgpcd[:, :, 1] = einops.rearrange((pos_y - cy) * depth / fy, 'b h w -> b (h w)')
    fgpcd[:, :, 2] = einops.rearrange(depth, 'b h w -> b (h w)')
    # print(time.time() - t)
    # breakpoint()
    return fgpcd


def lift_point_cloud_batch(depths, Ks, poses, max_depth=1.5, maintain_dims=False) -> torch.Tensor:
    # depths: [B, H, W] numpy array in meters
    # Ks: [B, 3, 3] numpy array
    # poses: [B, 4, 4] numpy array
    # masks: [B, H, W] numpy array in bool
    B, H, W = depths.shape

    # visualize scaled COLMAP poses

    cam_param = [Ks[:, 0, 0], Ks[:, 1, 1], Ks[:, 0, 2], Ks[:, 1, 2]]  # fx, fy, cx, cy
    # mask = np.ones_like(depth, dtype=bool)
    pcd = depth2fgpcd_batch(depths, cam_param)
    trans_pcd = batch_transform_point_cloud(pcd, poses)

    if maintain_dims:
        trans_pcd = einops.rearrange(trans_pcd, 'b (h w) c -> b h w c', h=H)

    return trans_pcd

    # breakpoint()
    pose = torch.linalg.inv(poses)

    trans_pcd = pose @ torch.cat(
        [pcd.T, torch.ones((1, pcd.shape[0]), device=depth.device)], axis=0
    )
    trans_pcd = trans_pcd[:3, :].T

    pcd_np = trans_pcd
    pcds.append(pcd_np)

    pcds = torch.cat(pcds, axis=0)
    return pcds

def batch_transform_point_cloud(pcd, transform):
    B, N, _ = pcd.shape

    pcd_1 = torch.cat([pcd, torch.ones((B, N, 1), device=pcd.device, dtype=pcd.dtype)], dim=-1)
    transform = transform.to(dtype=pcd.dtype)

    trans_pcd_1 = torch.einsum('bnd,bid->bni', pcd_1, transform)
    trans_pcd = trans_pcd_1[:, :, :-1]
    return trans_pcd
