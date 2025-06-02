import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from imitation.algos.utils.obs_core import CropRandomizer
from imitation.utils.geometry import batch_axis_angle_to_rotation_matrix, batch_rotation_matrix_to_axis_angle
import imitation.utils.camera_utils as cu

import imitation.envs.utils as eu
import einops
import matplotlib.pyplot as plt


class IdentityAug(nn.Module):
    def __init__(self, shape_meta=None, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class TranslationAug(nn.Module):
    """
    Utilize the random crop from robomimic.
    """

    def __init__(
        self,
        shape_meta,
        translation,
        use_image=True,
        use_depth=False,
    ):
        super().__init__()

        self.randomizers = {}
        self.shape_meta = shape_meta
        self.use_image = use_image
        self.use_depth = use_depth

        obs_meta = shape_meta['observation']

        # for camera_name in shape_meta['observation']['rgb'].items():
        for camera_name in eu.list_cameras(shape_meta):
            channels = 0
            rgb_name = eu.camera_name_to_image_key(camera_name)
            depth_name = eu.camera_name_to_depth_key(camera_name)
            if use_image and rgb_name in obs_meta['rgb']:
                channels += 3
                size = obs_meta['rgb'][rgb_name][1:]
            if use_depth and depth_name in obs_meta['depth']:
                channels += 1
                size = obs_meta['depth'][depth_name][1:]
            
            input_shape = [channels] + size

            # pc_full_name = camera_name + '_pointcloud_full'
            # if pc_full_name in self.shape_meta['observation']['pointcloud']:
            #     input_shape[0] += 3

            input_shape = tuple(input_shape)

            self.pad_translation = translation // 2
            pad_output_shape = (
                input_shape[0],
                input_shape[1] + translation,
                input_shape[2] + translation,
            )

            crop_randomizer = CropRandomizer(
                input_shape=pad_output_shape,
                crop_height=input_shape[1],
                crop_width=input_shape[2],
            )
            self.randomizers[input_shape] = crop_randomizer

    def forward(self, data):
        if self.training:

            obs_data = data['obs']
            for camera_name in eu.list_cameras(self.shape_meta):
                x = []
                rgb_name = eu.camera_name_to_image_key(camera_name)
                depth_name = eu.camera_name_to_depth_key(camera_name)
                if rgb_name in obs_data:
                    x.append(obs_data[rgb_name])
                if depth_name in obs_data and self.use_depth:
                    x.append(obs_data[depth_name] / 1000)

                x = torch.cat(x, dim=2)
                
                batch_size, temporal_len, img_c, img_h, img_w = x.shape

                input_shape = (img_c, img_h, img_w)
                crop_randomizer = self.randomizers[input_shape]

                intrinsic_name = eu.camera_name_to_intrinsic_key(camera_name)
                if intrinsic_name in obs_data:
                    intrinsics = obs_data[intrinsic_name]
                    # intrinsics = einops.rearrange(intrinsics, 'b t i j -> (b t) i j')
                    intrinsics = cu.pad_update_intrinsics(intrinsics, self.pad_translation)
                else:
                    intrinsics = None

                x = x.reshape(batch_size, temporal_len * img_c, img_h, img_w)
                out = F.pad(x, pad=(self.pad_translation,) * 4, mode="replicate")
                out, intrinsics = crop_randomizer.forward_in((out, intrinsics))
                out = out.reshape(batch_size, temporal_len, img_c, img_h, img_w)

                if rgb_name in obs_data:
                    rgb = out[:, :, :3].to(dtype=torch.uint8)
                    out = out[:, :, 3:]
                    obs_data[rgb_name] = rgb
                if depth_name in obs_data and self.use_depth:
                    depth = (out * 1000).to(dtype=torch.uint16)
                    obs_data[depth_name] = depth

                if intrinsics is not None:
                    obs_data[intrinsic_name] = intrinsics
        return data


class ImgColorJitterAug(torch.nn.Module):
    """
    Conduct color jittering augmentation outside of proposal boxes
    """

    def __init__(
        self,
        shape_meta,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        epsilon=0.05,
    ):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.epsilon = epsilon
        self.shape_meta = shape_meta

    def forward(self, data):
        if self.training and np.random.rand() > self.epsilon:
            for name in self.shape_meta['observation']['rgb']:
                data['obs'][name] = self.color_jitter(data['obs'][name])
        return data


class ImgColorJitterGroupAug(torch.nn.Module):
    """
    Conduct color jittering augmentation outside of proposal boxes
    """

    def __init__(
        self,
        shape_meta,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        epsilon=0.05,
    ):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.epsilon = epsilon
        self.shape_meta = shape_meta

    def forward(self, x):
        raise NotImplementedError
        if self.training and np.random.rand() > self.epsilon:
            out = self.color_jitter(x)
        else:
            out = x
        return out


class BatchWiseImgColorJitterAug(torch.nn.Module):
    """
    Color jittering augmentation to individual batch.
    This is to create variation in training data to combat
    BatchNorm in convolution network.
    """

    def __init__(
        self,
        shape_meta,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        epsilon=0.1,
    ):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.epsilon = epsilon
        self.shape_meta = shape_meta

    def forward(self, data):
        if self.training:
            obs_data = data['obs']
            for camera_name in eu.list_cameras(self.shape_meta):
                image_name = eu.camera_name_to_image_key(camera_name)
                if image_name not in obs_data:
                    continue

                x = obs_data[image_name]
                mask = torch.rand((x.shape[0], *(1,)*(len(x.shape)-1)), device=x.device) > self.epsilon
                
                jittered = self.color_jitter(x)

                out = mask * jittered + torch.logical_not(mask) * x

                obs_data[image_name] = out
        
        # self.count += 1
        return data


class DataAugGroup(nn.Module):
    """
    Add augmentation to multiple inputs
    """

    def __init__(self, aug_list, shape_meta):
        super().__init__()
        aug_list = [aug(shape_meta) for aug in aug_list]
        self.aug_layer = nn.Sequential(*aug_list)

    def forward(self, data):
        return self.aug_layer(data)
    

class PointcloudRotationAug(nn.Module):
    def __init__(self, shape_meta, output_frame='hand', action_space='world'):
        super().__init__()

        assert output_frame in ('hand', 'world')
        self.output_frame = output_frame

        assert action_space in ('hand', 'world')
        self.action_space = action_space

        self.pointcloud_keys = list(shape_meta['observation']['pointcloud'].keys())
        assert len(self.pointcloud_keys) == 1, 'several pointclouds not supported'

        # TODO: this is hardcoded for libero and I get it through
        # self.env.env.robots[0].controller.output_max
        self.action_scale = torch.tensor([0.05, 0.05, 0.05, 0.5 , 0.5 , 0.5 ], device='cuda')

    def forward(self, data):
        obs_data = data['obs']
        pointcloud = obs_data[self.pointcloud_keys[0]]
        B, T, N, D = pointcloud.shape
        
        hand_mat = obs_data['hand_mat']
        hand_mat_inv = obs_data['hand_mat_inv']

        # whether the pointcloud is in the world coordinate frame. If it is, augment in the hand frame
        world = self.pointcloud_keys[0] == 'world_pointcloud'
        if world:
            pcd_1 = torch.cat((pointcloud[..., :3], torch.ones((B, T, N, 1), device=pointcloud.device)), dim=-1)
            # pointcloud_xyz = (hand_mat_inv @ pcd_1.T).T[:, :3]
            pointcloud_xyz = torch.einsum('bfnj,bfij->bfni', pcd_1, hand_mat_inv)[..., :-1]
        else:
            pointcloud_xyz = pointcloud[:, :, :, :3]
        pointcloud_other = pointcloud[:, :, :, 3:]

        
        actions = data['actions']
        # Extract rotation matrices
        hand_rot_mat = hand_mat[:, :, :3, :3]
        hand_rot_mat_inv = hand_mat_inv[:, :, :3, :3]
        # B = actions.shape[0]
        # frame_stack = hand_mat.shape[1]

        theta = torch.rand(B, device=actions.device) * 2 * np.pi
        axes = torch.tensor([(1., 0., 0.)] * B, device=actions.device)
        rot_mats = batch_axis_angle_to_rotation_matrix(axes, theta)

        # transform into hand coordinate frame, rotate, and then back into world
        if self.action_space == 'world':
            # transform to hand action space, rotate, and then transform back to world
            action_transform = hand_rot_mat[:, -1] @ rot_mats @ hand_rot_mat_inv[:, -1]
        elif self.action_space == 'hand':
            # transform to hand action space and rotate. Don't transform back
            action_transform = rot_mats @ hand_rot_mat_inv[:, -1]

        if actions.shape[-1] == 4:
            action_pos = actions[:, :, :3]
            actions_rot = None
            action_gripper = actions[:, :, 3:]
        
            actions_transformed = torch.einsum('btj,bij->bti', action_pos, action_transform)
            actions_transformed = torch.concatenate((actions_transformed, action_gripper), dim=2)
        elif actions.shape[-1] == 7:
            _, H, _ = actions.shape
            # In this we assume that rotations are in the axis angle representation used here
            # https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L515
            # EDIT: ignore this line: so we can transform them by simply performing the same transformation as everything else
            action_pos = actions[:, :, :3]
            actions_pos_transformed = torch.einsum('btj,bij->bti', action_pos, action_transform)

            action_rot = actions[:, :, 3:6]
            action_rot = action_rot * self.action_scale[3:]
            action_rot_angle = torch.linalg.norm(action_rot, dim=-1, keepdims=True)
            action_rot_axis = action_rot / (action_rot_angle + 1e-8)
            action_rot_mats = batch_axis_angle_to_rotation_matrix(
                action_rot_axis.view(B*H, 3), 
                action_rot_angle.view(B*H)).view(B, H, 3, 3)
            
            ptp_inv = hand_rot_mat[:, -1] @ rot_mats @ hand_rot_mat_inv[:, -1]
            p_t_inv_p_inv = torch.linalg.inv(ptp_inv)
            rot_mats_inv = torch.linalg.inv(rot_mats)

            # PTP*APT*P*
            action_rot_mats_transformed = torch.einsum('bij,btjk,bkl->btil', ptp_inv, action_rot_mats, p_t_inv_p_inv)
            # action_rot_mats_transformed = torch.einsum('bij,btjk,bkl->btil', rot_mats, action_rot_mats, rot_mats_inv)
            # action_rot_mats_transformed = torch.einsum('btjk,bkl->btjl', action_rot_mats, rot_mats_inv)

            action_axis_trans, action_angle_trans = batch_rotation_matrix_to_axis_angle(action_rot_mats_transformed.reshape(B * H, 3, 3))
            action_rot_transformed = (action_axis_trans * action_angle_trans.unsqueeze(-1)).view(B, H, 3) / self.action_scale[3:]

            # actions_pos_transformed = torch.einsum('btj,bij->bti', action_pos, action_transform)
            # action_rot_transformed = torch.einsum('btj,bij->bti', action_rot, action_transform)

            action_gripper = actions[:, :, 6:]

            # actions_transformed = torch.concatenate((actions_pos_transformed, 
            #                                          action_rot,
            #                                          action_gripper), dim=2)
            actions_transformed = torch.concatenate((actions_pos_transformed, 
                                                     action_rot_transformed,
                                                     action_gripper), dim=2)
        

        pointcloud_xyz_transformed = torch.einsum('bfnj,bfij->bfni', pointcloud_xyz, rot_mats.view(B, 1, 3, 3))

        # if necessary, transform back to the world coordinate frame
        if world:
            pcd_xyz_transformed_1 = torch.cat((pointcloud_xyz_transformed[..., :3], 
                                               torch.ones((B, T, N, 1), device=pointcloud.device)), 
                                               dim=-1)
            pointcloud_xyz_transformed = torch.einsum('bfnj,bfij->bfni', pcd_xyz_transformed_1, hand_mat)[..., :-1]

        pointcloud_transformed = torch.concatenate((pointcloud_xyz_transformed.reshape(B, T, N, 3), pointcloud_other), dim=-1)

        # actions_cumsum = torch.cumsum(actions_transformed[..., :3], dim=1) * self.action_scale[:3]
        # red = einops.repeat(torch.tensor((1, 0, 1), device='cuda'), 'd -> 128 8 d')
        # actions_color = torch.concatenate((actions_cumsum, red), dim=-1)
        # pc_with_actions = torch.cat((pointcloud_transformed.squeeze(), actions_color), dim=1)
        # for i in range(128):
        #     pc = pc_with_actions[i].cpu().numpy()
        #     pc_comp = pc.astype(np.float16)

        #     breakpoint()
        #     show_point_cloud(pc)

        

        data['actions'] = actions_transformed
        data['transform'] = rot_mats
        obs_data[self.pointcloud_keys[0]] = pointcloud_transformed

        return data
    

class ZAxisRotationAug(nn.Module):
    def __init__(self, shape_meta, output_frame='hand', action_space='world'):
        super().__init__()

        assert output_frame in ('hand', 'world')
        self.output_frame = output_frame

        assert action_space in ('hand', 'world')
        self.action_space = action_space

        self.pointcloud_keys = list(shape_meta['observation']['pointcloud'].keys())
        assert set(self.pointcloud_keys) == {'world_pointcloud'}, 'only world pointclouds supported'

        # TODO: this is hardcoded for libero and I get it through
        # self.env.env.robots[0].controller.output_max
        self.action_scale = torch.tensor([0.05, 0.05, 0.05, 0.5 , 0.5 , 0.5 ], device='cuda')

    def forward(self, data):
        obs_data = data['obs']
        pointcloud = obs_data[self.pointcloud_keys[0]]
        B, T, N, D = pointcloud.shape
        
        hand_mat = obs_data['hand_mat']
        hand_mat_inv = obs_data['hand_mat_inv']

        # If in the world frame we need to translate such that the end effector position is 0
        # If in the hand frame we need to rotate into the world frame
        # world = self.pointcloud_keys[0] == 'world_pointcloud'
        # if world:
            # pcd_1 = torch.cat((pointcloud[..., :3], torch.ones((B, T, N, 1), device=pointcloud.device)), dim=-1)
            # # pointcloud_xyz = (hand_mat_inv @ pcd_1.T).T[:, :3]
            # pointcloud_xyz = torch.einsum('bfnj,bfij->bfni', pcd_1, hand_mat_inv)[..., :-1]

        pointcloud_xyz = pointcloud[:, :, :, :3]
        ee_pos = einops.rearrange(hand_mat[:, :, :3, 3:], 'B F I J -> B F J I')
        pointcloud_xyz = pointcloud_xyz - ee_pos
        # else:
        #     pointcloud_xyz_hand = pointcloud[:, :, :, :3]
        #     pointcloud_xyz = torch.einsum('bfnj,bfij->bfni', pointcloud_xyz_hand, hand_mat[:, :, :3, :3])
        pointcloud_other = pointcloud[:, :, :, 3:]

        
        actions = data['actions']
        # Extract rotation matrices
        # hand_rot_mat = hand_mat[:, :, :3, :3]
        # hand_rot_mat_inv = hand_mat_inv[:, :, :3, :3]
        # B = actions.shape[0]
        # frame_stack = hand_mat.shape[1]

        theta = torch.rand(B, device=actions.device) * 2 * np.pi
        axes = torch.tensor([(0, 0., 1.0)] * B, device=actions.device)
        rot_mats = batch_axis_angle_to_rotation_matrix(axes, theta)

        # transform into hand coordinate frame, rotate, and then back into world
        # action_transform = hand_rot_mat[:, -1] @ rot_mats @ hand_rot_mat_inv[:, -1]
        if actions.shape[-1] == 4:
            action_pos = actions[:, :, :3]
            actions_rot = None
            action_gripper = actions[:, :, 3:]
        
            actions_transformed = torch.einsum('btj,bij->bti', action_pos, rot_mats)
            actions_transformed = torch.concatenate((actions_transformed, action_gripper), dim=2)
        elif actions.shape[-1] == 7:
            _, H, _ = actions.shape
            # In this we assume that rotations are in the axis angle representation used here
            # https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L515
            # EDIT: ignore this line: so we can transform them by simply performing the same transformation as everything else
            action_pos = actions[:, :, :3]
            actions_pos_transformed = torch.einsum('btj,bij->bti', action_pos, rot_mats)

            # action_rot = actions[:, :, 3:6]
            # action_gripper = actions[:, :, 6:]
            # actions_transformed = torch.concatenate((actions_pos_transformed, 
            #                                          action_rot,
            #                                          action_gripper), dim=2)

            action_rot = actions[:, :, 3:6] * self.action_scale[3:]
            action_rot_angle = torch.linalg.norm(action_rot, dim=-1, keepdims=True)
            action_rot_axis = action_rot / (action_rot_angle + 1e-8)
            action_rot_mats = batch_axis_angle_to_rotation_matrix(
                action_rot_axis.view(B*H, 3), 
                action_rot_angle.view(B*H)).view(B, H, 3, 3)
            
            # action_transform_inv = torch.linalg.inv(action_transform)
            rot_mats_inv = torch.linalg.inv(rot_mats)

            # PTP*APT*P*
            # action_rot_mats_transformed = torch.einsum('bij,btjk,bkl->btil', action_transform, action_rot_mats, action_transform_inv)
            action_rot_mats_transformed = torch.einsum('bij,btjk,bkl->btil', rot_mats, action_rot_mats, rot_mats_inv)
            # action_rot_mats_transformed = torch.einsum('btjk,bkl->btjl', action_rot_mats, rot_mats_inv)

            action_axis_trans, action_angle_trans = batch_rotation_matrix_to_axis_angle(action_rot_mats_transformed.reshape(B * H, 3, 3))
            action_rot_transformed = (action_axis_trans * action_angle_trans.unsqueeze(-1)).view(B, H, 3) / self.action_scale[3:]

            # actions_pos_transformed = torch.einsum('btj,bij->bti', action_pos, action_transform)
            # action_rot_transformed = torch.einsum('btj,bij->bti', action_rot, action_transform)

            action_gripper = actions[:, :, 6:]

            # actions_transformed = torch.concatenate((actions_pos_transformed, 
            #                                          action_rot / self.action_scale[3:],
            #                                          action_gripper), dim=2)
            actions_transformed = torch.concatenate((actions_pos_transformed, 
                                                     action_rot_transformed,
                                                     action_gripper), dim=2)
        

        pointcloud_xyz_transformed = torch.einsum('bfnj,bfij->bfni', pointcloud_xyz, rot_mats.view(B, 1, 3, 3))

        pointcloud_xyz_transformed = pointcloud_xyz_transformed + ee_pos
            

        pointcloud_transformed = torch.concatenate((pointcloud_xyz_transformed.reshape(B, T, N, 3), pointcloud_other), dim=-1)
        # breakpoint()

        data['actions'] = actions_transformed
        data['transform'] = rot_mats
        obs_data[self.pointcloud_keys[0]] = pointcloud_transformed

        return data
