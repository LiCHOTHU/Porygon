from typing import Dict, List, Optional, Tuple, Union

import einops
import dgl.geometry as dgl_geo
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork

from imitation.algos.utils.misc import weight_init
from imitation.algos.encoders.rgb_modules import ResnetEncoder
from imitation.algos.encoders.point_cloud_base import PointCloudBaseEncoder
from imitation.algos.encoders.adapt3r import Adapt3REncoder
import imitation.envs.utils as eu
import imitation.utils.point_cloud_utils as pcu

from .clip import load_clip
from imitation.algos.utils.position_encodings import NeRFSinusoidalPosEmb
from .resnet import load_resnet18, load_resnet50


class BimanualAdapt3REncoder(Adapt3REncoder):
    """Adapt3R Encoder for processing point clouds with RGB images and language embeddings.
    
    This encoder combines point cloud data with RGB images and optional language embeddings
    to create a rich representation of the scene. It supports multiple camera views
    """
    


    def forward(self, data, obs_key):
        obs_data = data[obs_key]
        
        rgb = []
        pcd = []

        pcds = self._build_point_cloud(obs_data)
        for camera_name in eu.list_cameras(self.shape_meta):
            rgb.append(obs_data[eu.camera_name_to_image_key(camera_name)])
            pcd.append(pcds[camera_name])

        assert len(rgb) == len(pcd)

        rgb = torch.stack(rgb).to(dtype=torch.float32) / 255
        pcd = torch.stack(pcd).to(dtype=torch.float32)

        device = rgb.device

        n_cam, B, fs, _, _, _ = rgb.shape

        # Create RGB tinted versions of each camera view
        # rgb_tinted = []
        # for i in range(n_cam):
        #     # Create a copy of the RGB data for this camera
        #     rgb_cam = rgb[i].clone()
            
        #     # Apply color tint based on camera index
        #     if i == 0:  # First camera - red tint
        #         rgb_cam[:, :, 1:] *= 0.5  # Reduce green and blue channels
        #     elif i == 1:  # Second camera - green tint
        #         rgb_cam[:, :, 0] *= 0.5  # Reduce red channel
        #         rgb_cam[:, :, 2] *= 0.5  # Reduce blue channel
        #     else:  # Third camera - blue tint
        #         rgb_cam[:, :, :2] *= 0.5  # Reduce red and green channels
                
        #     rgb_tinted.append(rgb_cam)
            
        # # Stack the tinted views back together
        # rgb = torch.stack(rgb_tinted)
        # pcd_vis = einops.rearrange(pcd, "ncam b fs h w c -> (b fs) (ncam h w) c")
        # rgb_vis = einops.rearrange(rgb, "ncam b fs c h w -> (b fs) (ncam h w) c")
        # pos_right, rot_right, extra_right, pos_left, rot_left, extra_left = torch.split(
        #     data["actions"], 
        #     [3, 6, 6, 3, 6, 6], dim=-1)
        # extra_points = torch.cat([pos_right, pos_left], dim=1)
        # for i in range(pcd_vis.shape[0]):
        #     print('left\n', pos_left[i])
        #     print('right\n', pos_right[i])
        #     pcu.show_point_cloud(pcd_vis[i], rgb_vis[i], extra_points=extra_points[i])

        # pcd_vis = einops.rearrange(pcd, "ncam b fs h w c -> (b fs) ncam (h w) c")
        # rgb_vis = einops.rearrange(rgb, "ncam b fs c h w -> (b fs) ncam (h w) c")
        # for i in range(pcd_vis.shape[0]):
        #     for j in range(pcd_vis.shape[1]):
        #         pcu.show_point_cloud(pcd_vis[i, j], rgb_vis[i, j])

        rgb = einops.rearrange(rgb, "ncam b fs c h w -> (b fs ncam) c h w")
        pcd = einops.rearrange(pcd, "ncam b fs h w c -> (b fs ncam) c h w")

            
        # Pass each view independently through backbone
        if self.do_image:
            rgb_normalized = self.normalize(rgb)
            if self.finetune:
                if self.backbone_type == "fusion":
                    task_emb = einops.repeat(
                        data["task_emb"], "b d -> (b fs ncam) d", fs=fs, ncam=n_cam
                    )
                    rgb_features = self.backbone(rgb_normalized, langs=task_emb)
                else:
                    rgb_features = self.backbone(rgb_normalized)
            else:
                with torch.no_grad():
                    if self.backbone_type == "fusion":
                        task_emb = einops.repeat(
                            data["task_emb"], "b d -> (b fs ncam) d", fs=fs, ncam=n_cam
                        )
                        rgb_features = self.backbone(rgb_normalized, langs=task_emb)
                    else:
                        rgb_features = self.backbone(rgb_normalized)

            # Pass visual features through feature pyramid network
            rgb_features = self.feature_pyramid(rgb_features)
        else:
            rgb_features = {"out": torch.zeros((B * n_cam, 60, 32, 32), device="cuda"),}

        rgb_features = rgb_features['out']

        # Interpolate xy-depth to get the locations for this level
        feat_h, feat_w = rgb_features.shape[-2:]
        pcd = F.interpolate(pcd, (feat_h, feat_w), mode="nearest")
        rgb = F.interpolate(rgb, (feat_h, feat_w), mode="bilinear")

        # Merge different cameras for clouds, separate for rgb features
        pcd = einops.rearrange(
            pcd, "(bt fs ncam) c h w -> bt fs (ncam h w) c", ncam=n_cam, fs=fs
        )
        rgb_features = einops.rearrange(
            rgb_features, "(bt fs ncam) c h w -> bt fs (ncam h w) c", ncam=n_cam, fs=fs
        )
        rgb = einops.rearrange(
            rgb, "(bt fs ncam) c h w -> bt fs (ncam h w) c", ncam=n_cam, fs=fs
        )

        # Apply cropping
        mask = self._crop_point_cloud(pcd=pcd, task_id=data["task_id"], hand_mat_inv=data["obs"]["hand_mat_inv"])


        pcd = pcd * mask.unsqueeze(-1)
        rgb = rgb * mask.unsqueeze(-1)
        rgb_features = rgb_features * mask.unsqueeze(-1)


        if self.hand_frame:
            pcd = pcu.batch_transform_point_cloud(pcd, data["obs"]["hand_mat_inv"])

        for i in range(pcd.shape[0]):
            pcu.show_point_cloud(pcd[i, 0], rgb[i, 0])
        
        pcd, rgb_features, rgb, mask = self._downsample_point_cloud(pcd=pcd, rgb_features=rgb_features, rgb=rgb, mask=mask)

        pcd_pos_emb = self.xyz_proj(pcd)

        cat_cloud = []
        if self.do_pos:
            cat_cloud.append(pcd_pos_emb)
        if self.do_image:
            cat_cloud.append(rgb_features)
        if self.do_lang:
            lang_emb = self.get_task_emb(data)
            lang_emb = self.lang_proj(lang_emb)
            lang_emb = einops.repeat(lang_emb, "b d -> b fs n d", fs=fs, n=self.num_points)
            cat_cloud.append(lang_emb)
        if self.do_rgb:
            cat_cloud.append(rgb)
        cat_cloud = torch.cat(cat_cloud, dim=-1)

        out = self.pointcloud_extractor(
            cat_cloud,
            mask=mask,
        )
        out = list(einops.rearrange(out, "b fs d -> fs b d"))

        lowdim_out = self._encode_lowdim(obs_data)

        return out, lowdim_out



