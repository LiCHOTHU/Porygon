"""
This file contains all neural modules related to encoding the spatial
information of obs_t, i.e., the abstracted knowledge of the current visual
input conditioned on the language.
"""

import einops

import dgl.geometry as dgl_geo
import torch
import torch.nn.functional as F

from imitation.algos.utils.adapt3r_encoder import Adapt3REncoder


# to allow 'list(x)' in pdb
def lisst(x):
    return list(x)


HAND_FRAME_CROP = ((-0.05, -1, -1), (1, 1, 1))



class BeefyHybridEncoder(Adapt3REncoder):
    def forward(self, data, obs_key):
        obs_data = data[obs_key]
        device = data['task_emb'].device
        rgb, pcd = self._gather_rgb_pcd(obs_data)
        n_cam, B, fs, _, H, W = rgb.shape

        rgb = rgb.permute(1, 2, 0, 3, 4, 5).reshape(B * fs * n_cam, 3, H, W)
        pcd = pcd.permute(1, 2, 0, 3, 4, 5).reshape(B * fs * n_cam, H, W, 3)

        rgb_features = self._encode_rgb(rgb, data, B, fs, n_cam)

        rgb_feats_pyramid, pcd_pyramid, rgb_pyramid = self._build_pyramids(rgb_features, pcd, rgb, n_cam, fs, H, W)
        rgb_feats, pcd, rgb_flat = [torch.cat(x, 1) for x in (rgb_feats_pyramid, pcd_pyramid, rgb_pyramid)]

        masked_pcd, masked_features, masked_rgb, mask = self._crop_pointcloud(data, pcd, rgb_feats, rgb_flat, B, fs, device)
        masked_pcd, masked_features, masked_rgb = self._apply_hand_crop(data, masked_pcd, masked_features, masked_rgb, B, fs, device, mask)

        downsampled_pcd, downsampled_feats, downsampled_rgb, downsample_mask = self._downsample(masked_pcd, masked_features, masked_rgb, device, B)
        
        cat_cloud = self._build_final_cloud(data, downsampled_pcd, downsampled_feats, downsampled_rgb, B, fs, downsample_mask)
        out = self.pointcloud_extractor(cat_cloud['xyz'], cat_cloud['features'], mask=cat_cloud['mask'],
                                        pc_pos=downsampled_pcd, pc_rgb=downsampled_rgb, vis_attn=False)

        lowdim_out = self._encode_lowdim(obs_data) if self.lowdim_encoder else []

        return [out], [], lowdim_out

    def _gather_rgb_pcd(self, obs_data):
        rgb, pcd = [], []
        for rgb_key in self.shape_meta["observation"]["rgb"]:
            rgb.append(obs_data[rgb_key])
            pcd.append(obs_data[image_key_to_pointcloud_key(rgb_key)])
        return torch.stack(rgb), torch.stack(pcd).float()

    def _encode_rgb(self, rgb, data, B, fs, n_cam):
        if not self.do_image:
            return {"res2": torch.zeros((B * n_cam, 60, 32, 32), device=rgb.device),
                    "res3": torch.zeros((B * n_cam, 60, 32, 32), device=rgb.device)}
        
        rgb_normalized = self.normalize(rgb)
        task_emb = None
        if self.backbone_type == "fusion":
            task_emb = einops.repeat(data["task_emb"], "b d -> (b fs ncam) d", fs=fs, ncam=n_cam)

        with torch.set_grad_enabled(self.finetune):
            if task_emb is not None:
                features = self.backbone(rgb_normalized, langs=task_emb)
            else:
                features = self.backbone(rgb_normalized)
            features = self.feature_pyramid(features)
        return features

    def _build_pyramids(self, rgb_features, pcd, rgb, n_cam, fs, H, W):
        rgb_feats_pyramid, pcd_pyramid, rgb_pyramid = [], [], []
        for level in self.feature_map_pyramid:
            f = F.interpolate(rgb_features[level], size=(H, W), mode='bilinear')
            f = f.view(-1, fs, n_cam, f.shape[1], H, W)
            pcd_flat = pcd.view(-1, fs, n_cam, H, W, 3)
            rgb_flat = rgb.view(-1, fs, n_cam, 3, H, W)

            rgb_feats_pyramid.append(f.flatten(2, 4).permute(0, 2, 1, 3))
            pcd_pyramid.append(pcd_flat.flatten(2, 4))
            rgb_pyramid.append(rgb_flat.flatten(2, 4))
        return rgb_feats_pyramid, pcd_pyramid, rgb_pyramid

    def _crop_pointcloud(self, data, pcd, rgb_feats, rgb_flat, B, fs, device):
        if not self.do_crop:
            mask = torch.ones(pcd.shape[:-1], device=device)
            return pcd, rgb_feats, rgb_flat, mask

        boundaries = self.boundaries[data["task_id"]]
        low, high = [einops.repeat(x, 'b d -> (b fs) 1 d', fs=fs) for x in (boundaries[:,0], boundaries[:,1])]
        mask = (pcd > low).all(-1) & (pcd < high).all(-1)

        masked_pcd = pcd * mask.unsqueeze(-1)
        masked_feats = rgb_feats * mask.unsqueeze(-1)
        masked_rgb = rgb_flat * mask.unsqueeze(-1)

        return masked_pcd, masked_feats, masked_rgb, mask

    def _apply_hand_crop(self, data, pcd, feats, rgb, B, fs, device, mask):
        if not self.do_hand_crop:
            return pcd, feats, rgb

        mat_inv = einops.rearrange(data['obs']['hand_mat_inv'], 'b fs i j -> (b fs) i j')
        pcd = torch.einsum('bnj,bij->bni', torch.cat([pcd, torch.ones_like(pcd[..., :1])], dim=-1), mat_inv)[...,:-1]
        low, high = [torch.tensor(x, device=device).expand(B * fs, 1, 3) for x in HAND_FRAME_CROP]
        mask_hand = (pcd > low).all(-1) & (pcd < high).all(-1)

        final_mask = mask & mask_hand
        return pcd * final_mask.unsqueeze(-1), feats * final_mask.unsqueeze(-1), rgb * final_mask.unsqueeze(-1)

    def _downsample(self, pcd, feats, rgb, device, B):
        if self.downsample_mode == "none":
            mask = torch.ones(B, pcd.shape[1], device=device, dtype=torch.bool)
            return pcd, feats, rgb, mask

        feat_base = feats[..., :30] if self.downsample_mode.startswith("feat") else pcd
        idx = dgl_geo.farthest_point_sampler(feat_base, self.num_points, 0).clamp(min=0)

        gather = lambda x, d: torch.gather(x, 1, einops.repeat(idx, 'b n -> b n d', d=d))
        pcd_ds = gather(pcd, 3)
        rgb_ds = gather(rgb, 3)
        feats_ds = gather(feats, feats.shape[-1])
        mask = ~(idx == -1)

        return pcd_ds, feats_ds, rgb_ds, mask

    def _build_final_cloud(self, data, pcd, feats, rgb, B, fs, mask):
        pcs_emb = self.xyz_proj(pcd)
        features = []
        if self.do_pos: features.append(pcs_emb)
        if self.do_image: features.append(feats)
        if self.do_lang:
            lang_emb = einops.repeat(self.lang_proj(self.get_task_emb(data)), 'b d -> (b fs) n d', fs=fs, n=pcd.shape[1])
            features.append(lang_emb)
        if self.do_rgb: features.append(rgb)
        features = torch.cat(features, dim=-1)

        return {
            'xyz': einops.rearrange(pcd, '(b fs) n c -> b fs n c', b=B),
            'features': einops.rearrange(features, '(b fs) n d -> b fs n d', b=B),
            'mask': einops.rearrange(mask, '(b fs) n -> b fs n', b=B)
        }

    def _encode_lowdim(self, obs_data):
        lowdim = torch.cat([obs_data[name] for name, _ in self.shape_meta['observation']['lowdim'].items()], dim=-1)
        return [self.lowdim_encoder(lowdim)]


def image_key_to_pointcloud_key(image_key):
    if "rgb" in image_key: #agentview_rgb
        return f"{image_key[:-4]}_pointcloud_full"
    elif "image" in image_key: #agentview_image
        return f"{image_key[:-6]}_pointcloud_full"
    else:
        raise ValueError(f"Unknown image key: {image_key}")
    
def image_key_to_camera_name(image_key):
    if "rgb" in image_key: #agentview_rgb
        return image_key[:-4]
    elif "image" in image_key: #agentview_image
        return image_key[:-6]
    else:
        raise ValueError(f"Unknown image key: {image_key}")