from typing import Dict, List, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork

from imitation.algos.utils.misc import weight_init
from imitation.algos.encoders.point_cloud_base import PointCloudBaseEncoder
import imitation.envs.utils as eu
import imitation.utils.point_cloud_utils as pcu
from imitation.algos.utils.position_encodings import NeRFSinusoidalPosEmb

from .clip import load_clip

class OTTER3DEncoder(PointCloudBaseEncoder):
    """OTTER3D Encoder for processing point clouds with CLIP features and language embeddings.
    
    This encoder combines point cloud data with CLIP features and language embeddings
    to create a rich representation of the scene. It uses CLIP's text-aware visual features
    aligned with point cloud data.
    
    Args:
        pointcloud_extractor_factory: Factory function to create point cloud extractor
        hidden_dim (int): Dimension of hidden features
        do_pos (bool): Whether to include position encoding
        do_lang (bool): Whether to include language embeddings
        do_rgb (bool): Whether to include raw RGB values
        hand_frame (bool): Whether to transform points to hand frame
        do_rot_aug (bool): Whether to apply rotation augmentation
        xyz_proj_type (str): Type of position encoding ('nerf' or 'none')
        temperature (float): Temperature for text-aware visual feature extraction
    """
    
    def __init__(
        self,
        pointcloud_extractor_factory,
        hidden_dim: int,
        do_pos: bool = True,
        do_lang: bool = True,
        do_rgb: bool = False,
        hand_frame: bool = True,
        do_rot_aug: bool = False,
        xyz_proj_type: str = "nerf",
        temperature: float = 0.07,
        **kwargs,
    ) -> None:
        # Initialize parent class
        super().__init__(**kwargs)

        # Calculate point cloud input dimension
        pc_in = (
            hidden_dim  # CLIP features
            + do_pos * (3 if xyz_proj_type == "none" else hidden_dim)
            + (3 if do_rgb else 0)
        )
        
        # Initialize pointcloud extractor
        self._init_pointcloud_extractor(pointcloud_extractor_factory, pc_in)
        
        # Set additional flags
        self.hand_frame = hand_frame
        self.do_rot_aug = do_rot_aug
        self.do_pos = do_pos
        self.do_lang = do_lang
        self.do_rgb = do_rgb
        self.n_out_perception = 1
        self.d_out_perception = self.pointcloud_extractor.out_channels

        # Initialize CLIP model
        self._init_clip()

        # Initialize text-aware visual extraction
        self._init_text_aware_extraction(temperature)

        # Setup position encoding
        self._init_position_encoding(xyz_proj_type, hidden_dim)

    def _init_pointcloud_extractor(self, factory, in_shape: int) -> None:
        """Initialize the point cloud extractor."""
        self.pointcloud_extractor = factory(in_shape=in_shape)
        self.pointcloud_extractor.apply(weight_init)

    def _init_clip(self) -> None:
        """Initialize CLIP model and hooks for feature extraction."""
        self.clip_model, self.normalize = load_clip()
        
        # Freeze CLIP parameters
        for p in self.clip_model.parameters():
            p.requires_grad = False
            
        # Setup activation hooks
        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
            
        # Register hooks for CLIP features
        self.hooks = [
            self.clip_model.visual.transformer.resblocks[-1].attn.register_forward_hook(
                get_activation('image_patches')
            ),
            self.clip_model.transformer.register_forward_hook(
                get_activation('text_features')
            )
        ]

    def _init_text_aware_extraction(self, temperature: float) -> None:
        """Initialize text-aware visual feature extraction."""
        self.text_aware_extraction = TextAwareVisualExtraction(
            temperature=temperature
        )

    def _init_position_encoding(self, xyz_proj_type: str, hidden_dim: int) -> None:
        """Initialize position encoding."""
        if xyz_proj_type == "nerf":
            self.xyz_proj = NeRFSinusoidalPosEmb(hidden_dim)
        elif xyz_proj_type == "none":
            self.xyz_proj = nn.Identity()
        else:
            raise ValueError(f"Unsupported xyz_proj_type: {xyz_proj_type}")

    def _extract_clip_features(self, rgb: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract CLIP features from RGB images and text."""
        # Normalize and process images
        rgb_normalized = self.normalize(rgb)
        
        # Get CLIP features
        with torch.no_grad():
            _ = self.clip_model.encode_text(text)
            _ = self.clip_model.encode_image(rgb_normalized)
            
        # Process text features
        text_features = self.activation['text_features'].permute(1, 0, 2)
        text_features = self.clip_model.ln_final(text_features) @ self.clip_model.text_projection
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Process patch features
        patch_features = self.activation['image_patches'][0]
        patch_features = patch_features.permute(1, 0, 2)
        patch_features = patch_features[:, 1:]  # Remove CLS token
        patch_features = self.clip_model.visual.ln_post(patch_features)
        
        if self.clip_model.visual.proj is not None:
            patch_features = patch_features @ self.clip_model.visual.proj
            
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
        
        return patch_features, text_features

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

        # Rearrange for processing
        rgb = einops.rearrange(rgb, "ncam b fs c h w -> (b fs ncam) c h w")
        pcd = einops.rearrange(pcd, "ncam b fs h w c -> (b fs ncam) c h w")
        
        # Extract CLIP features
        patch_features, text_features = self._extract_clip_features(rgb, data["task_emb"])
        
        # Get text-aware visual features
        text_aware_features = self.text_aware_extraction(patch_features, text_features)
        
        # Reshape features to match point cloud dimensions
        feat_h, feat_w = text_aware_features.shape[-2:]
        pcd = F.interpolate(pcd, (feat_h, feat_w), mode="nearest")
        rgb = F.interpolate(rgb, (feat_h, feat_w), mode="bilinear")
        
        # Merge different cameras
        pcd = einops.rearrange(
            pcd, "(bt fs ncam) c h w -> bt fs (ncam h w) c", ncam=n_cam, fs=fs
        )
        text_aware_features = einops.rearrange(
            text_aware_features, "(bt fs ncam) c h w -> bt fs (ncam h w) c", ncam=n_cam, fs=fs
        )
        rgb = einops.rearrange(
            rgb, "(bt fs ncam) c h w -> bt fs (ncam h w) c", ncam=n_cam, fs=fs
        )

        # Apply cropping
        mask = self._crop_point_cloud(pcd=pcd, task_id=data["task_id"], hand_mat_inv=data["obs"]["hand_mat_inv"])

        pcd = pcd * mask.unsqueeze(-1)
        rgb = rgb * mask.unsqueeze(-1)
        text_aware_features = text_aware_features * mask.unsqueeze(-1)

        if self.hand_frame:
            pcd = pcu.batch_transform_point_cloud(pcd, data["obs"]["hand_mat_inv"])
        
        pcd, text_aware_features, rgb, mask = self._downsample_point_cloud(
            pcd=pcd, 
            rgb_features=text_aware_features, 
            rgb=rgb, 
            mask=mask
        )

        # Combine features
        cat_cloud = []
        if self.do_pos:
            pcd_pos_emb = self.xyz_proj(pcd)
            cat_cloud.append(pcd_pos_emb)
        cat_cloud.append(text_aware_features)
        if self.do_rgb:
            cat_cloud.append(rgb)
            
        cat_cloud = torch.cat(cat_cloud, dim=-1)

        # Process through point cloud extractor
        out = self.pointcloud_extractor(
            cat_cloud,
            mask=mask,
        )

        lowdim_out = self._encode_lowdim(obs_data)

        return [out], lowdim_out


class TextAwareVisualExtraction(nn.Module):
    """Extract text-aware visual features using CLIP, following ClearCLIP approach"""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
    def forward(self, image_patch_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        # Calculate similarity between text and patch features
        # image_patch_features: (batch_size, num_patches, embedding_dim)
        # text_features: (batch_size, num_tokens, embedding_dim)
        similarity = torch.einsum('bij,bkj->bik', text_features, image_patch_features)
        
        # Apply temperature scaling and softmax
        attention = F.softmax(similarity / self.temperature.clamp(0, 100), dim=-1)
        
        return attention