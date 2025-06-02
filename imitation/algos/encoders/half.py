"""
This file contains all neural modules related to encoding the spatial
information of obs_t, i.e., the abstracted knowledge of the current visual
input conditioned on the language.
"""

import einops

import torch

from imitation.algos.encoders.base import BaseEncoder

from .clip import load_clip



class HalfEncoder(BaseEncoder):
    def __init__(
        self,
        finetune: True,
        **kwargs,
    ):
        super().__init__(**kwargs)


        self.backbone, self.normalize = load_clip()

        self.finetune = finetune
        if not finetune:
            for p in self.backbone.parameters():
                p.requires_grad = False


    def forward(self, data, obs_key):
        obs_data = data[obs_key]

        rgb = []
        for rgb_key in self.shape_meta["observation"]["rgb"]:
            rgb.append(obs_data[rgb_key])

        rgb = torch.stack(rgb)

        B = rgb.shape[1]
        rgb = einops.rearrange(rgb, "ncam b fs c h w -> (b fs ncam) c h w")
        
        # TODO: remove this. This is for vis purposes

        # Pass each view independently through backbone
        rgb_normalized = self.normalize(rgb)
        if self.finetune:
            rgb_features = self.backbone(rgb_normalized)
        else:
            with torch.no_grad():
                rgb_features = self.backbone(rgb_normalized)
                
        feats = rgb_features['res2']
        feats = einops.rearrange(feats, '(b ncam) d h w -> b ncam h w d', b=B)
        return feats
        # if self.obs_reduction == 'cat':
        #     encoded = img_encodings + pc_encodings + lowdim_encodings
        #     encoded = torch.cat(encoded, -1)  # (B, T, H_all)
        #     if self.obs_proj is not None:
        #         obs_emb = self.obs_proj(encoded) # TODO I feel that this projection should be algorithm-specific
        # elif self.obs_reduction == 'stack':
        #     encoded = img_encodings + pc_encodings + lowdim_encodings
        #     encoded = torch.stack(encoded, dim=2)
        #     obs_emb = self.obs_proj(encoded)
        # elif self.obs_reduction == 'none':
        # return obs_emb
