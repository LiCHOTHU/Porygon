"""Tiled-view variant of RGBEncoder.

Identical to imitation.algos.encoders.rgb.RGBEncoder except the two LIBERO
cameras (agentview + wrist) are horizontally concatenated into a SINGLE
(3, H, 2*W) image inside ``forward``, before the image encoder runs.

Purpose: a single CNN sees the full multi-camera scene without aspect-ratio
distortion. shape_meta still declares the two cameras at their native
(3, H, W); the encoder factory is overridden in __init__ to build a CNN for
the tiled (3, H, 2*W) input shape.

Drop-in replacement for `rgb_no_pool` in any `algo.encoder` config.
"""
import einops
import torch
import torch.nn as nn

import imitation.envs.utils as eu
from imitation.algos.encoders.rgb import RGBEncoder
from imitation.algos.utils.misc import weight_init


_AGENT_KEY = "agentview_image"
_WRIST_KEY = "robot0_eye_in_hand_image"


class RGBTiledEncoder(RGBEncoder):
    """RGBEncoder variant that concatenates the agent + wrist cameras along
    the width axis before the image-encoder forward pass."""

    def __init__(
        self,
        image_encoder_factory,
        lowdim_encoder_factory,
        share_image_encoder=True,
        share_lowdim_encoder=True,
        language_fusion=False,
        load_depth=False,
        **kwargs,
    ):
        # Skip RGBEncoder.__init__ and call its grandparent (BaseEncoder).
        # We rebuild the image_encoders ourselves so the CNN gets the tiled
        # input shape, not the per-camera shape.
        from imitation.algos.encoders.base import BaseEncoder
        BaseEncoder.__init__(self, **kwargs)

        self.language_fusion = language_fusion
        language_fusion_input = "film" if language_fusion else None
        self.load_depth = load_depth

        do_lowdim = lowdim_encoder_factory is not None

        # --- image encoder built for the TILED shape (3, H, 2*W) ---
        obs_meta = self.shape_meta["observation"]
        cam_names = eu.list_cameras(self.shape_meta)
        if len(cam_names) != 2:
            raise ValueError(
                f"RGBTiledEncoder expects exactly 2 cameras in shape_meta; "
                f"got {cam_names}"
            )
        cam_shapes = [tuple(obs_meta["rgb"][eu.camera_name_to_image_key(c)]) for c in cam_names]
        if cam_shapes[0] != cam_shapes[1]:
            raise ValueError(
                f"RGBTiledEncoder expects both cameras to have the same shape; "
                f"got {cam_shapes}"
            )
        C, H, W = cam_shapes[0]
        tiled_shape = [C, H, 2 * W]
        if self.load_depth:
            tiled_shape[0] += 1
        # share_image_encoder is implicit (single CNN over the tile).
        self.image_encoders = image_encoder_factory(
            tiled_shape,
            language_fusion=language_fusion_input,
            language_dim=self.lang_embed_dim,
        )
        # Per RGBEncoder accounting: the tile counts as ONE camera.
        self.n_out_perception = self.frame_stack * 1 * self.image_encoders.n_out
        self.d_out_perception = self.image_encoders.out_channels

        # --- lowdim encoder (verbatim copy of RGBEncoder.__init__) ---
        self.lowdim_encoders = {}
        if do_lowdim and len(obs_meta["lowdim"]) > 0:
            if share_lowdim_encoder:
                total_lowdim = 0
                for name, shape in obs_meta["lowdim"].items():
                    total_lowdim += shape
                encoder = lowdim_encoder_factory(total_lowdim)
                encoder.apply(weight_init)
                self.lowdim_encoders = encoder
                self.n_out_lowdim += 1
                self.d_out_lowdim = encoder.out_channels
            else:
                for name, shape in obs_meta["lowdim"].items():
                    encoder = lowdim_encoder_factory(shape)
                    encoder.apply(weight_init)
                    self.lowdim_encoders[name] = encoder
                    self.n_out_lowdim += 1
                    self.d_out_lowdim = encoder.out_channels
                self.lowdim_encoders = nn.ModuleDict(self.lowdim_encoders)

    def forward(self, data, obs_key):
        obs_data = data[obs_key]
        img_encodings, lowdim_encodings = [], []
        langs = self.get_task_emb(data) if self.language_fusion else None

        # ---- clip each camera to [0, 1] (parity with RGBEncoder.forward) ----
        for camera_name in eu.list_cameras(self.shape_meta):
            x = obs_data[eu.camera_name_to_image_key(camera_name)]
            x = torch.clip(x, 0, 1)
            obs_data[eu.camera_name_to_image_key(camera_name)] = x

        # ---- concat agent + wrist along width axis ----
        agent = obs_data[_AGENT_KEY]   # (B, T, 3, H, W)
        wrist = obs_data[_WRIST_KEY]   # (B, T, 3, H, W)
        x = torch.cat([agent, wrist], dim=-1)  # (B, T, 3, H, 2W)
        if self.load_depth:
            agent_d = torch.clamp(obs_data[_AGENT_KEY.replace("_image", "_depth")], 0.001, 5) - 2.5
            wrist_d = torch.clamp(obs_data[_WRIST_KEY.replace("_image", "_depth")], 0.001, 5) - 2.5
            d = torch.cat([agent_d, wrist_d], dim=-1)
            x = torch.cat([x, d], dim=2)

        B, T, C, H, W2 = x.shape
        e = self.image_encoders(
            x.reshape(B * T, C, H, W2),
            langs=langs,
        )
        e = e.view(B, T, *e.shape[1:])
        e = list(einops.rearrange(e, "b t m d -> (t m) b d"))
        img_encodings.extend(e)

        # ---- lowdim (verbatim copy of RGBEncoder.forward path) ----
        if type(self.lowdim_encoders) in (dict, nn.ModuleDict):
            for lowdim_name in self.lowdim_encoders.keys():
                x = self.lowdim_encoders[lowdim_name](obs_data[lowdim_name])
                x = list(einops.rearrange(x, "b t d -> t b d"))
                lowdim_encodings.extend(x)
        else:
            lowdims = []
            for lowdim_name in self.shape_meta["observation"]["lowdim"].keys():
                lowdims.append(obs_data[lowdim_name])
            lowdim_input = torch.cat(lowdims, dim=-1)
            x = self.lowdim_encoders(lowdim_input)
            x = list(einops.rearrange(x, "b t d -> t b d"))
            lowdim_encodings.extend(x)

        return img_encodings, lowdim_encodings
