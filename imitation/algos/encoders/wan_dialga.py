"""Composable RGB + frozen Wan-VAE + frozen DIALGA encoder for LIBERO BC.

One class drives all five experimental variants by toggling flags:

    Run 1  obs + proprio + z_dyn + z_static     emit_zdyn=True,  emit_zstatic=True,  language_fusion=False
    Run 2  obs + proprio + lang  (baseline)     emit_zdyn=False, emit_zstatic=False, language_fusion=True
    Run 3  obs + proprio + z_dyn                emit_zdyn=True,  emit_zstatic=False, language_fusion=False
    Run 4  obs + proprio + wan_flat             emit_wanflat=True,                   language_fusion=False
    Run 5  obs + proprio + z_static             emit_zstatic=True,                   language_fusion=False

Alignment contract — the load-bearing detail
============================================
LIBERO BC dataset returns RGB obs of shape (B, T, 3, H, W). With frame_stack=33,
the last frame (T-1) is contemporaneous with the first action of the predicted
chunk a_t. That matches our DIALGA encoder's training contract: it consumes 33
consecutive pixel frames (Wan-VAE stride 4 → 9 latent frames per chunk) and
produces z_static + 9 z_dyn slots whose temporal index aligns with the action
chunk's first 9 frames.

So:
  frame_stack must equal CHUNK_T (default 33).  Hard asserted at init.
  The RGB ResNet branch consumes only the LAST `rgb_frames_used` frames
  (default 1) to keep token count manageable, while the DIALGA branch always
  consumes the full 33.

Frozen models loaded at init
============================
- Wan-VAE encoder (diffusers AutoencoderKLWan).  Encoder weights only.
- DIALGA LatentEncoder3D from libero_v512_big/v5_best.pt.
Both are frozen end-to-end.  No grads flow into them.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from imitation.algos.encoders.base import BaseEncoder
from imitation.algos.utils.misc import weight_init
import imitation.envs.utils as eu


# ---------- DIALGA encoder loader -------------------------------------------
# We need src.model.latent_encoder.LatentEncoder3D from the Dialga repo.
# Resolve the import lazily — Dialga is a sibling repo, not pip-installable.

DIALGA_REPO = Path("/storage/home/hcoda1/8/lwang831/workspace/Dialga")
if str(DIALGA_REPO) not in sys.path:
    sys.path.insert(0, str(DIALGA_REPO))


def _build_dialga_encoder(ckpt_path: str):
    """Mirrors scripts/eval_libero_action_probe.py::build_encoder_from_ckpt."""
    from src.model.latent_encoder import LatentEncoder3D

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    use_ln = "norm_static.weight" in ckpt["encoder"]
    enc = LatentEncoder3D(
        d_static=int(a.get("d_static", 64)),
        d_dyn=int(a.get("d_dyn", 64)),
        hidden_ch=int(a.get("enc_hidden_ch", 128)),
        shared_trunk=bool(a.get("shared_trunk", False)),
        use_layer_norm=use_ln,
    ).eval()
    enc.load_state_dict(ckpt["encoder"])
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def _load_wan_vae_encoder(model_id: str, dtype: torch.dtype):
    """Frozen Wan-2.2 TI2V-5B VAE.  Only the encoder half is exercised."""
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


# ---------- the encoder ------------------------------------------------------

CHUNK_T = 33                                # DIALGA pretraining chunk length
WAN_LAT_T = (CHUNK_T - 1) // 4 + 1           # = 9 latent frames


class WanDialgaEncoder(BaseEncoder):
    """RGB + proprio + optional {z_dyn, z_static, wan_flat} feature tokens.

    All produced tokens are projected to ``algo.image_embed_dim`` so the
    FlowMatchingPolicy's d_out_perception == d_out_lowdim invariant holds.
    """

    def __init__(
        self,
        image_encoder_factory,             # ResnetEncoder factory (for RGB branch)
        lowdim_encoder_factory,            # MLPProj factory (for proprio)
        share_image_encoder: bool = True,
        share_lowdim_encoder: bool = True,
        language_fusion: bool = False,
        load_depth: bool = False,
        # Wan / DIALGA flags --------------------------------------------------
        emit_zdyn: bool = False,
        emit_zstatic: bool = False,
        emit_wanflat: bool = False,
        # Frozen models -------------------------------------------------------
        wan_model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        dialga_ckpt_path: Optional[str] = None,
        wan_dtype: str = "float16",
        # Pre-encoded Wan-VAE latent cache --------------------------------
        # When set (path to a directory of per-task .h5 files), the dataset
        # injects obs[f"_wan_lat_{cam}"] for every sample and the encoder
        # skips its inline Wan-VAE forward. Live Wan-VAE is still loaded
        # so rollout (no cache available) falls back to the live encode.
        wan_cache_dir: Optional[str] = None,
        # Alignment / token-budget knobs -------------------------------------
        rgb_frames_used: int = 1,          # last K of the 33-frame stack go to RGB ResNet
        dialga_cameras: tuple = ("agentview_image",),  # cameras to feed into DIALGA
        chunk_T: int = CHUNK_T,
        feature_proj_dim: Optional[int] = None,  # default = image_embed_dim (from kwargs)
        # Hydra also passes task_embedding_format and image_embed_dim/lowdim_embed_dim implicitly
        **kwargs,
    ):
        super().__init__(**kwargs)

        # frame_stack must equal chunk_T so the dataset's 33-frame stack
        # exactly fills the DIALGA chunk.  This is the alignment guarantee.
        if any([emit_zdyn, emit_zstatic, emit_wanflat]):
            assert self.frame_stack == chunk_T, (
                f"frame_stack ({self.frame_stack}) must equal chunk_T ({chunk_T}) "
                f"when emitting DIALGA / wan_flat tokens. Set algo.frame_stack=33."
            )

        assert rgb_frames_used <= self.frame_stack, (
            f"rgb_frames_used ({rgb_frames_used}) cannot exceed "
            f"frame_stack ({self.frame_stack})."
        )

        self.language_fusion = language_fusion
        self.load_depth = load_depth
        self.emit_zdyn = emit_zdyn
        self.emit_zstatic = emit_zstatic
        self.emit_wanflat = emit_wanflat
        self.rgb_frames_used = rgb_frames_used
        self.dialga_cameras = tuple(dialga_cameras)
        self.chunk_T = chunk_T
        self.wan_cache_dir = wan_cache_dir

        # Resolve feature projection dim — match image_embed_dim by default.
        # Image_encoder_factory carries output_size; we read it after first build.
        self._feature_proj_dim_hint = feature_proj_dim

        # =========================== RGB branch =============================
        # This mirrors RGBEncoder but with rgb_frames_used overriding the
        # effective time dim during forward.
        do_lowdim = lowdim_encoder_factory is not None
        language_fusion_input = "film" if language_fusion else None

        obs_meta = self.shape_meta["observation"]

        if share_image_encoder:
            assert (len(set(tuple(s) for _, s in obs_meta["rgb"].items())) == 1), \
                "all rgb cameras must have the same shape"
            shape = list(list(obs_meta["rgb"].items())[0][1])
            if load_depth:
                shape[0] += 1
            self.image_encoders = image_encoder_factory(
                shape,
                language_fusion=language_fusion_input,
                language_dim=self.lang_embed_dim,
            )
            n_cams = len(eu.list_cameras(self.shape_meta))
            self.n_out_perception = self.rgb_frames_used * n_cams * self.image_encoders.n_out
            self.d_out_perception = self.image_encoders.out_channels
            self._rgb_shared = True
        else:
            self.image_encoders = {}
            for camera_name in eu.list_cameras(self.shape_meta):
                shape_in = list(self.shape_meta["observation"]["rgb"][eu.camera_name_to_image_key(camera_name)])
                if load_depth:
                    shape_in[0] += 1
                encoder = image_encoder_factory(
                    shape_in,
                    language_fusion=language_fusion_input,
                    language_dim=self.lang_embed_dim,
                )
                self.image_encoders[camera_name] = encoder
                self.n_out_perception += self.rgb_frames_used * encoder.n_out
                self.d_out_perception = encoder.out_channels
            self.image_encoders = nn.ModuleDict(self.image_encoders)
            self._rgb_shared = False

        feature_proj_dim = self._feature_proj_dim_hint or self.d_out_perception

        # =========================== Lowdim (proprio) =======================
        self.lowdim_encoders = {}
        if do_lowdim and len(obs_meta["lowdim"]) > 0:
            if share_lowdim_encoder:
                total_lowdim = sum(obs_meta["lowdim"].values())
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

        # =========================== Frozen Wan-VAE =========================
        # Always load when any wan token stream is requested, even with the
        # latent cache active. This keeps state_dict shape stable across
        # resume (old checkpoints saved Wan-VAE keys) and lets rollout fall
        # back to live encode. With the cache active, the forward simply
        # skips ``_encode_wan`` — the module sits idle in GPU memory.
        needs_wan = emit_zdyn or emit_zstatic or emit_wanflat
        if needs_wan:
            wan_dtype_t = {"float16": torch.float16, "float32": torch.float32,
                           "bfloat16": torch.bfloat16}[wan_dtype]
            self._wan = _load_wan_vae_encoder(wan_model_id, wan_dtype_t)
            self._wan_dtype = wan_dtype_t
        else:
            self._wan = None
            self._wan_dtype = torch.float32

        # =========================== Frozen DIALGA ==========================
        needs_dialga = emit_zdyn or emit_zstatic
        if needs_dialga:
            assert dialga_ckpt_path is not None, "dialga_ckpt_path required when emitting z_dyn/z_static"
            self._dialga = _build_dialga_encoder(dialga_ckpt_path)
            d_static = self._dialga.d_static
            d_dyn = self._dialga.d_dyn
        else:
            self._dialga = None
            d_static = d_dyn = 0

        # =========================== Projections ============================
        # Project each feature stream to feature_proj_dim so all perception
        # tokens land in the same dim space (FM policy stacks them).
        n_extra_tokens = 0
        n_cams_dialga = len(self.dialga_cameras)
        if emit_zstatic:
            self.proj_zstatic = nn.Linear(d_static, feature_proj_dim)
            n_extra_tokens += 1 * n_cams_dialga
        if emit_zdyn:
            self.proj_zdyn = nn.Linear(d_dyn, feature_proj_dim)
            n_extra_tokens += WAN_LAT_T * n_cams_dialga       # 9 slots per camera
        if emit_wanflat:
            wan_per_slot = 48 * 8 * 8                          # raw Wan latent per t
            self.proj_wanflat = nn.Linear(wan_per_slot, feature_proj_dim)
            n_extra_tokens += WAN_LAT_T * n_cams_dialga

        # Bump n_out_perception by the extra tokens so downstream contracts hold.
        self.n_out_perception += n_extra_tokens

    # =========================================================================
    # forward
    # =========================================================================
    def forward(self, data, obs_key):
        obs_data = data[obs_key]

        # ---- language emb (optional) -------------------------------------
        langs = self.get_task_emb(data) if self.language_fusion else None

        # ---- RGB branch — last K frames only -----------------------------
        # All RGB obs are (B, T, C, H, W).  We slice the last rgb_frames_used.
        for camera_name in eu.list_cameras(self.shape_meta):
            x = obs_data[eu.camera_name_to_image_key(camera_name)]
            x = torch.clip(x, 0, 1)
            obs_data[eu.camera_name_to_image_key(camera_name)] = x

        img_encodings = self._rgb_forward(obs_data, langs)

        # ---- Wan + DIALGA branch on full 33-frame stack ------------------
        if self.emit_zdyn or self.emit_zstatic or self.emit_wanflat:
            extra_tokens = self._wan_dialga_forward(obs_data)
            img_encodings.extend(extra_tokens)

        # ---- Lowdim (proprio) --------------------------------------------
        lowdim_encodings = self._lowdim_forward(obs_data)

        return img_encodings, lowdim_encodings

    # ---------------------------------------------------------------------
    def _rgb_forward(self, obs_data, langs):
        K = self.rgb_frames_used
        encodings = []

        if self._rgb_shared:
            # Stack cameras then time -> (B*N*K, C, H, W)
            imgs = []
            for camera_name in eu.list_cameras(self.shape_meta):
                img = obs_data[eu.camera_name_to_image_key(camera_name)]   # (B, T, C, H, W)
                if self.load_depth:
                    depth = torch.clamp(
                        obs_data[eu.camera_name_to_depth_key(camera_name)], 0.001, 5) - 2.5
                    img = torch.cat((img, depth), dim=2)
                imgs.append(img[:, -K:])                                   # last K frames
            x = torch.stack(imgs, dim=1)                                   # (B, N, K, C, H, W)
            B, N, T, C, H, W = x.shape
            x = einops.rearrange(x, "b n t c h w -> (b n t) c h w")
            langs_rep = einops.repeat(langs, "b d -> (b n t) d", n=N, t=T) if langs is not None else None
            x = self.image_encoders(x, langs=langs_rep)
            x = x.view(B, N, T, *x.shape[1:])
            encodings = list(einops.rearrange(x, "b ncam t m d -> (ncam t m) b d"))
        else:
            for camera_name in eu.list_cameras(self.shape_meta):
                img_name = eu.camera_name_to_image_key(camera_name)
                x = obs_data[img_name][:, -K:]                              # (B, K, C, H, W)
                if self.load_depth:
                    depth = torch.clamp(
                        obs_data[eu.camera_name_to_depth_key(camera_name)][:, -K:],
                        0.001, 5) - 2.5
                    x = torch.cat((x, depth), dim=2)
                B, T, C, H, W = x.shape
                e = self.image_encoders[img_name](
                    x.reshape(B * T, C, H, W), langs=langs)
                e = e.view(B, T, *e.shape[1:])
                e = list(einops.rearrange(e, "b t m d -> (t m) b d"))
                encodings.extend(e)
        return encodings

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _encode_wan(self, rgb_btchw: torch.Tensor) -> torch.Tensor:
        """rgb_btchw: (B, T=33, 3, H, W) in [0, 1] -> wan_lat (B, 48, 9, 8, 8) fp32.

        The Wan-VAE expects (B, 3, T, H, W) in [-1, 1].
        """
        B, T, C, H, W = rgb_btchw.shape
        assert T == self.chunk_T, f"need T={self.chunk_T}, got {T}"
        x = rgb_btchw.to(self._wan_dtype) * 2.0 - 1.0                    # [-1, 1]
        x = x.permute(0, 2, 1, 3, 4).contiguous()                         # (B, 3, T, H, W)
        out = self._wan.encode(x)
        z = out.latent_dist.mean if hasattr(out, "latent_dist") else out.latents
        return z.float()                                                   # (B, 48, 9, 8, 8)

    @torch.no_grad()
    def _encode_dialga(self, wan_lat: torch.Tensor) -> dict:
        """wan_lat: (B, 48, 9, 8, 8) -> {z_static: (B, d_s), z_dyn: (B, 9, d_d)}."""
        return self._dialga(wan_lat)

    def _wan_dialga_forward(self, obs_data) -> list:
        """Run Wan(+DIALGA) on every camera in self.dialga_cameras and emit
        a list of per-token tensors of shape (B, feature_proj_dim).

        Fast path: if the dataset injected ``obs[_wan_lat_<cam>]`` (the
        pre-encoded Wan-VAE latent for this sample's 33-frame stack), use
        it directly and skip the live Wan-VAE forward.
        """
        tokens = []
        for cam in self.dialga_cameras:
            cache_key = f"_wan_lat_{cam}"
            if cache_key in obs_data:
                # Cached latents are stored fp16 on disk; cast to fp32 to
                # match DIALGA's training dtype.
                wan_lat = obs_data[cache_key].float()                       # (B, 48, 9, 8, 8)
            else:
                assert self._wan is not None, (
                    f"Wan-VAE not loaded but {cache_key} missing in obs. "
                    "Either set wan_cache_dir=null or provide cached latents."
                )
                rgb = obs_data[cam]                                         # (B, T, C, H, W) in [0, 1]
                wan_lat = self._encode_wan(rgb)                             # (B, 48, 9, 8, 8)

            if self.emit_zdyn or self.emit_zstatic:
                out = self._encode_dialga(wan_lat)

                if self.emit_zstatic:
                    zs = self.proj_zstatic(out["z_static"])                 # (B, d_proj)
                    tokens.append(zs)
                if self.emit_zdyn:
                    zd = self.proj_zdyn(out["z_dyn"])                       # (B, 9, d_proj)
                    tokens.extend(list(zd.unbind(dim=1)))                   # 9 (B, d_proj)

            if self.emit_wanflat:
                # (B, 48, 9, 8, 8) -> (B, 9, 48*8*8=3072)
                B, C, Tl, Hl, Wl = wan_lat.shape
                wflat = wan_lat.permute(0, 2, 1, 3, 4).reshape(B, Tl, C * Hl * Wl)
                wf = self.proj_wanflat(wflat)                               # (B, 9, d_proj)
                tokens.extend(list(wf.unbind(dim=1)))
        return tokens

    # ---------------------------------------------------------------------
    def _lowdim_forward(self, obs_data) -> list:
        encodings = []
        if isinstance(self.lowdim_encoders, (dict, nn.ModuleDict)):
            for lowdim_name in self.lowdim_encoders.keys():
                x = self.lowdim_encoders[lowdim_name](obs_data[lowdim_name])
                x = list(einops.rearrange(x, "b t d -> t b d"))
                encodings.extend(x)
        elif isinstance(self.lowdim_encoders, nn.Module):
            lowdims = []
            for lowdim_name in self.shape_meta["observation"]["lowdim"].keys():
                lowdims.append(obs_data[lowdim_name])
            lowdim_input = torch.cat(lowdims, dim=-1)
            x = self.lowdim_encoders(lowdim_input)
            x = list(einops.rearrange(x, "b t d -> t b d"))
            encodings.extend(x)
        return encodings
