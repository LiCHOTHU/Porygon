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
consecutive pixel frames (Wan-VAE stride 4 -> 9 latent frames per chunk) and
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
- DIALGA base+delta MemoryEncoder from libero_full/F_cand_s0/ckpt.pt, loaded via
  the Dialga repo's shared builder scripts.local.eval_psnr.build.
Both are frozen end-to-end.  No grads flow into them.

NOTE (2026 base+delta port): the DIALGA model changed from the old v512 dict API
(enc(lat) -> {"z_static", "z_dyn"}) to the base+delta tuple API
(enc(lat[B,1,48,9,8,8]) -> (grids[B,1,9,8,8], zdyn[B,1,9,d_dyn], _)). The loader and
`_encode_dialga` below were adapted to the tuple API; z_static = flattened static grid
(d_static = 9*8*8 = 576), z_dyn = per-slot dyn code (d_dyn = 64). Everything downstream
(projections, token emission) is unchanged.
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
# The base+delta DIALGA model lives in the sibling Dialga repo (not pip-installable).
# scripts.local.eval_psnr.build is the same loader we use everywhere else.

DIALGA_REPO = Path("/storage/home/hcoda1/8/lwang831/workspace/Dialga")
if str(DIALGA_REPO) not in sys.path:
    sys.path.insert(0, str(DIALGA_REPO))


def _build_dialga_encoder(ckpt_path: str, device: str = "cpu"):
    """Load the NEW base+delta DIALGA encoder via the Dialga repo's shared builder.

    ``scripts.local.eval_psnr.build(ckpt, device) -> (enc, dec, args)``.
    ``enc(wan_lat[B,1,48,9,8,8]) -> (grids[B,1,9,8,8], zdyn[B,1,9,d_dyn], _)``.

    Returns ``(enc, d_static, d_dyn)`` where d_static is the flattened static-grid
    width (9*8*8 = 576) and d_dyn is the per-slot dynamic width (64). Frozen.
    """
    from scripts.local.eval_psnr import build

    enc, _dec, a = build(ckpt_path, torch.device(device))
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    d_static = int(a["d_static"])   # 576 = flattened static grid (9*8*8)
    d_dyn = int(a["d_dyn"])         # 64 per z_dyn slot
    return enc, d_static, d_dyn


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
        # Ablation switch: drop the RGB ResNet branch entirely (proprio + feature
        # tokens only). Isolates what the DIALGA/wan feature carries, unmasked by
        # the pretrained image encoder. When False, feature tokens are projected
        # to `image_embed_dim` (there is no RGB out_channels to inherit).
        use_rgb: bool = True,
        image_embed_dim: int = 256,
        # Wan / DIALGA flags --------------------------------------------------
        emit_zdyn: bool = False,
        emit_zstatic: bool = False,
        emit_wanflat: bool = False,
        # Rate-matched baseline: PCA of the flattened Wan latent down to 576
        # floats (same budget as z_dyn/z_static, no DIALGA structure).
        # pca_path -> torch file with {'mean': (27648,), 'components': (576, 27648)}.
        emit_pca: bool = False,
        pca_path: Optional[str] = None,
        # Capacity/architecture control: DIALGA encoder with RANDOM weights
        # (same arch, ckpt ignored after arch construction).
        dialga_random_init: bool = False,
        # External frozen embedding as the ONLY perception input (Table-3 arms).
        # Train: the dataset's wan-cache channel injects the pre-extracted feature
        # vector (ext cache npy is (n_demos, max_T, ext_dim)). Rollout: live-encode
        # the 33-frame window with the named extractor.
        emit_ext: bool = False,
        ext_model: Optional[str] = None,   # dinov2 | videomae | vidtwin_dyn
        ext_dim: Optional[int] = None,
        # Frozen models -------------------------------------------------------
        wan_model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        dialga_ckpt_path: Optional[str] = None,
        wan_dtype: str = "float16",
        # Pre-encoded Wan-VAE latent cache --------------------------------
        # When set (path to a directory of per-task .npy/.h5 files), the dataset
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

        # Alignment guarantee. Two regimes:
        #   * LIVE encode (wan_cache_dir is None): the encoder itself runs the
        #     Wan-VAE on the raw 33-frame stack, so frame_stack MUST equal
        #     chunk_T to fill the DIALGA chunk.
        #   * CACHED (wan_cache_dir set): the DIALGA latent for the 33-frame
        #     window ending at t is injected from the cache (keyed by timestep,
        #     independent of frame_stack), and the RGB branch only ever uses the
        #     last `rgb_frames_used` frames. So frame_stack need not be 33 —
        #     using frame_stack=1 loads 1 frame/sample instead of 33 (≈33x less
        #     image I/O) with IDENTICAL model inputs. Require only that the stack
        #     covers the frames the RGB branch consumes.
        if any([emit_zdyn, emit_zstatic, emit_wanflat, emit_pca]):
            if wan_cache_dir is None:
                assert self.frame_stack == chunk_T, (
                    f"frame_stack ({self.frame_stack}) must equal chunk_T ({chunk_T}) "
                    f"for LIVE Wan encode. Set algo.frame_stack=33 or provide wan_cache_dir."
                )
            else:
                assert self.frame_stack >= rgb_frames_used, (
                    f"frame_stack ({self.frame_stack}) must be >= rgb_frames_used "
                    f"({rgb_frames_used}) when using the cached latents."
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
        self.emit_pca = emit_pca
        self.emit_ext = emit_ext
        self.ext_model = ext_model
        self.__dict__['_ext'] = None    # lazy live extractor (rollout only)
        self.rgb_frames_used = rgb_frames_used
        self.dialga_cameras = tuple(dialga_cameras)
        self.chunk_T = chunk_T
        self.wan_cache_dir = wan_cache_dir
        self.use_rgb = use_rgb

        # Resolve feature projection dim — match image_embed_dim by default.
        # Image_encoder_factory carries output_size; we read it after first build.
        self._feature_proj_dim_hint = feature_proj_dim

        # =========================== RGB branch =============================
        # This mirrors RGBEncoder but with rgb_frames_used overriding the
        # effective time dim during forward.
        do_lowdim = lowdim_encoder_factory is not None
        language_fusion_input = "film" if language_fusion else None

        obs_meta = self.shape_meta["observation"]

        if not use_rgb:
            # No image encoder: perception comes only from the feature tokens
            # (added below). Feature width falls back to image_embed_dim.
            self.image_encoders = None
            self.n_out_perception = 0
            self.d_out_perception = feature_proj_dim or image_embed_dim
            self._rgb_shared = False
        elif share_image_encoder:
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

        # Frozen Wan-VAE + DIALGA are stored OUTSIDE the nn.Module registry
        # (via __dict__) so they are never trained, never enter the policy
        # optimizer, and — crucially — never enter the checkpoint state_dict.
        # That keeps the state_dict identical whether we trained on the cache
        # (no live Wan-VAE) or roll out live (Wan-VAE loaded), so strict
        # load_state_dict works in both directions. They are device-migrated
        # lazily on first use (see _frozen_to).
        self._frozen_device = torch.device("cpu")
        self.__dict__["_dialga"] = None
        self.__dict__["_wan"] = None
        self._wan_dtype = {"float16": torch.float16, "float32": torch.float32,
                           "bfloat16": torch.bfloat16}[wan_dtype]

        # =========================== Frozen Wan-VAE =========================
        # Load the live Wan-VAE only when NO cache is provided (training uses the
        # cache and skips it; rollout has no cache and needs the live encode).
        # Skipping the 10 GB load when cached is the whole point of the pivot.
        needs_wan = emit_zdyn or emit_zstatic or emit_wanflat or emit_pca
        if needs_wan and (wan_cache_dir is None):
            self.__dict__["_wan"] = _load_wan_vae_encoder(wan_model_id, self._wan_dtype)

        # =========================== Frozen DIALGA ==========================
        needs_dialga = emit_zdyn or emit_zstatic
        if needs_dialga:
            assert dialga_ckpt_path is not None, "dialga_ckpt_path required when emitting z_dyn/z_static"
            enc, d_static, d_dyn = _build_dialga_encoder(dialga_ckpt_path)
            if dialga_random_init:
                # same architecture, random weights (capacity control)
                torch.manual_seed(0)
                enc.apply(lambda m: m.reset_parameters()
                          if hasattr(m, "reset_parameters") else None)
                for pm in enc.parameters():
                    pm.requires_grad_(False)
            self.__dict__["_dialga"] = enc
        else:
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
        if emit_pca:
            assert pca_path is not None, "emit_pca requires pca_path"
            pca = torch.load(pca_path, map_location="cpu", weights_only=False)
            self.register_buffer("pca_mean", pca["mean"].float())          # (27648,)
            self.register_buffer("pca_comp", pca["components"].float())    # (576, 27648)
            self.proj_pca = nn.Linear(self.pca_comp.shape[0], feature_proj_dim)
            n_extra_tokens += 1 * n_cams_dialga
        if emit_ext:
            assert ext_model in ("dinov2", "videomae", "vidtwin_dyn") and ext_dim,                 "emit_ext requires ext_model and ext_dim"
            self.proj_ext = nn.Linear(int(ext_dim), feature_proj_dim)
            n_extra_tokens += 1

        # Bump n_out_perception by the extra tokens so downstream contracts hold.
        self.n_out_perception += n_extra_tokens

        # FlowMatchingPolicy asserts d_out_perception == d_out_lowdim (one shared
        # token width). When the proprio branch is dropped (no lowdim tokens),
        # declare the lowdim width equal to perception so the invariant holds —
        # the value is never used because no lowdim tokens are emitted.
        if self.n_out_lowdim == 0:
            self.d_out_lowdim = self.d_out_perception

    # =========================================================================
    # forward
    # =========================================================================
    def forward(self, data, obs_key):
        obs_data = data[obs_key]

        # ---- language emb (optional) -------------------------------------
        langs = self.get_task_emb(data) if self.language_fusion else None

        # ---- RGB branch — last K frames only (skipped when use_rgb=False) --
        # All RGB obs are (B, T, C, H, W).  We slice the last rgb_frames_used.
        if self.use_rgb:
            for camera_name in eu.list_cameras(self.shape_meta):
                x = obs_data[eu.camera_name_to_image_key(camera_name)]
                x = torch.clip(x, 0, 1)
                obs_data[eu.camera_name_to_image_key(camera_name)] = x
            img_encodings = self._rgb_forward(obs_data, langs)
        else:
            img_encodings = []

        # ---- Wan + DIALGA branch on full 33-frame stack ------------------
        if self.emit_zdyn or self.emit_zstatic or self.emit_wanflat or self.emit_pca:
            extra_tokens = self._wan_dialga_forward(obs_data)
            img_encodings.extend(extra_tokens)

        # ---- External frozen embedding (Table-3 arms) --------------------
        if self.emit_ext:
            img_encodings.append(self._ext_forward(obs_data))

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
    def _frozen_to(self, device):
        """Migrate the __dict__-stored frozen models to `device` once."""
        if torch.device(device) == self._frozen_device:
            return
        if self._wan is not None:
            self.__dict__["_wan"] = self._wan.to(device)
        if self._dialga is not None:
            self.__dict__["_dialga"] = self._dialga.to(device)
        self._frozen_device = torch.device(device)

    @torch.no_grad()
    def _encode_wan(self, rgb_btchw: torch.Tensor) -> torch.Tensor:
        """rgb_btchw: (B, T=33, 3, H, W) in [0, 1] -> wan_lat (B, 48, 9, 8, 8) fp32.

        The Wan-VAE expects (B, 3, T, H, W) in [-1, 1].
        """
        self._frozen_to(rgb_btchw.device)
        B, T, C, H, W = rgb_btchw.shape
        assert T == self.chunk_T, f"need T={self.chunk_T}, got {T}"
        x = rgb_btchw.to(self._wan_dtype) * 2.0 - 1.0                    # [-1, 1]
        x = x.permute(0, 2, 1, 3, 4).contiguous()                         # (B, 3, T, H, W)
        out = self._wan.encode(x)
        z = out.latent_dist.mean if hasattr(out, "latent_dist") else out.latents
        return z.float()                                                   # (B, 48, 9, 8, 8)

    @torch.no_grad()
    def _encode_dialga(self, wan_lat: torch.Tensor) -> dict:
        """wan_lat: (B, 48, 9, 8, 8) -> {z_static: (B, 576), z_dyn: (B, 9, 64)}.

        Base+delta tuple API: enc(lat[B,1,48,9,8,8]) -> (grids[B,1,9,8,8],
        zdyn[B,1,9,64], _). z_static is the flattened static grid; z_dyn is the
        per-slot dynamic code for the 9 latent frames.
        """
        self._frozen_to(wan_lat.device)
        grids, zdyn, _ = self._dialga(wan_lat.unsqueeze(1))
        return {"z_static": grids[:, 0].flatten(1), "z_dyn": zdyn[:, 0]}

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

            if self.emit_pca:
                # rate-matched control: 27648 -> 576 via frozen PCA, one token
                flat = wan_lat.flatten(1)                                   # (B, 27648)
                feat = (flat - self.pca_mean) @ self.pca_comp.T             # (B, 576)
                tokens.append(self.proj_pca(feat))
        return tokens

    # ---------------------------------------------------------------------
    def _get_ext(self):
        """Lazy live extractor for rollout (training uses the injected cache)."""
        if self._ext is not None:
            return self._ext
        import numpy as _np
        if self.ext_model in ("dinov2", "videomae"):
            from scripts.probes.clevrer_baselines_probe import build_extractor
            self.__dict__["_ext"] = ("dialga_ext", build_extractor(self.ext_model, next(self.parameters()).device))
        else:  # vidtwin_dyn
            import sys as _sys, os as _os
            VREPO = "/storage/project/r-agarg35-0/lwang831/tools/VidTok"
            if VREPO not in _sys.path:
                _sys.path.insert(0, VREPO)
            from omegaconf import OmegaConf as _OC
            from vidtok.modules.util import instantiate_from_config as _ifc
            c = _OC.load(_os.path.join(VREPO, "configs/vidtwin/vidtwin_structure_7_7_8_dynamics_7_8.yaml"))
            c.model.params.encoder_config.params.enable_flashattn = False
            c.model.params.decoder_config.params.enable_flashattn = False
            c.model.params.loss_config = _OC.create({"target": "torch.nn.Identity"})
            c.model.params.ckpt_path = None
            m = _ifc(c.model)
            sd = torch.load("/storage/project/r-agarg35-0/lwang831/hf_cache/hub/models--microsoft--vidtwin/"
                            "snapshots/d367b26a3451abc6da1d374b7cd96ac82568145b/checkpoints/"
                            "vidtwin_structure_7_7_8_dynamics_7_8.ckpt", map_location="cpu")["state_dict"]
            m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("loss")}, strict=False)
            dev = next(self.parameters()).device
            self.__dict__["_ext"] = ("vidtwin", m.to(dev).eval())
        return self._ext

    @torch.no_grad()
    def _ext_feat_live(self, win_f01):
        """win_f01 (B, T, 3, H, W) in [0,1] -> (B, ext_dim) via the frozen extractor."""
        import numpy as _np
        kind, e = self._get_ext()
        if kind == "dialga_ext":
            u8 = (win_f01.clamp(0, 1) * 255).byte().permute(0, 1, 3, 4, 2).cpu().numpy()
            out = [_np.asarray(e.feat(w, [0], w.shape[0]), _np.float32) for w in u8]
            return torch.from_numpy(_np.stack(out)).to(win_f01.device)
        dev = win_f01.device
        B, T, C, H, W = win_f01.shape
        idx = torch.linspace(0, T - 1, 16).round().long()
        x = win_f01[:, idx].to(dev) * 2 - 1                                   # (B,16,3,H,W)
        x = F.interpolate(x.reshape(B * 16, C, H, W), size=(224, 224),
                          mode="bilinear", align_corners=False)
        x = x.reshape(B, 16, C, 224, 224).permute(0, 2, 1, 3, 4).contiguous()
        _, zc, zmx, zmy, _ = e.encode(x, return_reg_log=True)
        return torch.cat([zmx.flatten(1), zmy.flatten(1)], 1).float()

    def _ext_forward(self, obs_data):
        cache_key = "_wan_lat_agentview_image"      # the injection channel (reused)
        if cache_key in obs_data:
            feat = obs_data[cache_key].float()      # (B, ext_dim) from the ext cache
            if feat.dim() > 2:
                feat = feat.flatten(1)
        else:
            win = torch.clip(obs_data["agentview_image"], 0, 1)
            feat = self._ext_feat_live(win).float()
        return self.proj_ext(feat)

    # ---------------------------------------------------------------------
    def _lowdim_forward(self, obs_data) -> list:
        # Emit exactly ONE proprio token: the LAST frame's. Training runs with
        # frame_stack=1 (cache regime) so the policy learns a 1-token layout;
        # rollout overrides frame_stack=33 for the live Wan window, and emitting
        # 33 proprio tokens there is a train/eval token-layout shift that
        # collapsed the no-RGB (+proprio) arms to 0.000 success. Last-frame
        # proprio makes eval identical to training.
        encodings = []
        if isinstance(self.lowdim_encoders, (dict, nn.ModuleDict)):
            for lowdim_name in self.lowdim_encoders.keys():
                v = obs_data[lowdim_name]
                if v.dim() == 3:
                    v = v[:, -1:]                       # (B, 1, d)
                x = self.lowdim_encoders[lowdim_name](v)
                x = list(einops.rearrange(x, "b t d -> t b d"))
                encodings.extend(x)
        elif isinstance(self.lowdim_encoders, nn.Module):
            lowdims = []
            for lowdim_name in self.shape_meta["observation"]["lowdim"].keys():
                v = obs_data[lowdim_name]
                if v.dim() == 3:
                    v = v[:, -1:]                       # (B, 1, d)
                lowdims.append(v)
            lowdim_input = torch.cat(lowdims, dim=-1)
            x = self.lowdim_encoders(lowdim_input)
            x = list(einops.rearrange(x, "b t d -> t b d"))
            encodings.extend(x)
        return encodings
