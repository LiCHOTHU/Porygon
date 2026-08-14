from imitation.algos.base import ChunkPolicy
from imitation.algos.utils.diffusion_policy_utils.dit_modules import DiTNoiseNet
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

class _BatchNorm1DHelper(nn.BatchNorm1d):
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.transpose(1, 2)
            x = super().forward(x)
            return x.transpose(1, 2)
        return super().forward(x)


from imitation.algos.sib import (ActionConditioner, BandEMA, SpectralBands, sib_penalty,
                                 use_double_backward_safe_attention)
from imitation.algos.regularizers import maybe_mixup, sam_ascent, sam_descend


class FlowMatchingPolicy(ChunkPolicy):
    """
    Policy from https://dit-policy.github.io/ adapted for flow matching.
    """
    def __init__(
        self,
        num_inference_steps: int = 10,
        dropout=0,
        feat_norm=None,
        embed_dim=None,
        velocity_net_kwargs=dict(),
        flow_sampling: str = 'beta',
        flow_alpha: float = 1.5,
        flow_beta: float = 1.,
        flow_sig_min: float = 0.001,
        sib_beta: float = 0.0,
        sib_n_bands: int = 8,
        sib_code_dim: int = 8,
        sib_stop_grad: bool = False,
        sib_n_probes: int = 1,
        mixup_alpha: float = 0.0,
        sam_rho: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sib_beta = sib_beta
        self.sib_stop_grad = sib_stop_grad
        self.sib_n_probes = sib_n_probes
        self.mixup_alpha = mixup_alpha
        self.sam_rho = sam_rho
        self._sib_ready = False
        self._sib_cfg = dict(n_bands=sib_n_bands, code_dim=sib_code_dim)
        if self.sib_beta > 0:
            use_double_backward_safe_attention()
            # Turn on capture immediately.  The bands cannot be built until the
            # feature-map size is known, and that is only known after a real
            # forward -- so capture has to precede them.
            self._set_encoder_flag("sib_capture", True)

        # initialize obs and img tokenizers

        self.velocity_net = DiTNoiseNet(
            ac_dim=self.network_action_dim,
            ac_chunk=self.chunk_size,
            **velocity_net_kwargs,
        )
        # self._network_action_dim, self._chunk_size = self.network_action_dim, self.ch

        self.num_inference_steps = num_inference_steps
        self.flow_sampling = flow_sampling
        self.flow_alpha = flow_alpha
        self.flow_beta = flow_beta
        self.flow_sig_min = flow_sig_min
        assert self.flow_sampling in [
            "uniform",
            "beta",
        ], f"Invalid flow matching timestep sampling mode: {self.flow_sampling}"
        if self.flow_sampling == "beta":
            self.flow_t_max = 1 - flow_sig_min
            self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)
        
        # build (optional) token feature projection layer
        linear_proj = nn.Identity()
        assert self.encoder.d_out_perception == self.encoder.d_out_lowdim
        encoder_out_dim = self.encoder.d_out_perception
        if embed_dim is not None and embed_dim != encoder_out_dim:
            linear_proj = nn.Linear(encoder_out_dim, embed_dim)
            encoder_out_dim = embed_dim

        # build feature normalization layers
        if feat_norm == "batch_norm":
            norm = _BatchNorm1DHelper(encoder_out_dim)
        elif feat_norm == "layer_norm":
            norm = nn.LayerNorm(encoder_out_dim)
        else:
            assert feat_norm is None
            norm = nn.Identity()

        # final token post proc network
        self.post_proc = nn.Sequential(linear_proj, norm, nn.Dropout(dropout))

    def _attach_sib(self, z_example, action_numel, device):
        """Build the band/conditioner modules once the feature-map size is known."""
        _, _, h, w = z_example.shape
        self.sib_bands = SpectralBands(h, w, self._sib_cfg["n_bands"]).to(device)
        self.sib_conditioner = ActionConditioner(
            action_numel, code_dim=self._sib_cfg["code_dim"]
        ).to(device)
        self.sib_ema = BandEMA(self._sib_cfg["n_bands"]).to(device)
        self._sib_ready = True

    def _image_encoders(self):
        encoders = getattr(self.encoder, "image_encoders", None)
        if encoders is None:
            return []
        return list(encoders.values()) if hasattr(encoders, "values") else [encoders]

    def _set_encoder_flag(self, name, value):
        for enc in self._image_encoders():
            setattr(enc, name, value)

    def _enable_sib_hook(self):
        """Point every image encoder at the shared band module."""
        self._set_encoder_flag("sib_bands", getattr(self, "sib_bands", None))

    def _collect_sib_z(self):
        """Concatenate the exposed feature maps across cameras along the batch axis.

        Both wrist and third-person views get the same treatment; stacking them
        means one penalty over all views rather than a per-camera weighting we
        would then have to justify.
        """
        zs = [e.sib_z for e in self._image_encoders() if getattr(e, "sib_z", None) is not None]
        return zs or None

    def compute_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        actions_raw = data["abs_actions"] if self.abs_action else data["actions"]

        # Mixup blends observations and action chunks with a shared coefficient.
        # For a policy this is a stronger assumption than in classification -- it
        # asserts the action is locally linear in the observation -- so it is
        # reported as the weakest of the baseline arms.
        data, actions_raw, lam = maybe_mixup(data, actions_raw, self.mixup_alpha)

        def _flow_terms():
            cond = self.get_cond(data)
            actions = actions_raw
            B = cond.shape[0]
            t = self._sample_fm_time(B).to(device=actions.device)
            x0 = torch.randn_like(actions)
            x1 = torch.clamp(actions, -1, 1)
            psi_t = self._psi_t(x0, x1, t)
            _, v_psi = self.velocity_net(psi_t, t, cond)
            d_psi = x1 - (1 - self.flow_sig_min) * x0
            return torch.mean((v_psi - d_psi) ** 2), v_psi, x1, t

        if self.sib_beta > 0 and not self._sib_ready:
            # one throwaway forward to learn the feature-map size
            with torch.no_grad():
                self.get_cond(data)
            probe = self._collect_sib_z()
            if probe is None:
                # A projection-style encoder has no spatial map, so the spectral
                # penalty cannot be defined.  Fail loudly: silently running an
                # unregularized policy under a name that says otherwise is how a
                # null result gets mistaken for a real one.
                raise RuntimeError(
                    "sib_beta > 0 but no spatial feature map was captured. The SIB "
                    "penalty needs an encoder that keeps (B,C,H,W) -- use "
                    "'override encoder: rgb_no_pool' (do_projection: false)."
                )
            self._attach_sib(probe[0], actions_raw[0].numel(), actions_raw.device)
            self._enable_sib_hook()

        loss, v_psi, x1, t = _flow_terms()
        info = {"loss": loss.item()}

        if self.sib_beta > 0:
            zs = self._collect_sib_z()
            if zs is not None:
                penalty, sib_info = sib_penalty(
                    zs, v_psi, x1, t,
                    self.sib_bands, self.sib_conditioner, self.sib_ema,
                    n_probes=self.sib_n_probes, stop_grad=self.sib_stop_grad,
                )
                loss = loss + self.sib_beta * penalty
                info.update(sib_info)
                info["loss"] = loss.item()

        # SAM needs the caller to do two backward passes; the trainer handles it
        # through `sam_step_fn` when sam_rho > 0.
        if self.sam_rho > 0:
            self._sam_closure = lambda: _flow_terms()[0]
        return loss, info
    
    def get_cond(self, data):
        perception_encodings, lowdim_encodings = self.obs_encode(data)
        encodings = perception_encodings + lowdim_encodings
        encodings = torch.stack(encodings, dim=1)
        return encodings

    def sample_actions(self, data):
        with torch.no_grad():
            data = self.preprocess_input(data, train_mode=False)
            cond = self.get_cond(data)

            # get observation encoding and sample noise
            B, device = cond.shape[0], cond.device
            noise_actions = torch.randn(B, self.chunk_size, self.network_action_dim, device=device)
            enc_cache = self.velocity_net.forward_enc(cond)

            delta_t = 1.0 / self.num_inference_steps
            t = torch.zeros(B, device=device, dtype=cond.dtype)
            for _ in range(self.num_inference_steps):

                # predict velocity given timestep
                action_vel = self.velocity_net.forward_dec(noise_actions, t, enc_cache)
                noise_actions += delta_t * action_vel
                t += delta_t
                

            # # begin diffusion process
            # self.diffusion_schedule.set_timesteps(self._eval_diffusion_steps)
            # self.diffusion_schedule.alphas_cumprod = (
            #     self.diffusion_schedule.alphas_cumprod.to(device)
            # )
            # for timestep in self.diffusion_schedule.timesteps:
            #     # predict noise given timestep
            #     batched_timestep = timestep.unsqueeze(0).repeat(B).to(device)
            #     noise_pred = self.noise_net.forward_dec(noise_actions, batched_timestep, enc_cache)

            #     # take diffusion step
            #     noise_actions = self.diffusion_schedule.step(
            #         model_output=noise_pred, timestep=timestep, sample=noise_actions
            #     ).prev_sample

            action = torch.clamp(noise_actions, -1, 1)

            # return final action post diffusion
            return action.cpu().numpy()

    # Flow matching utils
    def _psi_t(
        self,
        x: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Conditional Flow"""
        t = t[:, None, None]  # (B, 1, 1)
        return (1 - (1 - self.flow_sig_min) * t) * x + t * x1

    def _sample_fm_time(self, bsz: int,) -> torch.FloatTensor:
        if self.flow_sampling == "uniform":  # uniform between 0 and 1
            """https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb"""
            eps = 1e-5
            t = (torch.rand(1) + torch.arange(bsz) / bsz) % (1 - eps)
        elif self.flow_sampling == "beta":  # from pi0 paper
            z = self.flow_beta_dist.sample((bsz,))
            t = self.flow_t_max * (1 - z)  # flip and shift
        return t

