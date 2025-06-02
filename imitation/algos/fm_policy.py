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
        **kwargs,
    ):
        super().__init__(**kwargs)

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

    def compute_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        cond = self.get_cond(data)
        actions = data["abs_actions"] if self.abs_action else data["actions"]

        B = cond.shape[0]
        device = cond.device

        t = self._sample_fm_time(B).to(device=actions.device)
        x0 = torch.randn_like(actions)
        x1 = torch.clamp(actions, -1, 1)
        psi_t = self._psi_t(x0, x1, t)

        _, v_psi = self.velocity_net(psi_t, t, cond)

        d_psi = x1 - (1 - self.flow_sig_min) * x0
        loss = torch.mean((v_psi - d_psi) ** 2)

        info = {
            "loss": loss.item(),
        }
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

