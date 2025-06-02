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


class DiffusionTransformerPolicy(ChunkPolicy):
    """
    Policy from https://dit-policy.github.io/
    """
    def __init__(
        self,
        train_diffusion_steps,
        eval_diffusion_steps,
        dropout=0,
        feat_norm=None,
        embed_dim=None,
        noise_net_kwargs=dict(),
        **kwargs,
    ):
        super().__init__(**kwargs)

        # initialize obs and img tokenizers

        self.noise_net = DiTNoiseNet(
            ac_dim=self.network_action_dim,
            ac_chunk=self.chunk_size,
            **noise_net_kwargs,
        )
        # self._network_action_dim, self._chunk_size = self.network_action_dim, self.ch

        assert (
            eval_diffusion_steps <= train_diffusion_steps
        ), "Can't eval with more steps!"
        self._train_diffusion_steps = train_diffusion_steps
        self._eval_diffusion_steps = eval_diffusion_steps
        self.diffusion_schedule = DDIMScheduler(
            num_train_timesteps=train_diffusion_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )

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
        timesteps = torch.randint(
            low=0, high=self._train_diffusion_steps, size=(B,), device=device
        ).long()

        noise = torch.randn_like(actions)

        # construct noise actions given real actions, noise, and diffusion schedule
        noise_acs = self.diffusion_schedule.add_noise(actions, noise, timesteps)
        _, noise_pred = self.noise_net(noise_acs, timesteps, cond)

        # calculate loss for noise net
        loss = nn.functional.mse_loss(noise_pred, noise)
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
            enc_cache = self.noise_net.forward_enc(cond)

            # begin diffusion process
            self.diffusion_schedule.set_timesteps(self._eval_diffusion_steps)
            self.diffusion_schedule.alphas_cumprod = (
                self.diffusion_schedule.alphas_cumprod.to(device)
            )
            for timestep in self.diffusion_schedule.timesteps:
                # predict noise given timestep
                batched_timestep = timestep.unsqueeze(0).repeat(B).to(device)
                noise_pred = self.noise_net.forward_dec(noise_actions, batched_timestep, enc_cache)

                # take diffusion step
                noise_actions = self.diffusion_schedule.step(
                    model_output=noise_pred, timestep=timestep, sample=noise_actions
                ).prev_sample

            # return final action post diffusion
            return noise_actions.cpu().numpy()

