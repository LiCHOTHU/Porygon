import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imitation.utils.tensor_utils as TensorUtils
import itertools

from imitation.algos.base import ChunkPolicy


class QueST(ChunkPolicy):
    def __init__(self,
                 autoencoder,
                 policy_prior,
                 stage,
                 loss_fn,
                 l1_loss_scale,
                 embed_dim,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.autoencoder = autoencoder
        self.policy_prior = policy_prior
        self.stage = stage

        self.start_token = self.policy_prior.start_token
        self.l1_loss_scale = l1_loss_scale if stage == 2 else 0
        self.codebook_size = np.array(autoencoder.fsq_level).prod()
        
        self.loss = loss_fn

        obs_channels = self.encoder.n_out_perception * self.encoder.d_out_perception + \
                       self.encoder.n_out_lowdim * self.encoder.d_out_lowdim
        # TODO: this assumes frame_stack=1
        self.obs_proj = nn.Linear(obs_channels, embed_dim)
        
    def get_optimizers(self):
        if self.stage == 0:
            decay, no_decay = TensorUtils.separate_no_decay(self.autoencoder)
            optimizers = [
                self.optimizer_factory(params=decay),
                self.optimizer_factory(params=no_decay, weight_decay=0.)
            ]
            return optimizers
        elif self.stage == 1:
            decay, no_decay = TensorUtils.separate_no_decay(self, 
                                                            name_blacklist=('autoencoder',))
            optimizers = [
                self.optimizer_factory(params=decay),
                self.optimizer_factory(params=no_decay, weight_decay=0.)
            ]
            return optimizers
        elif self.stage == 2:
            decay, no_decay = TensorUtils.separate_no_decay(self, 
                                                            name_blacklist=('autoencoder',))
            decoder_decay, decoder_no_decay = TensorUtils.separate_no_decay(self.autoencoder.decoder)
            optimizers = [
                self.optimizer_factory(params=itertools.chain(decay, decoder_decay)),
                self.optimizer_factory(params=itertools.chain(no_decay, decoder_no_decay), weight_decay=0.)
            ]
            return optimizers

    def get_context(self, data):
        img_encodings, pc_encodings, lowdim_encodings = self.obs_encode(data)
        encodings = img_encodings + pc_encodings + lowdim_encodings
        encodings_cat = torch.cat(encodings, dim=-1)
        obs_emb = self.obs_proj(encodings_cat)
        task_emb = self.get_task_emb(data).unsqueeze(1)
        context = torch.cat([task_emb, obs_emb], dim=1)
        return context

    def compute_loss(self, data):
        if self.stage == 0:
            return self.compute_autoencoder_loss(data)
        elif self.stage == 1:
            return self.compute_prior_loss(data)
        elif self.stage == 2:
            return self.compute_prior_loss(data)

    def compute_autoencoder_loss(self, data):
        pred, pp, pp_sample, aux_loss, _ = self.autoencoder(data["actions"])
        recon_loss = self.loss(pred, data["actions"])
        if self.autoencoder.vq_type == 'vq':
            loss = recon_loss + aux_loss
        else:
            loss = recon_loss
            
        info = {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'aux_loss': aux_loss.sum().item(),
            'pp': pp.item(),
            'pp_sample': pp_sample.item(),
        }
        return loss, info

    def compute_prior_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        with torch.no_grad():
            indices = self.autoencoder.get_indices(data["actions"]).long()
        context = self.get_context(data)
        start_tokens = (torch.ones((context.shape[0], 1), device=self.device, dtype=torch.long) * self.start_token)
        x = torch.cat([start_tokens, indices[:,:-1]], dim=1)
        targets = indices.clone()
        logits = self.policy_prior(x, context)
        prior_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        with torch.no_grad():
            logits = logits[:,:,:self.codebook_size]
            probs = torch.softmax(logits, dim=-1)
            sampled_indices = torch.multinomial(probs.view(-1,logits.shape[-1]),1)
            sampled_indices = sampled_indices.view(-1,logits.shape[1])
        
        pred_actions = self.autoencoder.decode_actions(sampled_indices)
        l1_loss = self.loss(pred_actions, data["actions"])
        total_loss = prior_loss + self.l1_loss_scale * l1_loss
        info = {
            'loss': total_loss.item(),
            'nll_loss': prior_loss.item(),
            'l1_loss': l1_loss.item()
        }
        return total_loss, info

    def sample_actions(self, data):
        data = self.preprocess_input(data, train_mode=False)
        context = self.get_context(data)
        sampled_indices = self.policy_prior.get_indices_top_k(context, self.codebook_size)
        pred_actions = self.autoencoder.decode_actions(sampled_indices)
        pred_actions = pred_actions
        return pred_actions.detach().cpu().numpy()