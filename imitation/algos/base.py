from abc import ABC, abstractmethod
from collections import deque

import einops
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ConstantLR

import imitation.utils.obs_utils as ObsUtils

# from imitation.modules.v1 import *
import imitation.utils.tensor_utils as TensorUtils
from imitation.algos.encoders.base import BaseEncoder
from imitation.algos.utils.normalizer import Normalizer
from imitation.algos.utils.rotation_transformer import RotationTransformer
from imitation.utils.utils import map_tensor_to_device


class Policy(nn.Module, ABC):
    """
    Super class with some basic functionality and functions we expect
    from all policy classes in our training loop
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        aug_factory,
        optimizer_factory,
        shape_meta,
        abs_action,
        device,
        eecf=False,
        normalizer: Normalizer = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.use_augmentation = aug_factory is not None
        self.shape_meta = shape_meta
        # self.build_pointcloud = isinstance(encoder, Adapt3REncoder)

        # aug_shape_meta = shape_meta.copy()

        # if self.build_pointcloud:
        #     for key, input_shape in shape_meta["observation"]["depth"].items():
        #         camera_name = self._depth_key_to_camera_name(key)
        #         pcd_key = f"{camera_name}_pointcloud_full"
        #         if pcd_key not in shape_meta["observation"]["pointcloud"]:
        #             input_shape[0] = 3
        #             aug_shape_meta["observation"]["pointcloud"][pcd_key] = input_shape

        #     aug_shape_meta["observation"]["depth"] = {}

        self.optimizer_factory = optimizer_factory
        if normalizer is None:
            normalizer = Normalizer(mode="identity")
        self.normalizer = normalizer
        self.abs_action = abs_action
        self.eecf = eecf
        # self.action_key = 'actions'
        self.device = device

        # Use 6D actions if we are predicting abs actions, else axis angle
        self.network_action_dim = 10 if abs_action else 7
        rot_rep = "rotation_6d" if abs_action else "axis_angle"
        self.rotation_transformer = RotationTransformer(from_rep="axis_angle", to_rep=rot_rep)

        if self.use_augmentation:
            self.aug = aug_factory(shape_meta=shape_meta)

        self.device = device

    @abstractmethod
    def compute_loss(self, data):
        raise NotImplementedError("Implement in subclass")

    def get_optimizers(self):
        decay, no_decay = TensorUtils.separate_no_decay(self)
        optimizers = [
            self.optimizer_factory(params=decay),
            self.optimizer_factory(params=no_decay, weight_decay=0.0),
        ]
        return optimizers

    def get_schedulers(self, optimizers, total_steps, schedule_type, warmup_steps, lr, end_factor=0.01):
        return [
            create_scheduler(optimizer, total_steps, schedule_type, warmup_steps, lr, end_factor) 
            for optimizer in optimizers
        ]

    def preprocess_input(self, data, train_mode=True):
        obs_data = data['obs']

        # TODO: move division by 1000 and normalization to proper encoders
        # for key in self.shape_meta["observation"]["depth"]:

        # if self.build_pointcloud:
        #     for depth_key in self.shape_meta["observation"]["depth"]:
        #         camera_name = self._depth_key_to_camera_name(depth_key)
        #         rgb_key = f"{camera_name}_rgb"
        #         intrinsic_key = f"{camera_name}_intrinsic"
        #         extrinsic_key = f"{camera_name}_extrinsic"
                
        #         if all(k in obs_data for k in [depth_key, intrinsic_key, extrinsic_key]):
        #             depths = obs_data[depth_key].squeeze(2)
        #             intrinsics = obs_data[intrinsic_key]
        #             extrinsics = obs_data[extrinsic_key]
                    
        #             B, T, H, W = depths.shape
        #             depths = depths.reshape(B, 1, H, W)
        #             intrinsics = intrinsics.reshape(B, 1, 3, 3)
        #             extrinsics = extrinsics.reshape(B, 1, 4, 4)
                    
        #             from imitation.utils.point_cloud_utils import lift_point_cloud_batch
        #             pcd = lift_point_cloud_batch(
        #                 depths,        # [B, 1, H, W]
        #                 intrinsics,    # [B, 1, 3, 3]
        #                 extrinsics,    # [B, 1, 4, 4]
        #                 keepdims=True
        #             )
        #             pcd_key = self._image_key_to_pointcloud_key(rgb_key)
        #             obs_data[pcd_key] = pcd

        # for depth_key in self.shape_meta["observation"]["depth"]:
            # obs_data[key] = obs_data[key].to(torch.float32) / 1000.0
            # obs_data[depth_key] = torch.clamp(obs_data[depth_key], 0.1, 5.0)
            # obs_data[depth_key] = (obs_data[depth_key] - 2.55) / 2.0 

        if train_mode and self.use_augmentation:
            data = self.aug(data)

        # TODO: move to encoder
        # for key in self.shape_meta["observation"]["rgb"]:
        #     for obs_key in ("obs", "next_obs"):
        #         if obs_key in data:
        #             x = TensorUtils.to_float(data[obs_key][key])
        #             x = x / 255.0
        #             x = torch.clip(x, 0, 1)
        #             data[obs_key][key] = x
        
        action_norm_keys = ("abs_actions" if self.abs_action else "actions",)
        # norm_keys = tuple(self.shape_meta["observation"]["lowdim"])
        norm_keys = action_norm_keys + tuple(self.shape_meta["observation"]["lowdim"])
        data = self.normalizer.normalize(data, keys=norm_keys)

        return data
    
    def extract_actions(self, data):
        actions = data["abs_actions"] if self.abs_action else data["actions"]
        if self.eecf:
            assert 'hand_mat_inv' in data['obs'], "EECF requires hand_mat_inv in obs"
            # breakpoint()
            rot_mat_inv = data["obs"]["hand_mat_inv"][..., :3, :3]
            actions_pos, actions_rest = torch.split(actions, [3, actions.shape[-1] - 3], dim=-1)
            actions_pos = torch.einsum("...ij,...j->...i", rot_mat_inv, actions_pos)
            actions = torch.cat((actions_pos, actions_rest), dim=-1)
        return actions

    def obs_encode(self, data, obs_key="obs"):
        return self.encoder(data, obs_key)

    def reset(self):
        return

    def get_task_emb(self, data):
        return self.encoder.get_task_emb(data)

    def get_action(self, obs, task_id, task_emb=None):
        self.eval()
        for key, value in obs.items():
            if key in self.shape_meta["rgb"]:
                value = ObsUtils.process_frame(value, channel_dim=3)
            obs[key] = torch.tensor(value)
        batch = {}
        batch["obs"] = obs
        if task_emb is not None:
            batch["task_emb"] = task_emb
        else:
            # TODO: repeat for parallel envs, can be done inside env runner
            batch["task_id"] = torch.tensor([task_id], dtype=torch.long)
        batch = map_tensor_to_device(batch, self.device)
        with torch.no_grad():
            action = self.sample_actions(batch)
        action = self.normalizer.unnormalize({self.action_key: action})[self.action_key]
        return action

    def postprocess_action(self, action):
        if self.abs_action:
            pos, rot_raw, gripper = torch.split(action, [3, action.shape[-1] - 4, 1], dim=-1)
            rot = self.rotation_transformer.inverse(rot_raw)
            action = torch.cat([pos, rot, gripper], dim=-1)
        return action

    def preprocess_dataset(self, dataset, use_tqdm=True):
        return

    @abstractmethod
    def sample_actions(self, obs):
        raise NotImplementedError("Implement in subclass")

    # def compute_norm_stats(self, cfg):
    #     if cfg.pace_copy:
    #         pace_tmp_dir = os.getenv('TMPDIR')
    #         copy_data_pace(cfg, pace_tmp_dir)
    #         dataset = instantiate(cfg.task.dataset,
    #                             data_prefix=os.path.join(pace_tmp_dir, 'data'))
    #         dataset_stats = instantiate(cfg.task.dataset, 
    #                                     data_prefix=os.path.join(pace_tmp_dir, 'data'),
    #                                     stats_mode=True)
    #     else:
    #         dataset = instantiate(cfg.task.dataset)
    #         dataset_stats = instantiate(cfg.task.dataset, stats_mode=True)

    # def normalize(self, data):
    #     if self.normalizer is None:
    #         return data
    #     else:
    #         return self.normalizer.normalize(data)

    # def unnormalize(self, data):
    #     if self.normalizer is None:
    #         return data
    #     else:
    #         return self.normalizer.unnormalize(data)


class ChunkPolicy(Policy):
    """
    Super class for policies which predict chunks of actions
    """

    def __init__(self, action_horizon, chunk_size, temporal_agg=False, **kwargs):
        super().__init__(**kwargs)

        self.action_horizon = action_horizon
        self.action_horizon = 4
        self.chunk_size = chunk_size
        self.temporal_agg = temporal_agg
        self.action_queue = None
        self.action_history = None
        self.batch_size = None
        self.actions_in_queue = 0

    def reset(self):
        if self.temporal_agg:
            if self.batch_size is not None:
                self.action_history = np.zeros(
                    (
                        self.batch_size,
                        self.chunk_size,
                        self.chunk_size,
                        self.network_action_dim,
                    )
                )
                self.actions_in_queue = 0
        else:
            self.action_queue = deque(maxlen=self.action_horizon)

    def get_action(self, obs, task_id, task_emb):
        if self.temporal_agg:
            actions = self._get_action_agg(obs, task_id, task_emb)
        else:
            actions = self._get_action_no_agg(obs, task_id, task_emb)
        
        actions = self.postprocess_action(actions)
        return actions.to(torch.float32).cpu().numpy()

    def _get_action_agg(self, obs, task_id, task_emb):  # obs, task_id, task_emb=None):
        self.eval()
        if self.batch_size is None:
            self.batch_size = obs[list(obs.keys())[0]].shape[0]
            self.reset()

        batch = self._make_batch(obs, task_id, task_emb)
        with torch.no_grad():
            actions = self.sample_actions(batch)
            action_key = "abs_actions" if self.abs_action else "actions"
            actions = self.normalizer.unnormalize({action_key: actions})[action_key]

        # Chop off the actions corresponding to the last timestep
        # and the oldest action in the history
        self.action_history = self.action_history[:, :-1, 1:]
        self.actions_in_queue = min(self.chunk_size, self.actions_in_queue + 1)
        self.action_history = np.concatenate(
            (
                self.action_history,
                np.zeros((self.batch_size, self.chunk_size - 1, 1, self.network_action_dim)),
            ),
            axis=2,
        )
        actions = einops.rearrange(actions, "b sbs d_act -> b 1 sbs d_act")
        self.action_history = np.concatenate((actions, self.action_history), axis=1)

        action_sums = np.sum(self.action_history, axis=1)
        action_denoms = self.chunk_size - np.arange(self.chunk_size)
        action_denoms = np.minimum(action_denoms, self.actions_in_queue)
        action_denoms = einops.repeat(action_denoms, "sbs -> B sbs 1", B=self.batch_size)
        out_actions = action_sums / action_denoms
        out_actions = torch.tensor(out_actions)
        return out_actions[:, 0]

    def _get_action_no_agg(self, obs, task_id, task_emb=None):
        assert (
            self.action_queue is not None
        ), "you need to call policy.reset() before getting actions"

        # self.eval()
        # TODO: can shift preprocessing to the env wrapper
        if len(self.action_queue) == 0:
            batch = self._make_batch(obs, task_id, task_emb)
            with torch.no_grad():
                actions = self.sample_actions(batch)
                action_key = "abs_actions" if self.abs_action else "actions"
                actions = self.normalizer.unnormalize({action_key: actions})[action_key]
                # actions = actions.cpu().numpy()
                actions = np.transpose(actions, (1, 0, 2))
                self.action_queue.extend(actions[: self.action_horizon])
        action = self.action_queue.popleft()
        return torch.tensor(action)

    def _make_batch(self, obs, task_id, task_emb):
        for key, value in obs.items():
            if key in self.shape_meta["observation"]["rgb"]:
                value = ObsUtils.process_frame(value, channel_dim=3)
            elif key in self.shape_meta["observation"]["lowdim"]:
                value = TensorUtils.to_float(value)  # from double to float
            elif "depth" in key:
                value = ObsUtils.process_frame(value, channel_dim=1)
            obs[key] = torch.tensor(value)
        batch = {}
        batch["obs"] = obs
        if task_emb is not None:
            batch["task_emb"] = task_emb
        # else:
        # TODO: repeat for parallel envs, can be done inside env runner
        batch["task_id"] = torch.tensor([task_id], dtype=torch.long)
        batch = map_tensor_to_device(batch, self.device)
        return batch

    @abstractmethod
    def sample_actions(self, obs):
        raise NotImplementedError("Implement in subclass")


def create_scheduler(optimizer, total_steps, schedule_type, warmup_steps, lr, end_factor=0.01):
    if schedule_type is None:
        return []
    eta_min = end_factor * lr
    # If no warmup is requested, just return the main scheduler
    if warmup_steps <= 0:
        if schedule_type == 'cosine':
            return CosineAnnealingLR(optimizer, eta_min=eta_min, T_max=total_steps)
        elif schedule_type == 'linear':
            return LinearLR(optimizer, start_factor=1.0, end_factor=end_factor, total_iters=total_steps)
        elif schedule_type == 'constant':
            return ConstantLR(optimizer, factor=1.0)
        else:
            raise ValueError(f"Unknown scheduler type: {schedule_type}")
    
    # Create warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.001,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Create main scheduler based on the specified type
    if schedule_type == 'cosine':
        main_scheduler = CosineAnnealingLR(optimizer, eta_min=eta_min, T_max=total_steps - warmup_steps)
    elif schedule_type == 'linear':
        main_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=end_factor,
            total_iters=total_steps - warmup_steps
        )
    elif schedule_type == 'constant':
        main_scheduler = ConstantLR(optimizer, factor=1.0)
    else:
        raise ValueError(f"Unknown scheduler type: {schedule_type}")
    
    # Combine schedulers
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps]
    )