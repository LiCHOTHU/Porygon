import copy
import os
from collections import OrderedDict, deque

import cv2

# import gym.spaces
# import gym.wrappers
import gymnasium
import numpy as np
import torch
import torch.nn as nn
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from transformers import AutoModel, AutoTokenizer, logging

import imitation.utils.file_utils as FileUtils
import imitation.utils.obs_utils as ObsUtils
import imitation.utils.utils as utils
from imitation.dataset.sequence_dataset import SequenceDataset
from imitation.utils.draw_utils import show_point_cloud
from imitation.utils.frame_stack import FrameStackObservationFixed

# import gym
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
import multiprocessing
import time

import h5py
import matplotlib.pyplot as plt
import open3d as o3d
import pytorch3d.ops as torch3d_ops
import robosuite.utils.transform_utils as T
from gymnasium.vector.utils import batch_space, concatenate
from hydra.utils import to_absolute_path
from tqdm import tqdm, trange
from transformers import AutoModel, AutoTokenizer, logging

from imitation.utils.draw_utils import aggr_point_cloud_from_data
from imitation.utils.mujoco_point_cloud import posRotMat2Mat, quat2Mat

np.set_printoptions(suppress=True)
from copy import deepcopy

import einops
import mujoco

from imitation.utils.domain_randomization_wrapper import FixedDomainRandomizationWrapper

benchmarks = {
    "debug_1": ["apple_in_basket"],
    "banana_2": [
        "banana_in_box",
        "banana_in_box_2",
    ],
    "cup_stack": ["cup_stack"],
    "multitask": [
        "apple_in_blue_cup",
        "apple_in_grey_bowl",
        "banana_in_grey_bowl",
        "blue_cup_in_pink_cup",
        "grapes_in_blue_bowl",
        "grey_bowl_in_blue_bowl",
    ],
    "multitask_extra": [
        "apple_in_blue_cup",
        "apple_in_grey_bowl",
        "banana_in_grey_bowl",
        "blue_cup_in_pink_cup",
        "grapes_in_blue_bowl",
        "grey_bowl_in_blue_bowl",
        "apple_in_blue_cup_2",
        "apple_in_grey_bowl_2",
        "banana_in_grey_bowl_2",
        "blue_cup_in_pink_cup_2",
    ],
}


task_info = {
    "apple_in_basket": {
        "crop_info": {
            "front_realsense": {"x_low": 80, "y_low": 0, "x_high": 320, "y_high": 240},
            "wrist_left_realsense": {"x_low": 92, "y_low": 0, "x_high": 332, "y_high": 240},
        },
        "instruction": "pick up the apple and place it in the box",
    },
    "banana_in_box": {
        "crop_info": {
            "front_realsense": {"x_low": 80, "y_low": 0, "x_high": 320, "y_high": 240},
            "wrist_left_realsense": {"x_low": 92, "y_low": 0, "x_high": 332, "y_high": 240},
        },
        "instruction": "pick up the banana and place it in the box",
    },
    "banana_in_box_2": {
        "crop_info": {
            "front_realsense": {"x_low": 80, "y_low": 0, "x_high": 320, "y_high": 240},
            "wrist_left_realsense": {"x_low": 92, "y_low": 0, "x_high": 332, "y_high": 240},
        },
        "instruction": "pick up the banana and place it in the box",
    },
    "cup_stack": {
        "crop_info": {
            "front_realsense": {"x_low": 80, "y_low": 0, "x_high": 320, "y_high": 240},
            "wrist_left_realsense": {"x_low": 92, "y_low": 0, "x_high": 332, "y_high": 240},
        },
        "instruction": "pick up the blue cup and stack it in the red cup",
    },
    "apple_in_blue_cup": {
        "crop_info": {
            "front_realsense": {"x_low": 80, "y_low": 0, "x_high": 320, "y_high": 240},
            "wrist_left_realsense": {"x_low": 92, "y_low": 0, "x_high": 332, "y_high": 240},
        },
        "instruction": "pick up the apple and put it in the blue cup",
    },
    "apple_in_grey_bowl": {
        "crop_info": {
            "front_realsense": {"x_low": 80, "y_low": 0, "x_high": 320, "y_high": 240},
            "wrist_left_realsense": {"x_low": 92, "y_low": 0, "x_high": 332, "y_high": 240},
        },
        "instruction": "pick up the apple and put it in the grey bowl",
    },
    "banana_in_grey_bowl": {
        "crop_info": {
            "front_realsense": {"x_low": 80, "y_low": 0, "x_high": 320, "y_high": 240},
            "wrist_left_realsense": {"x_low": 92, "y_low": 0, "x_high": 332, "y_high": 240},
        },
        "instruction": "pick up the banana and put it in the grey bowl",
    },
    "blue_cup_in_pink_cup": {
        "crop_info": {
            "front_realsense": {"x_low": 80, "y_low": 0, "x_high": 320, "y_high": 240},
            "wrist_left_realsense": {"x_low": 92, "y_low": 0, "x_high": 332, "y_high": 240},
        },
        "instruction": "pick up the blue cup and stack it in the pink cup",
    },
    "grapes_in_blue_bowl": {
        "crop_info": {
            "front_realsense": {"x_low": 80, "y_low": 0, "x_high": 320, "y_high": 240},
            "wrist_left_realsense": {"x_low": 92, "y_low": 0, "x_high": 332, "y_high": 240},
        },
        "instruction": "pick up the grapes and put them in the blue bowl",
    },
    "grey_bowl_in_blue_bowl": {
        "crop_info": {
            "front_realsense": {"x_low": 80, "y_low": 0, "x_high": 320, "y_high": 240},
            "wrist_left_realsense": {"x_low": 92, "y_low": 0, "x_high": 332, "y_high": 240},
        },
        "instruction": "pick up the grey bowl and put it in the blue bowl",
    },
    "apple_in_blue_cup_2": {
        "crop_info": {
            "front_realsense": {"x_low": 80, "y_low": 0, "x_high": 320, "y_high": 240},
            "wrist_left_realsense": {"x_low": 92, "y_low": 0, "x_high": 332, "y_high": 240},
        },
        "instruction": "pick up the apple and put it in the blue cup",
    },
    "apple_in_grey_bowl_2": {
        "crop_info": {
            "front_realsense": {"x_low": 80, "y_low": 0, "x_high": 320, "y_high": 240},
            "wrist_left_realsense": {"x_low": 92, "y_low": 0, "x_high": 332, "y_high": 240},
        },
        "instruction": "pick up the apple and put it in the grey bowl",
    },
    "banana_in_grey_bowl_2": {
        "crop_info": {
            "front_realsense": {"x_low": 80, "y_low": 0, "x_high": 320, "y_high": 240},
            "wrist_left_realsense": {"x_low": 92, "y_low": 0, "x_high": 332, "y_high": 240},
        },
        "instruction": "pick up the banana and put it in the grey bowl",
    },
    "blue_cup_in_pink_cup_2": {
        "crop_info": {
            "front_realsense": {"x_low": 80, "y_low": 0, "x_high": 320, "y_high": 240},
            "wrist_left_realsense": {"x_low": 92, "y_low": 0, "x_high": 332, "y_high": 240},
        },
        "instruction": "pick up the blue cup and stack it in the pink cup",
    },
}


def get_boundaries(benchmark_name):
    boundaries = np.array(((-0.2, 0.1, -0.2), (0.6, 1, 0.6)))
    boundaries = einops.repeat(boundaries, "i j -> n i j", n=len(benchmarks[benchmark_name]))
    # boundaries = np.repeat(boundaries, (len(benchmarks[benchmark_name]), 1, 1))
    return boundaries


def process_images_and_intrinsics(
    rgb, depth, intrinsics, img_height=None, img_width=None, crop_info=None
):
    B, H, W, _ = rgb.shape
    if crop_info is not None:
        x_low = crop_info["x_low"]
        y_low = crop_info["y_low"]
        x_high = crop_info["x_high"]
        y_high = crop_info["y_high"]

        rgb = rgb[:, y_low:y_high, x_low:x_high]
        depth = depth[:, y_low:y_high, x_low:x_high]
        _, cropped_H, cropped_W, _ = rgb.shape

        intrinsics = crop_update_intrinsics(intrinsics, crop_info)
    else:
        cropped_H = H
        cropped_W = W

    assert (img_height is not None) == (
        img_width is not None
    ), "if you want to resize you must specify height and width"
    if img_height is not None:
        rgb_new = []
        depth_new = []
        for i in range(B):
            rgb_new.append(cv2.resize(rgb[i], (img_width, img_height)))
            depth_new.append(cv2.resize(depth[i], (img_width, img_height)))
        rgb = np.array(rgb_new)
        depth = np.array(depth_new)

        intrinsics = resize_update_intrinsics(
            intrinsics, (cropped_H, cropped_W), (img_height, img_width)
        )

    return rgb, depth, intrinsics


def glue_obs(obs):
    """
    For our real experiments where the observations and actions are broken into chunks
    """
    keys = list(obs)
    chunks = list(obs[keys[0]])
    chunks.sort()

    obs_out = {}
    for key in keys:
        try:
            obs_out[key] = np.concatenate([np.asarray(obs[key][chunk]) for chunk in chunks], axis=0)
        except KeyError:
            continue
    return obs_out


def remove_no_action_head(obs, actions):
    """
    Some demos have a long head with zero actions and a video of
    us rearranging the scene. Remove it
    """

    diffs = actions[1:] - actions[:-1]
    norm_diffs = np.linalg.norm(diffs, axis=1)
    start_ind = np.argmax(norm_diffs > 1e-8)  # returns the first true
    actions = actions[start_ind:]
    obs = {key: value[start_ind:] for key, value in obs.items()}
    return obs, actions


def crop_update_intrinsics(intrinsics, crop_info):
    x_low = crop_info["x_low"]
    y_low = crop_info["y_low"]

    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    new_cx = cx - x_low
    new_cy = cy - y_low

    new_intrinsics = np.array(intrinsics)
    new_intrinsics[0, 2] = new_cx
    new_intrinsics[1, 2] = new_cy
    return new_intrinsics


def resize_update_intrinsics(intrinsics, orig_size, new_size):
    # orig_height, orig_width = orig_size
    # new_height, new_width = new_size
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    factor_y, factor_x = np.array(new_size) / np.array(orig_size)

    fx_new = fx * factor_x
    fy_new = fy * factor_y
    cx_new = cx * factor_x
    cy_new = cy * factor_y

    return np.array([[fx_new, 0, cx_new], [0, fy_new, cy_new], [0, 0, 1]])


def build_dataset(
    data_prefix,
    suite_name,
    benchmark_name,
    seq_len,
    frame_stack,
    shape_meta,
    n_demos,
    hdf5_cache_mode="low_dim",
    extra_obs_modality=None,
    obs_seq_len=1,
    load_obs=True,
    task_embedding_format="clip",
    load_next_obs=False,
    dataset_keys=("actions",),
):
    assert (
        benchmark_name is not None
    ), "you must specify a benchmark_name with `task.benchmark_name=xyz`"

    task_list = benchmarks[benchmark_name]
    n_tasks = len(task_list)
    manip_datasets = []
    descriptions = []
    # for key, value in shape_meta
    obs_modality = {
        "rgb": list(shape_meta["observation"]["rgb"].keys()),
        "depth": list(shape_meta["observation"]["depth"].keys()),
        "low_dim": list(shape_meta["observation"]["lowdim"].keys())
        + list(shape_meta["observation"]["pointcloud"].keys()),
    }
    if extra_obs_modality is not None:
        for key in extra_obs_modality:
            obs_modality[key] = obs_modality[key] + extra_obs_modality[key]

    # if abs_action:
    #     dataset_keys = ('abs_actions',)
    # breakpoint()
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})
    for i, task_name in enumerate(tqdm(task_list)):
        # dataset_path=os.path.join(data_prefix, suite_name, benchmark.get_task_demonstration(i))
        # data = h5py.File(dataset_path, 'r')
        task_i_dataset = get_dataset(
            dataset_path=os.path.join(data_prefix, suite_name, f"{task_name}.hdf5"),
            obs_modality=obs_modality,
            seq_len=seq_len,
            obs_seq_len=obs_seq_len,
            frame_stack=frame_stack,
            load_obs=load_obs,
            n_demos=n_demos,
            hdf5_cache_mode=hdf5_cache_mode,
            load_next_obs=load_next_obs,
            dataset_keys=dataset_keys,
        )
        task_description = task_info[task_name]["instruction"]
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)
    task_embs = get_task_embs(task_embedding_format, descriptions)
    datasets = [
        SequenceVLDataset(ds, emb, i) for i, (ds, emb) in enumerate(zip(manip_datasets, task_embs))
    ]
    n_demos = [data.n_demos for data in datasets]
    n_sequences = [data.total_num_sequences for data in datasets]
    concat_dataset = ConcatDataset(datasets)
    print("\n===================  Benchmark Information  ===================")
    print(f" Name: {benchmark_name}")
    print(f" # Tasks: {n_tasks}")
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")
    return concat_dataset


def get_dataset(
    dataset_path,
    obs_modality,
    seq_len=1,
    obs_seq_len=1,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode="low_dim",
    load_obs=True,
    n_demos=None,
    load_next_obs=False,
    dataset_keys=None,
):
    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
    )
    seq_len = seq_len
    filter_key = filter_key
    if load_obs:
        obs_keys = shape_meta["all_obs_keys"]
    else:
        obs_keys = []

    if dataset_keys is None:
        dataset_keys = [
            "actions",
        ]
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        dataset_keys=dataset_keys,
        load_next_obs=load_next_obs,
        frame_stack=frame_stack,
        seq_length=seq_len,  # length-10 temporal sequences
        obs_seq_length=obs_seq_len,
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=False,
        hdf5_normalize_obs=None,
        filter_by_attribute=filter_key,  # can optionally provide a filter key here
        n_demos=n_demos,
    )
    # x = dataset.normalize_action()
    # breakpoint()
    # dataset.normalize_obs()
    # breakpoint()
    return dataset


class SequenceVLDataset(Dataset):
    def __init__(self, sequence_dataset, task_emb, task_id):
        self.sequence_dataset = sequence_dataset
        self.task_emb = task_emb
        self.task_id = task_id
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        return_dict = self.sequence_dataset.__getitem__(idx)
        return_dict["task_emb"] = self.task_emb
        return_dict["task_id"] = self.task_id
        return return_dict


def get_task_embs(task_embedding_format, descriptions):
    logging.set_verbosity_error()
    if task_embedding_format == "bert":
        tz = AutoTokenizer.from_pretrained("bert-base-cased", cache_dir=to_absolute_path("./bert"))
        model = AutoModel.from_pretrained("bert-base-cased", cache_dir=to_absolute_path("./bert"))
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        masks = tokens["attention_mask"]
        input_ids = tokens["input_ids"]
        task_embs = model(tokens["input_ids"], tokens["attention_mask"])["pooler_output"].detach()
    elif task_embedding_format == "gpt2":
        tz = AutoTokenizer.from_pretrained("gpt2")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("gpt2")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["last_hidden_state"].detach()[:, -1]
    elif task_embedding_format == "clip":
        tz = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model.get_text_features(**tokens).detach()
    elif task_embedding_format == "roberta":
        tz = AutoTokenizer.from_pretrained("roberta-base")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("roberta-base")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["pooler_output"].detach()
    return task_embs