import os
import json
import h5py
import time
import argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import torch

import robomimic.utils.tensor_utils as TensorUtils
import robosuite
import robosuite.utils.camera_utils as CameraUtils
from termcolor import colored
from scipy.spatial.transform import Rotation
import robosuite.utils.transform_utils as T

import sys
from imitation.utils.geometry import posRotMat2Mat, quat2mat

# IMPORTANT: you need to import the package to register the environments
print('HEY')
import dexmimicgen.utils

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)


def get_env_metadata_from_dataset(dataset_path, ds_format="robomimic"):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    if ds_format == "robomimic":
        env_meta = json.loads(f["data"].attrs["env_args"])
    else:
        raise ValueError
    f.close()
    return env_meta


def reset_to(env, state):
    """
    Reset to a specific simulator state.

    Args:
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml

    Returns:
        observation (dict): observation dictionary after setting the simulator state (only
            if "states" is in @state)
    """
    should_ret = False
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            # set relevant episode information
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):  # newer versions
            env.set_ep_meta(ep_meta)
        # this reset is necessary.
        # while the call to env.reset_from_xml_string does call reset,
        # that is only a "soft" reset that doesn't actually reload the model.
        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = env.edit_model_xml(state["model"])

        env.reset_from_xml_string(xml)
        env.sim.reset()
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
        should_ret = True

    # update state as needed
    if hasattr(env, "update_sites"):
        # older versions of environment had update_sites function
        env.update_sites()
    if hasattr(env, "update_state"):
        # later versions renamed this to update_state
        env.update_state()

    return None


# def get_camera_extrinsic_matrix(sim, camera_name):
#     """
#     Returns a 4x4 homogenous matrix corresponding to the camera pose in the
#     world frame. MuJoCo has a weird convention for how it sets up the
#     camera body axis, so we also apply a correction so that the x and y
#     axis are along the camera view and the z axis points along the
#     viewpoint.
#     Normal camera convention: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

#     Args:
#         sim (MjSim): simulator instance
#         camera_name (str): name of camera
#     Return:
#         R (np.array): 4x4 camera extrinsic matrix
#     """
#     cam_id = sim.model.camera_name2id(camera_name)
#     camera_pos = sim.data.cam_xpos[cam_id]
#     camera_rot = sim.data.cam_xmat[cam_id].reshape(3, 3)
#     R = T.make_pose(camera_pos, camera_rot)

#     # IMPORTANT! This is a correction so that the camera axis is set up along the viewpoint correctly.
#     camera_axis_correction = np.array(
#         [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
#     )
#     R = R @ camera_axis_correction
#     return R


def extract_trajectory(
    env, 
    initial_state, 
    states, 
    actions,
    done_mode,
    camera_names, 
    camera_height=84, 
    camera_width=84,
    randomize_camera_poses=False,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env: robosuite environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a 
            success state. If 1, done is 1 at the end of each trajectory. 
            If 2, do both.
        camera_names (list): list of camera names to use for observations
        camera_height (int): height of camera observations
        camera_width (int): width of camera observations
        randomize_camera_poses (bool): whether to randomize camera poses
    """
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    reset_to(env, initial_state)
    env.sim.step()
    obs = env._get_observations()

    if randomize_camera_poses and "agentview" in camera_names:
        # Get base position for randomization
        base_pos = env.sim.data.body('gripper0_eef').xpos if hasattr(env.sim.data, 'body') else np.array([0, 0, 0])

        camera_name = 'agentview'
        cam_id = env.sim.model.camera_name2id(camera_name)
        old_position = env.sim.model.cam_pos[cam_id].copy()
        old_rotation = env.sim.model.cam_quat[cam_id].copy()

        theta = np.random.uniform(-2 * np.pi / 3, 2 * np.pi / 3)
        r = old_position[0] - base_pos[0]
        x = r * np.cos(theta) + base_pos[0]
        y = r * np.sin(theta) + base_pos[1]
        z = old_position[2]
        new_position = np.array([x, y, z])

        Rz = Rotation.from_euler('z', theta).as_matrix()
        R_ref = Rotation.from_quat(old_rotation, scalar_first=True).as_matrix()
        R_total = Rz @ R_ref
        new_quat = Rotation.from_matrix(R_total).as_quat(scalar_first=True)

        env.sim.model.cam_pos[cam_id] = new_position
        env.sim.model.cam_quat[cam_id] = new_quat
        env.sim.step()
        obs = env._get_observations()

    traj = dict(
        obs=[], 
        rewards=[], 
        dones=[],
        actions=np.array(actions), 
        states=np.array(states), 
        initial_state_dict=initial_state,
    )

    traj_len = states.shape[0]

    for t in range(1, traj_len + 1):
        # get next observation
        if t == traj_len or True:
            # play final action to get next observation for last timestep
            next_obs, _, _, _ = env.step(actions[t - 1])
        else:
            # reset to previous state and step with action to update controller properly
            prev_state = {"states": states[t - 1]}
            reset_to(env, prev_state)
            # step with the action to update both state and controller
            next_obs, _, _, _ = env.step(actions[t - 1])

        # Process camera observations if requested
        for cam_name in camera_names:
            obs[cam_name + "_image"] = obs[cam_name + "_image"][::-1]

            depth_f32 = obs[cam_name + "_depth"]
            depth_f32 = CameraUtils.get_real_depth_map(env.sim, depth_f32)
            depth_uint16 = (depth_f32 * 1000).astype(np.uint16)[::-1]
            obs[cam_name + "_depth"] = depth_uint16

            # Try to get camera matrices if available
            R = CameraUtils.get_camera_extrinsic_matrix(
                env.sim,
                camera_name=cam_name
            )
            obs[cam_name + "_extrinsic"] = R.astype(np.float32)
            
            K = CameraUtils.get_camera_intrinsic_matrix(
                env.sim, 
                camera_name=cam_name, 
                camera_height=camera_height, 
                camera_width=camera_width
            )
            obs[cam_name + "_intrinsic"] = K.astype(np.float32)

        # Extract hand pose information
        for hand in ["right", "left"]:
            hand_pos = obs[f"robot0_{hand}_eef_pos"]
            hand_quat = obs[f"robot0_{hand}_eef_quat"]
            hand_rot_mat = Rotation.from_quat(hand_quat, scalar_first=True).as_matrix()
            correction_mat = np.array(((0, 0, 1), (1, 0, 0), (0, 1, 0)), dtype=np.float32)
            hand_rot_mat = hand_rot_mat @ correction_mat
            hand_mat = posRotMat2Mat(hand_pos, hand_rot_mat)
            hand_mat_inv = np.linalg.inv(hand_mat)
            obs[f"robot0_{hand}_eef_mat"] = hand_mat
            obs[f"robot0_{hand}_eef_mat_inv"] = hand_mat_inv

        # bodies = [env.sim.model.body_id2name(i) for i in range(env.sim.model.nbody)]
        # breakpoint()
        # try:
        #     if hasattr(env.sim.data, 'body') and 'gripper0_eef' in [env.sim.model.body_id2name(i) for i in range(env.sim.model.nbody)]:
        #         eef_data = env.sim.data.body('gripper0_eef')
        #         hand_pos = eef_data.xpos
        #         hand_quat = eef_data.xquat
        #         hand_rot_mat = quat2mat(hand_quat)
        #         hand_rot_mat = hand_rot_mat @ np.array(((0, 0, 1), (0, 1, 0), (1, 0, 0)))
        #         hand_mat = posRotMat2Mat(hand_pos, hand_rot_mat)
        #         hand_mat_inv = np.linalg.inv(hand_mat)
        #         obs["hand_mat"] = hand_mat.astype(np.float32)
        #         obs["hand_mat_inv"] = hand_mat_inv.astype(np.float32)
        # except:
        #     # If hand pose extraction fails, continue without it
        #     pass

        # infer reward signal
        r = 0.0
        if hasattr(env, 'get_reward'):
            try:
                r = env.get_reward()
            except:
                r = 0.0

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            if hasattr(env, '_check_success'):
                try:
                    done = done or env._check_success()
                except:
                    pass
            elif hasattr(env, 'is_success'):
                try:
                    success_info = env.is_success()
                    if isinstance(success_info, dict) and "task" in success_info:
                        done = done or success_info["task"]
                    else:
                        done = done or success_info
                except:
                    pass
        done = int(done)

        # collect transition
        traj["obs"].append(deepcopy(obs))
        traj["rewards"].append(r)
        traj["dones"].append(done)

        # update for next iter
        obs = deepcopy(next_obs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj


def dataset_states_to_obs(args):
    if args.depth:
        assert len(args.camera_names) > 0, "must specify camera names if using depth"

    # get env metadata and create environment using dexmimicgen approach
    env_meta = get_env_metadata_from_dataset(dataset_path=args.dataset)

    breakpoint()
    
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = len(args.camera_names) > 0
    env_kwargs["camera_names"] = args.camera_names
    env_kwargs["camera_heights"] = args.camera_height
    env_kwargs["camera_widths"] = args.camera_width
    env_kwargs["camera_depths"] = args.depth

    if args.verbose:
        print(
            colored(
                "Initializing environment for {}...".format(env_kwargs["env_name"]),
                "yellow",
            )
        )
    
    # Remove problematic keys if they exist
    if "env_lang" in env_kwargs:
        env_kwargs.pop("env_lang")
    
    env = robosuite.make(**env_kwargs)

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    # output file in same directory as input file
    output_path = args.output_name

    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    print("input file: {}".format(args.dataset))
    print("output file: {}".format(output_path))

    total_samples = 0
    for ind in tqdm(range(len(demos))):
        ep = demos[ind]

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        if args.use_current_model:
            initial_state["model"] = env.sim.model.get_xml()
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)

        # extract obs, rewards, dones
        actions = f["data/{}/actions".format(ep)][()]
        traj = extract_trajectory(
            env=env, 
            initial_state=initial_state, 
            states=states, 
            actions=actions,
            done_mode=args.done_mode,
            camera_names=args.camera_names, 
            camera_height=args.camera_height, 
            camera_width=args.camera_width,
            randomize_camera_poses=args.randomize_camera_poses,
        )

        # get action_dict from original demo
        action_dict = None
        if "action_dict" in f["data/{}".format(ep)]:
            action_dict_grp = f["data/{}/action_dict".format(ep)]
            action_dict = {}
            for key in action_dict_grp.keys():
                action_dict[key] = action_dict_grp[key][()]

        # maybe copy reward or done signal from source file
        if args.copy_rewards:
            traj["rewards"] = f["data/{}/rewards".format(ep)][()]
        if args.copy_dones:
            traj["dones"] = f["data/{}/dones".format(ep)][()]

        # store transitions
        ep_data_grp = data_grp.create_group(ep)
        ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
        ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
        ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
        ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
        
        # save action_dict if it exists
        if action_dict is not None:
            action_dict_grp = ep_data_grp.create_group("action_dict")
            for key, value in action_dict.items():
                action_dict_grp.create_dataset(key, data=value)
 
        for k in traj["obs"]:
            ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))

        # episode metadata
        ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]
        ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]
        if traj["initial_state_dict"].get("ep_meta", None) is not None:
            ep_data_grp.attrs["ep_meta"] = traj["initial_state_dict"]["ep_meta"]

        total_samples += traj["actions"].shape[0]

    # copy over all filter keys that exist in the original hdf5
    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(env_meta, indent=4)
    print("Wrote {} trajectories to {}".format(len(demos), output_path))

    f.close()
    f_out.close()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hdf5_path",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="path to output hdf5 dataset",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["agentview", "robot0_eye_in_right_hand", "robot0_eye_in_left_hand", "robot0_robotview"],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )
    parser.add_argument(
        "--camera_height",
        type=int,
        default=128, 
        help="(optional) height of image observations",
    )
    parser.add_argument(
        "--camera_width",
        type=int,
        default=128, 
        help="(optional) width of image observations",
    )
    parser.add_argument(
        "--depth", 
        action='store_true',
        help="(optional) use depth observations for each camera",
    )
    parser.add_argument(
        "--done_mode",
        type=int,
        default=2,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )
    parser.add_argument(
        "--copy_rewards", 
        action='store_true',
        help="(optional) copy rewards from source file instead of inferring them",
    )
    parser.add_argument(
        "--copy_dones", 
        action='store_true',
        help="(optional) copy dones from source file instead of inferring them",
    )
    parser.add_argument(
        "--randomize_camera_poses",
        action='store_true',
        help="(optional) randomize camera poses",
    )
    parser.add_argument(
        "--use_current_model",
        action='store_true',
        help="(optional) use the current model instead of the one stored in the dataset",
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="(optional) enable verbose output",
    )
    parser.add_argument(
        "--allow_overwrite",
        action='store_true',
        help="(optional) allow overwriting existing output file",
    )
  
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    input_path = args.hdf5_path
    output_dir = args.output_dir
            
    print(f"Processing {input_path}...")
    
    args.dataset = input_path
    args.output_name = output_dir

    if os.path.isfile(args.output_name) and not args.allow_overwrite:
        print(f"Output file {args.output_name} already exists. Skipping processing.")
        exit(0)
    
    print(f"input file: {args.dataset}")
    print(f"output file: {args.output_name}")
    
    dataset_states_to_obs(args)
