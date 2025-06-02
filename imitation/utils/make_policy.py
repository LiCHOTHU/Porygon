import einops
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from hydra.utils import instantiate
from imitation.utils.pytorch3d_transforms import quaternion_to_axis_angle, quaternion_to_matrix

# import imitation.utils.real_utils as ru
import imitation.utils.utils as utils

from imitation.utils.point_cloud_utils import lift_point_cloud_batch
from imitation.utils.geometry import posRotMat2Mat_batch


def make_policy(ckpt_path, task, device, hd_obs=False, overrides=None):
    state_dict = utils.load_state(ckpt_path)
    policy_cfg = state_dict["config"]["algo"]["policy"]
    if overrides is not None:
        utils.recursive_update(policy_cfg, overrides)

    # TODO: this is a hack to make DP work after I forgot to remove a bug
    if "rotation_transformer" in policy_cfg:
        policy_cfg["rotation_transformer"]["from_rep"] = "quaternion"
    model = instantiate(policy_cfg)
    model.to(device)
    model.load_state_dict(state_dict["model"])
    model.normalizer.fit(state_dict["norm_stats"])
    model.eval()

    return PolicyWrapper(model, task, hd_obs)


class PolicyWrapper:
    def __init__(self, model, benchmark, task, hd_obs):
        self.model = model
        self.hd_obs = hd_obs

        description = ru.task_info[task]["instruction"]
        self.task_emb = ru.get_task_embs("clip", [description])
        self.task_id = ru.get_task_id(benchmark, task)

    def predict_action(self, obs):
        print(set(obs["cameras"]))

        # TODO: add batch dim to everything
        if "front_realsense" in obs["cameras"]:
            scene_cam_obs = obs["cameras"]["front_realsense"]
        elif "front_zed" in obs["cameras"]:
            scene_cam_obs = obs["cameras"]["front_zed"]

        scene_cam = np.expand_dims(scene_cam_obs["rgb"], 0)
        scene_cam_depth = np.expand_dims(scene_cam_obs["depth"], 0)
        scene_cam_intrinsic = scene_cam_obs["intrinsics"]
        scene_cam_pos = np.expand_dims(scene_cam_obs["position"], 0)
        scene_cam_quat = np.expand_dims(scene_cam_obs["quat_wxyz"], 0)

        # TODO: update this
        wrist_cam_obs = obs["cameras"]["wrist_left_realsense"]
        wrist_cam = np.expand_dims(wrist_cam_obs["rgb"], 0)
        wrist_cam_depth = np.expand_dims(wrist_cam_obs["depth"], 0)
        wrist_cam_intrinsic = wrist_cam_obs["intrinsics"]
        wrist_cam_pos = np.expand_dims(wrist_cam_obs["position"], 0)
        wrist_cam_quat = np.expand_dims(wrist_cam_obs["quat_wxyz"], 0)

        if self.hd_obs:
            scene_cam, scene_cam_depth, scene_cam_intrinsic = ru.process_images_and_intrinsics(
                scene_cam,
                scene_cam_depth,
                scene_cam_intrinsic,
                img_height=240,
                img_width=424,
            )

        if wrist_cam_pos[0] is None:
            # TODO: replace with last outputted pose
            return np.array([[0.5, -0.5, 0.2, 1.0, 0.0, 0.0, 0.0, 0.085]], dtype=np.float64)

        scene_cam, scene_cam_depth, scene_cam_intrinsic = ru.process_images_and_intrinsics(
            scene_cam,
            scene_cam_depth,
            scene_cam_intrinsic,
            128,
            128,
            ru.task_info["apple_in_basket"]["crop_info"]["front_realsense"],
        )

        wrist_cam, wrist_cam_depth, wrist_cam_intrinsic = ru.process_images_and_intrinsics(
            wrist_cam,
            wrist_cam_depth,
            wrist_cam_intrinsic,
            128,
            128,
            ru.task_info["apple_in_basket"]["crop_info"]["wrist_left_realsense"],
        )

        scene_cam = torch.tensor(scene_cam, device="cpu")
        scene_cam_depth = torch.tensor(scene_cam_depth, dtype=torch.float16, device="cpu")
        scene_cam_intrinsic = torch.tensor(scene_cam_intrinsic, device="cpu").repeat(
            (scene_cam.shape[0], 1, 1)
        )
        scene_cam_pos = torch.tensor(scene_cam_pos, device="cpu")
        scene_cam_quat = torch.tensor(scene_cam_quat, device="cpu")
        scene_cam_rot_mat = quaternion_to_matrix(scene_cam_quat)
        scene_cam_mat = posRotMat2Mat_batch(scene_cam_pos, scene_cam_rot_mat)
        scene_cam_pcd = lift_point_cloud_batch(
            scene_cam_depth, scene_cam_intrinsic, scene_cam_mat, maintain_dims=True
        )

        # scene_cam_depth = scene_cam_depth.to(device='cuda')
        # scene_cam_intrinsic = scene_cam_intrinsic.to(device='cuda')
        # scene_cam_mat = scene_cam_mat.to(device='cuda')
        # import time
        # t = time.time()
        # for _ in range(10):
        #     lift_point_cloud_batch(scene_cam_depth, scene_cam_intrinsic, scene_cam_mat)
        # print(time.time() - t)
        # exit()

        wrist_cam = torch.tensor(wrist_cam, device="cpu")
        wrist_cam_depth = torch.tensor(wrist_cam_depth, dtype=torch.float16, device="cpu")
        wrist_cam_intrinsic = torch.tensor(wrist_cam_intrinsic, device="cpu").repeat(
            (wrist_cam.shape[0], 1, 1)
        )
        wrist_cam_pos = torch.tensor(wrist_cam_pos, device="cpu")
        wrist_cam_quat = torch.tensor(wrist_cam_quat, device="cpu")
        wrist_cam_rot_mat = quaternion_to_matrix(wrist_cam_quat)
        wrist_cam_mat = posRotMat2Mat_batch(wrist_cam_pos, wrist_cam_rot_mat)
        wrist_cam_pcd = lift_point_cloud_batch(
            wrist_cam_depth, wrist_cam_intrinsic, wrist_cam_mat, maintain_dims=True
        )

        # ee_pose = torch.tensor(obs["ee_pose"], device="cpu")
        # ee_pos, ee_quat = np.split(ee_pose, [3], axis=-1)
        ee_pos = torch.tensor(obs["robots"]["UR5_left"]["ee_trans"], device="cpu").unsqueeze(0)
        ee_quat = torch.tensor(obs["robots"]["UR5_left"]["ee_quat_wxyz"], device="cpu").unsqueeze(0)
        ee_axis_angle = quaternion_to_axis_angle(ee_quat)
        ee_rot_mat = quaternion_to_matrix(ee_quat)
        ee_mat = posRotMat2Mat_batch(ee_pos, ee_rot_mat)
        ee_mat_inv = torch.linalg.inv(ee_mat)

        clouds = {"scene_cam_pcd": scene_cam_pcd, "wrist_cam_pcd": wrist_cam_pcd}
        for key, cloud in clouds.items():
            cloud = einops.rearrange(cloud, "b h w c -> (b h w) c")

            o3d_cloud = o3d.geometry.PointCloud()
            o3d_cloud.points = o3d.utility.Vector3dVector(cloud.cpu().numpy().astype(np.float32))

            _, ind = o3d_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            mask = np.zeros(cloud.shape[0])
            mask[ind] = 1
            cloud[:, :] = cloud * torch.tensor(mask, device=cloud.device).reshape(-1, 1)

        # xyz = np.concatenate(
        #     [
        #         einops.rearrange(scene_cam_pcd, "b h w c -> b (h w) c").cpu().numpy(),
        #         einops.rearrange(wrist_cam_pcd, "b h w c -> b (h w) c").cpu().numpy(),
        #     ],
        #     axis=1,
        # )
        # rgb = (
        #     np.concatenate(
        #         [
        #             einops.rearrange(scene_cam, "b h w c -> b (h w) c").cpu().numpy(),
        #             einops.rearrange(wrist_cam, "b h w c -> b (h w) c").cpu().numpy(),
        #         ],
        #         axis=1,
        #     )
        #     / 255
        # )

        # o3d_cloud = o3d.geometry.PointCloud()
        # o3d_cloud.points = o3d.utility.Vector3dVector(xyz[0])
        # o3d_cloud.colors = o3d.utility.Vector3dVector(rgb[0])
        # o3d.visualization.draw_geometries([o3d_cloud])
        # show_point_cloud(xyz[0], rgb[0])

        # TODO: something less hacky
        proprios = [
            "elbow_joint",
            "shoulder_lift_joint",
            "shoulder_pan_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        joint_positions = [obs["robots"]["UR5_left"]["joint_position"][key] for key in proprios]
        joint_positions = torch.tensor(joint_positions).unsqueeze(0)
        gripper_pos = torch.tensor(obs["robots"]["UR5_left"]["gripper_pos"], dtype=torch.float32).unsqueeze(0)

        obs = {
            "scene_cam_rgb": einops.rearrange(scene_cam, "b h w c -> b c h w"),
            "scene_cam_pointcloud_full": scene_cam_pcd,
            "wrist_cam_rgb": einops.rearrange(wrist_cam, "b h w c -> b c h w"),
            "wrist_cam_pointcloud_full": wrist_cam_pcd,
            "hand_mat": ee_mat,
            "hand_mat_inv": ee_mat_inv,
            "ee_pos": ee_pos,
            "ee_axis_angle": ee_axis_angle,
            "gripper_pos": gripper_pos,
            "joint_positions": joint_positions,
        }
        for key, value in obs.items():
            value = einops.rearrange(value, 'b ... -> b 1 ...') # add frame stack dimension
            value = value.to(device='cuda')
            obs[key] = value

        task_id = self.task_id
        task_emb = torch.tensor(self.task_emb, device="cuda").reshape((1, -1))
        actions = self.model.get_action(obs, task_id, task_emb)
        actions = actions.squeeze()


        return actions.cpu().numpy().astype(np.float64)


__all__ = ["make_policy"]
