from argparse import ArgumentParser
import h5py
from hydra.utils import instantiate
import numpy as np
import torch
import torch.nn.functional as F
import einops
import dgl.geometry as dgl_geo
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm, trange
import imageio
import math
import os
import viser
from pathlib import Path

import imitation.utils.utils as utils
from imitation.algos.base import ChunkPolicy
from imitation.algos.utils.encoder import Adapt3REncoder, image_key_to_pointcloud_key, image_key_to_camera_name
from imitation.utils.point_cloud_utils import show_point_cloud, batch_transform_point_cloud

HAND_FRAME_CROP = ((-0.05, -1, -1), (1, 1, 1))


def obs_to_heatmap(data, encoder, start, end, alpha=0.5):
    with torch.no_grad():
        obs_data = data['obs']
            
        rgb = []
        pcd = []

        for rgb_key in encoder.shape_meta["observation"]["rgb"]:
            rgb.append(obs_data[rgb_key])
            pcd.append(obs_data[image_key_to_pointcloud_key(rgb_key)])

        assert len(rgb) == len(pcd)

        rgb = torch.stack(rgb)
        pcd = torch.stack(pcd).to(dtype=torch.float32)

        rgb = rgb[:, start:end]
        pcd = pcd[:, start:end]

        device = rgb.device

        n_cam, B, fs, _, H, W = rgb.shape

        orig_rgb = einops.rearrange(rgb, "ncam b fs c h w -> (b fs) (ncam h w) c")
        orig_pcd = einops.rearrange(pcd, "ncam b fs h w c -> (b fs) (ncam h w) c")

        rgb = einops.rearrange(rgb, "ncam b fs c h w -> (b fs ncam) c h w")
        pcd = einops.rearrange(pcd, "ncam b fs h w c -> (b fs ncam) c h w")
        # TODO: remove this. This is for vis purposes

        # Pass each view independently through backbone
        if encoder.do_image:
            rgb_normalized = encoder.normalize(rgb)
            if encoder.finetune:
                if encoder.backbone_type == "fusion":
                    task_emb = einops.repeat(
                        data["task_emb"], "b d -> (b fs ncam) d", fs=fs, ncam=n_cam
                    )
                    rgb_features = encoder.backbone(rgb_normalized, langs=task_emb)
                else:
                    rgb_features = encoder.backbone(rgb_normalized)
            else:
                with torch.no_grad():
                    if encoder.backbone_type == "fusion":
                        task_emb = einops.repeat(
                            data["task_emb"], "b d -> (b fs ncam) d", fs=fs, ncam=n_cam
                        )
                        rgb_features = encoder.backbone(rgb_normalized, langs=task_emb)
                    else:
                        rgb_features = encoder.backbone(rgb_normalized)

            # Pass visual features through feature pyramid network
            # breakpoint()
            rgb_features = {key: F.interpolate(value, (H, W), mode='bilinear') for key, value in rgb_features.items()}
            rgb_features = encoder.feature_pyramid(rgb_features)
        else:
            rgb_features = {
                "res2": torch.zeros((B * n_cam, 60, 32, 32), device="cuda"),
                "res3": torch.zeros((B * n_cam, 60, 32, 32), device="cuda"),
            }


        rgb_feats_pyramid = []
        pcd_pyramid = []
        rgb_pyramid = []
        for i in range(encoder.num_sampling_level):
            # Isolate level's visual features
            rgb_features_i = rgb_features[encoder.feature_map_pyramid[i]]

            # Interpolate xy-depth to get the locations for this level
            feat_h, feat_w = rgb_features_i.shape[-2:]
            # pcd_i = F.interpolate(pcd, (feat_h, feat_w), mode="nearest")
            # rgb_i = F.interpolate(rgb, (feat_h, feat_w), mode="bilinear")
            pcd_i = pcd
            rgb_i = rgb

            # Merge different cameras for clouds, separate for rgb features
            h, w = pcd_i.shape[-2:]
            pcd_i = einops.rearrange(
                pcd_i, "(bt fs ncam) c h w -> (bt fs) (ncam h w) c", ncam=n_cam, fs=fs
            )
            rgb_features_i = einops.rearrange(
                rgb_features_i, "(bt fs ncam) c h w -> (bt fs) (ncam h w) c", ncam=n_cam, fs=fs
            )
            rgb_i = einops.rearrange(
                rgb_i, "(bt fs ncam) c h w -> (bt fs) (ncam h w) c", ncam=n_cam, fs=fs
            )

            rgb_feats_pyramid.append(rgb_features_i)
            pcd_pyramid.append(pcd_i)
            rgb_pyramid.append(rgb_i)


        rgb_feats = torch.cat(rgb_feats_pyramid, 1)
        pcd = torch.cat(pcd_pyramid, 1)
        rgb_pyramid = torch.cat(rgb_pyramid, 1)

        del rgb_feats_pyramid
        torch.cuda.empty_cache()

        _, n_pts, d_feat = rgb_feats.shape
        if encoder.do_crop:
            boundaries = encoder.boundaries[data["task_id"]]
            boundaries_low = einops.repeat(boundaries[:, 0], "b d -> (b fs) 1 d", fs=fs)
            boundaries_high = einops.repeat(boundaries[:, 1], "b d -> (b fs) 1 d", fs=fs)

            above_lower = torch.all(pcd > boundaries_low, dim=-1)
            below_upper = torch.all(pcd < boundaries_high, dim=-1)
            mask = torch.logical_and(above_lower, below_upper)

            indices = torch.masked_fill(torch.cumsum(mask.int(), dim=1), ~mask, 0)
            indices_repeat_3 = einops.repeat(indices, "b n -> b n k", k=3)
            indices_repeat_feat = einops.repeat(indices, "b n -> b n k", k=d_feat)
            masked_pcd = torch.scatter(
                input=torch.zeros((B, n_pts + 1, 3), device=device, dtype=pcd.dtype),
                index=indices_repeat_3,
                src=pcd,
                dim=1,
            )[:, 1:]
            masked_features = torch.scatter(
                input=torch.zeros((B, n_pts + 1, d_feat), device=device, dtype=rgb_feats.dtype),
                index=indices_repeat_feat,
                src=rgb_feats,
                dim=1,
            )[:, 1:]
            masked_rgb = torch.scatter(
                input=torch.zeros((B, n_pts + 1, 3), device=device, dtype=rgb_pyramid.dtype),
                index=indices_repeat_3,
                src=rgb_pyramid,
                dim=1,
            )[:, 1:]
            mask = ~torch.all(masked_pcd == 0, dim=-1)

        else:
            masked_pcd = pcd
            masked_features = rgb_feats
            masked_rgb = rgb_pyramid
            mask = torch.ones(pcd.shape[:-1], device=device)

        del rgb_feats
        torch.cuda.empty_cache()

        if encoder.do_hand_crop:
            # Convert to hand frame
            masked_pcd_1 = torch.cat(
                (masked_pcd, torch.ones((B * fs, masked_pcd.shape[1], 1), device=device)), dim=-1
            )
            hand_mat_inv = einops.rearrange(data["obs"]["hand_mat_inv"][start:end], "b fs i j -> (b fs) i j")
            masked_pcd_hand = torch.einsum("bnj,bij->bni", masked_pcd_1, hand_mat_inv)[..., :-1]
            # show_point_cloud_plt(masked_pcd_hand[0], masked_rgb[0])
            boundaries_low = einops.repeat(
                torch.tensor(HAND_FRAME_CROP[0], device=device), "d -> (b fs) 1 d", b=B, fs=fs
            )
            boundaries_high = einops.repeat(
                torch.tensor(HAND_FRAME_CROP[1], device=device), "d -> (b fs) 1 d", b=B, fs=fs
            )

            above_lower = torch.all(masked_pcd_hand > boundaries_low, dim=-1)
            below_upper = torch.all(masked_pcd_hand < boundaries_high, dim=-1)
            mask = torch.logical_and(mask, torch.logical_and(above_lower, below_upper))

            indices = torch.masked_fill(torch.cumsum(mask.int(), dim=1), ~mask, 0)
            indices_repeat_3 = einops.repeat(indices, "b n -> b n k", k=3)
            indices_repeat_feat = einops.repeat(
                indices, "b n -> b n k", k=masked_features.shape[-1]
            )
            masked_pcd_hand = torch.scatter(
                input=torch.zeros((B, n_pts + 1, 3), device=device, dtype=masked_pcd_hand.dtype),
                index=indices_repeat_3,
                src=masked_pcd_hand,
                dim=1,
            )[:, 1:]
            masked_features = torch.scatter(
                input=torch.zeros(
                    (B, n_pts + 1, d_feat), device=device, dtype=masked_features.dtype
                ),
                index=indices_repeat_feat,
                src=masked_features,
                dim=1,
            )[:, 1:]
            masked_rgb = torch.scatter(
                input=torch.zeros((B, n_pts + 1, 3), device=device, dtype=masked_rgb.dtype),
                index=indices_repeat_3,
                src=masked_rgb,
                dim=1,
            )[:, 1:]

            masked_pcd_hand_1 = torch.cat(
                (masked_pcd_hand, torch.ones((B * fs, masked_pcd_hand.shape[1], 1), device=device)),
                dim=-1,
            )
            hand_mat = einops.rearrange(data["obs"]["hand_mat"][start:end], "b fs i j -> (b fs) i j")
            masked_pcd = torch.einsum("bnj,bij->bni", masked_pcd_hand_1, hand_mat)[..., :-1]

        if encoder.hand_frame:
            masked_pcd_1 = torch.cat(
                (masked_pcd, torch.ones((B * fs, masked_pcd.shape[1], 1), device=device)), dim=-1
            )
            hand_mat_inv = einops.rearrange(data["obs"]["hand_mat_inv"][start:end], "b fs i j -> (b fs) i j")
            masked_pcd = torch.einsum("bnj,bij->bni", masked_pcd_1, hand_mat_inv)[..., :-1]


        downsampled_pcd = masked_pcd
        downsampled_feats = masked_features
        downsampled_rgb = masked_rgb
        num_points = masked_pcd.shape[1]
        downsample_mask = torch.ones(B, num_points, device=device).bool()

        del masked_pcd, masked_pcd_1, masked_features
        # del masked_pcd_1

        pcd_pos_emb = encoder.xyz_proj(downsampled_pcd)

        cat_cloud = []
        if encoder.do_pos:
            cat_cloud.append(pcd_pos_emb)
        if encoder.do_image:
            cat_cloud.append(downsampled_feats)
        if encoder.do_lang:
            lang_emb = encoder.get_task_emb(data)
            lang_emb = encoder.lang_proj(lang_emb)
            lang_emb = einops.repeat(lang_emb, "fs d -> (b fs) n d", b=B, n=num_points)
            cat_cloud.append(lang_emb)
        if encoder.do_rgb:
            cat_cloud.append(downsampled_rgb)
        cat_cloud = torch.cat(cat_cloud, dim=-1)


        cat_cloud = einops.rearrange(cat_cloud, "(b fs) n d -> b fs n d", b=B)
        downsample_mask = einops.rearrange(downsample_mask, "(b fs) n -> b fs n", b=B)
        xyz = einops.rearrange(downsampled_pcd, "(b fs) n c -> b fs n c", b=B)
        
        del downsampled_feats
        torch.cuda.empty_cache()

        extractor = encoder.pointcloud_extractor
        # import gc
        # gc.collect()
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #     except:
        #         pass
        x = extractor.mlp(cat_cloud)
        Q = extractor.Q_mlp(cat_cloud)
        logits = torch.einsum('bfnd,d->bfn', Q, extractor.key) / 16
        if downsample_mask is not None:
            logits = logits.masked_fill(~downsample_mask, float('-inf'))
        weights = torch.softmax(logits, dim=-1)

        cmap = plt.get_cmap('inferno')
        # weights_norm = weights.cpu().numpy()
        weights_norm = (weights - weights.min(dim=-1, keepdim=True).values) / (weights.max(dim=-1, keepdim=True).values)
        weights_norm = weights_norm.squeeze(1).cpu().numpy()
        heatmap = cmap(weights_norm)[..., :3]
        rgb = downsampled_rgb.cpu().numpy()

        # colors = rgb.copy()
        # for c in range(3):
        #     colors[..., c] = (1 - alpha * weights_norm) * rgb[..., c] + \
        #         (alpha * weights_norm) * heatmap[..., c]

        colors = heatmap * alpha + rgb * (1 - alpha)

        downsampled_pcd = batch_transform_point_cloud(downsampled_pcd, data["obs"]["hand_mat"][start:end])
        # for i in range(len(colors)):
        #     show_point_cloud(downsampled_pcd[i], colors[i])

        return downsampled_pcd.cpu().numpy(), colors


def make_point_cloud_video(points, colors, camera_pose, output_path="output", fps=20):
    assert points.shape == colors.shape
    B, N, _ = points.shape

    # Offscreen renderer (no window needed)
    width, height = 640, 480
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    # Set background to white (or change to black if you want)
    renderer.scene.set_background([1, 1, 1, 1])  # RGBA

    # Setup camera
    # Interpret your camera_pose (4x4 matrix) into look_at parameters
    cam_pos = camera_pose[:3, 3]  # camera position
    cam_forward = camera_pose[:3, 2]  # forward vector (Z axis)
    cam_up = camera_pose[:3, 1]  # up vector (Y axis)
    cam_target = cam_pos + cam_forward  # look at a point along forward vector

    # breakpoint()
    cam_target = np.array([-0.20806444,  0.        ,  0.6438522 ], dtype=np.float32)
    cam_pos = np.array([0.5 , 0.  , 1.35], dtype=np.float32)
    cam_up = np.array([ 0.70614785,  0.        , -0.70806444], dtype=np.float32)

    # Set camera
    renderer.scene.camera.look_at(cam_target, cam_pos, -cam_up)
    # ctr = vis.get_view_control()
    # param = o3d.camera.PinholeCameraParameters()
    # param = ctr.convert_to_pinhole_camera_parameters()
    # param.extrinsic = camera_pose

    images = []

    for b in tqdm(range(B), desc="Rendering frames"):
        # Create a point cloud for this frame
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[b])
        pcd.colors = o3d.utility.Vector3dVector(colors[b])

        # Clear scene and add geometry
        renderer.scene.clear_geometry()
        renderer.scene.add_geometry("pcd", pcd, o3d.visualization.rendering.MaterialRecord())

        # Render
        img = renderer.render_to_image()
        img_np = np.asarray(img)
        images.append(img_np)

        # image_path = os.path.join(output_path, f"frame_{b:04d}.png")
        # imageio.imwrite(image_path, img_np)

    # Save video
    video_path = os.path.join(output_path, 'video.mp4')
    imageio.mimwrite(video_path, images, fps=fps, codec="libx264", quality=8)
    print(f"Video saved to {output_path}")


def visualize_point_cloud_sequence(point_clouds: np.ndarray, colors: np.ndarray, output_path: str, delay: float = 0.1):
    """
    Visualizes a sequence of point clouds using Viser.

    https://adapt3r-robot.github.io/viser-client?playbackPath=https://adapt3r-robot.github.io/recordings/threading.viser&initialCameraPosition=0.749,-0.078,1.134&initialCameraLookAt=-0.411,-0.100,0.360&initialCameraUp=0.000,0.000,1.000

    Args:
        point_clouds: (T, N, 3) numpy array of 3D points.
        colors: (T, N, 3) numpy array of RGB colors (values in [0, 1]).
        delay: Time in seconds between frames.
    """
    assert point_clouds.shape == colors.shape
    assert point_clouds.ndim == 3 and point_clouds.shape[2] == 3

    colors = colors

    T, N, _ = point_clouds.shape

    # Start a Viser server (can be viewed in browser at localhost:8080)
    server = viser.ViserServer()

    # Add initial point cloud (as a single geometry that will be updated)
    cloud = server.scene.add_point_cloud(
        "pc", 
        points=point_clouds[0], 
        colors=colors[0],
        point_size=0.003,
        point_shape='circle'    
    )

    # Create serializer.
    serializer = server.get_scene_serializer()

    for t in range(T):
        # Update point cloud
        cloud.points = point_clouds[t]
        cloud.colors = colors[t]
        # cloud.update(points=point_clouds[t], colors=colors[t])
        # Add a frame delay.
        serializer.insert_sleep(delay)
    
    data = serializer.serialize()
    Path(os.path.join(output_path, 'recording.viser')).write_bytes(data)

def main():
    parser = ArgumentParser()
    parser.add_argument('hdf5_path')
    parser.add_argument('checkpoint_path')
    parser.add_argument('--folder')
    parser.add_argument('--demo-idx', default=0, type=int)
    parser.add_argument('--output', default='video', choices=['video', 'viser'])
    args = parser.parse_args()

    file = h5py.File(args.hdf5_path, 'r')
    obs = file['data'][f'demo_{args.demo_idx}']['obs']

    state_dict = utils.load_state(args.checkpoint_path)
    norm_stats = None
    if 'norm_stats' in state_dict:
        norm_stats = state_dict['norm_stats']
    
    print('autoloading based on saved parameters')
    policy_cfg = state_dict['config']['algo']['policy']
    abs_action = policy_cfg['abs_action']
    model: ChunkPolicy = instantiate(policy_cfg)
    model.to('cuda')
    model.eval()
    model.load_state_dict(state_dict['model'])
    model.normalizer.fit(norm_stats)

    obs_dict = {key: np.expand_dims(np.asarray(value), axis=1) for key, value in obs.items()}
    data = model._make_batch(obs_dict, 0, None)
    data = model.preprocess_input(data, train_mode=False)

    encoder: Adapt3REncoder = model.encoder
    
    pcds, colors = [], []
    alpha = 0.8 if args.output == 'video' else 0.5
    for i in trange(math.ceil(len(data['obs']['agentview_image']) / 64)):
        start = i * 64
        end = (i + 1) * 64

        pcd, color = obs_to_heatmap(data, encoder, start, end, alpha=alpha)
        pcds.append(pcd)
        colors.append(color)

    pcds = np.concatenate(pcds)
    colors = np.concatenate(colors)
    
    output_path = os.path.join(args.folder)
    os.makedirs(output_path, exist_ok=True)

    if args.output == 'video':
        make_point_cloud_video(
            points=pcds,
            colors=colors,
            camera_pose=data['obs']['agentview_extrinsic'][0,0].cpu().numpy(),
            output_path=output_path
        )
    elif args.output == 'viser':
        visualize_point_cloud_sequence(
            point_clouds=pcds, 
            colors=colors, 
            output_path=output_path, 
            delay=1. / 20.
        )


if __name__ == '__main__':
    main()