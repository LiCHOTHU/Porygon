import torch
import numpy as np
import matplotlib.pyplot as plt
import dgl.geometry as dgl_geo
from imitation.algos.utils.position_encodings import NeRFSinusoidalPosEmb
import imitation.utils.point_cloud_utils as pcu
import einops

def radial_embedding(x, epsilon=1e-6, mode='log'):
    r = torch.norm(x, dim=-1, keepdim=True)  # [N, 1]
    if mode == 'linear':
        scale = r
    elif mode == 'log':
        scale = torch.log(r)
    elif mode == 'log1p':
        scale = torch.log1p(r)
    elif mode == 'inv':
        scale = 1.0 / (r + epsilon)
    elif mode == 'sqrt':
        scale = torch.sqrt(r + epsilon)
    elif mode == 'tanh':
        scale = torch.tanh(r)
    elif mode == 'exp':
        scale = torch.exp(r)
    elif mode == 'sigmoid':
        scale = torch.sigmoid(r)
    else:
        raise ValueError("Unknown mode")
    return (x / (r + epsilon)) * scale  # shape: [N, D]

def create_grid_points(resolution=20):
    """Create a grid of points in an L1 ball around the origin."""
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    z = np.linspace(-1, 1, resolution)
    xx, yy, zz = np.meshgrid(x, y, z)
    points = np.stack([xx, yy, zz], axis=-1)
    points = points.reshape(-1, 3)
    
    # Filter points to be within L1 ball
    l1_norm = np.sum(np.abs(points), axis=1)
    points = points[l1_norm <= 1]
    
    return torch.tensor(points, dtype=torch.float32, device="cuda")

def main():
    # Create grid points
    points = create_grid_points(resolution=100)
    points = points * 10
    print(f"Original number of points: {len(points)}")
    
    # Initialize NeRF positional embedding
    hidden_dim = 30
    pos_emb = NeRFSinusoidalPosEmb(hidden_dim)
    # pos_emb = radial_embedding(points)
    
    modes = ['linear', 'log', 'log1p', 'inv', 'sqrt', 'tanh', 'exp', 'sigmoid']
    for mode in modes:
        print(f"Mode: {mode}")

        # Get positional embeddings for all points
        pos_embeddings = radial_embedding(points, mode=mode).unsqueeze(0)
        
        # Run FPS using the positional embeddings
        num_points = 1000
        fps_indices = dgl_geo.farthest_point_sampler(pos_embeddings, num_points, 0)
        fps_indices = fps_indices
        
        # Get downsampled points
        downsampled_points = points[fps_indices].squeeze(0)
        
        # Calculate distances from origin for coloring
        distances = torch.norm(downsampled_points, dim=1)
        normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())
        
        # Create color map
        cmap = plt.get_cmap('viridis')
        colors = cmap(normalized_distances.cpu().numpy())[:, :3]
        
        # Visualize the downsampled point cloud
        pcu.show_point_cloud(downsampled_points, colors)

if __name__ == "__main__":
    main()
