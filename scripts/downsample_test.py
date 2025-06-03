import torch
import numpy as np
import matplotlib.pyplot as plt
import dgl.geometry as dgl_geo
from imitation.algos.utils.position_encodings import NeRFSinusoidalPosEmb
import imitation.utils.point_cloud_utils as pcu

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
    
    return torch.tensor(points, dtype=torch.float32)

def main():
    # Create grid points
    points = create_grid_points()
    print(f"Original number of points: {len(points)}")
    
    # Initialize NeRF positional embedding
    hidden_dim = 32
    pos_emb = NeRFSinusoidalPosEmb(hidden_dim)
    
    # Get positional embeddings for all points
    pos_embeddings = pos_emb(points)
    
    # Run FPS using the positional embeddings
    num_points = 1000
    fps_indices = dgl_geo.farthest_point_sampler(pos_embeddings.unsqueeze(0), num_points, 0)
    fps_indices = fps_indices.squeeze(0)
    
    # Get downsampled points
    downsampled_points = points[fps_indices]
    
    # Calculate distances from origin for coloring
    distances = torch.norm(downsampled_points, dim=1)
    normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())
    
    # Create color map
    cmap = plt.get_cmap('viridis')
    colors = cmap(normalized_distances.numpy())[:, :3]
    
    # Visualize the downsampled point cloud
    pcu.show_point_cloud(downsampled_points, colors)

if __name__ == "__main__":
    main()
