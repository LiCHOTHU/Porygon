import torch
import torch.nn as nn
import einops
import numpy as np

from typing import List, Type
import math

from imitation.utils.point_cloud_utils import show_point_cloud_plt


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules



class MaxExtractor(nn.Module):
    """
    Extracts from a point cloud by passing through an MLP followed by a max pooling.
    """

    def __init__(self,
                 in_shape,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 block_channel=(64, 128, 256, 512),
                 **kwargs
                 ):
        """
        Args:
            in_shape: (int, int) or (int, int, int)
                - (n_points, in_channels) if 2D
                - (n_points, frame_stack, in_channels) if 3D
            out_channels: int
            use_layernorm: bool
            final_norm: str
        """
        super().__init__()

        if type(in_shape) == int:
            in_channels = in_shape
        else:
            in_channels = in_shape[-1]
        self.out_channels = out_channels
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, pc, mask=None):
        # assume pc has dim batch, frame_stack, n, d
        x = self.mlp(pc)
        if mask is not None:
            x.masked_fill_(~mask.unsqueeze(-1), float('-inf'))
        x = torch.max(x, 2)[0]

        x = self.final_projection(x)
        return x
    

class AttentionExtractor(nn.Module):
    """
    Extracts from a point cloud by passing through an MLP followed by a attention pooling.
    """

    def __init__(self,
                 in_shape,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 hidden_dim=256,
                 num_heads=4,
                 **kwargs
                 ):
        super().__init__()
        if type(in_shape) == int:
            in_channels = in_shape
        else:
            in_channels = in_shape[-1]
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads

        # Create learnable key for each head
        self.queries = nn.Parameter(torch.randn(num_heads, hidden_dim))
        
        # Query projection MLP
        self.K_mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Value projection MLP
        self.V_mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(hidden_dim * num_heads, out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(hidden_dim * num_heads, out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
        
    def forward(self, pc, mask=None):
        # assume pc has dim batch, frame_stack, n, d
        B, F, N, D = pc.shape
        
        # Project to query and value
        K = self.K_mlp(pc)  # [B, F, N, hidden_dim]
        V = self.V_mlp(pc)  # [B, F, N, hidden_dim]
        
        # Expand key for broadcasting
        Q = self.queries.unsqueeze(0).unsqueeze(0)  # [1, 1, num_heads, head_dim]
        
        # Compute attention using scaled_dot_product_attention
        if mask is not None:
            # Expand mask for multi-head attention
            mask = mask.unsqueeze(2)
        
        # Use scaled_dot_product_attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False
        )  # [B, F, N, num_heads, head_dim]
        
        # Combine heads
        x = einops.rearrange(attn_output, 'b f h d -> b f (h d)')  # [B, F, hidden_dim]
        
        # Final projection
        x = self.final_projection(x)  # [B, F, out_channels]
        return x

        

class PointNetEncoder(nn.Module):
    """
    Encoder for Pointcloud

    Stolen from DP3 codebase
    """

    def __init__(self,
                 in_shape,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 block_channel=(64, 128, 256, 512),
                 reduction='max',
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()

        if type(in_shape) == int:
            in_channels = in_shape
        else:
            in_channels = in_shape[-1]
        self.reduction = reduction
        self.out_channels = out_channels
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
        if reduction == 'learned':
            self.red_mlp = nn.Sequential(
                nn.Linear(in_channels, 64),
                nn.LayerNorm(64) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.LayerNorm(16) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(16, 4),
                nn.LayerNorm(4) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(4, 1)
            )
        
        elif reduction == 'attention':
            self.key = nn.Parameter(torch.randn(256))
            self.Q_mlp = nn.Sequential(
                nn.Linear(in_channels, 256),
                nn.LayerNorm(256) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(256, 256)
            )
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, pc, mask=None):
        # assume pc has dim batch, frame_stack, n, d
        x = self.mlp(pc)
        if self.reduction == 'max':
            if mask is not None:
                x.masked_fill_(~mask.unsqueeze(-1), float('-inf'))
            x = torch.max(x, 2)[0]
        elif self.reduction == 'attention':
            Q = self.Q_mlp(pc)
            logits = torch.einsum('...nd,d->...n', Q, self.key) / 16
            if mask is not None:
                logits = logits.masked_fill(~mask, float('-inf'))
            weights = torch.softmax(logits, dim=-1)
            x = torch.einsum('...n,...nd->...d', weights, x)

        x = self.final_projection(x)
        return x


class iDP3Encoder(nn.Module):
    def __init__(self, 
                 in_shape,
                 out_channels: int=1024,
                 reduction='max',
                 h_dim=256, num_layers=4,
                 **kwargs):
        super().__init__()

        if type(in_shape) == int:
            in_channels = in_shape
        else:
            in_channels = in_shape[-1]
        self.out_channels = out_channels

        self.V_model = PCConvBlock(in_channels=in_channels, 
                                   out_channels=h_dim,
                                   h_dim=h_dim,
                                   num_layers=num_layers)
        self.h_dim = h_dim

        self.reduction = reduction
        if reduction == 'attention':
            self.key = nn.Parameter(torch.randn(h_dim))
            self.Q_model = PCConvBlock(in_channels=in_channels, 
                                       out_channels=h_dim,
                                       h_dim=h_dim,
                                       num_layers=num_layers)
        self.final_projection = nn.Linear(h_dim, out_channels)

    def forward(self, xyz, pc, mask, **kwargs):
        x = self.V_model(pc)

        if self.reduction == 'max':
            if mask is not None:
                # mask = einops.repeat(mask, 'b n -> b 1 ')
                x = x - 100000 * ~mask.unsqueeze(-1)
            x = torch.max(x, 2)[0]
        elif self.reduction == 'attention':
            Q = self.Q_model(pc)
            logits = torch.einsum('bfnd,d->bfn', Q, self.key) / np.sqrt(self.h_dim)
            if mask is not None:
                logits = logits.masked_fill(~mask, float('-inf'))
            weights = torch.softmax(logits, dim=-1)
            x = torch.einsum('bfn,bfnd->bfd', weights, x)
            
        x = self.final_projection(x)

        return x


class PCConvBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels: int=1024,
                 h_dim=128, 
                 num_layers=4,
                 **kwargs):
        super().__init__()

        # if type(in_shape) == int:
        #     in_channels = in_shape
        # else:
        #     in_channels = in_shape[-1]
        self.h_dim = h_dim
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=False)

        self.conv_in = nn.Conv1d(in_channels, h_dim, kernel_size=1)
        self.layers, self.global_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.Conv1d(h_dim, h_dim, kernel_size=1))
            self.global_layers.append(nn.Conv1d(h_dim * 2, h_dim, kernel_size=1))
        self.conv_out = nn.Conv1d(h_dim * self.num_layers, out_channels, kernel_size=1)


    def forward(self, x):
        # assume x has dim batch, frame_stack, n, d
        B, F, N, D = x.shape
        x = einops.rearrange(x, 'b f n d -> (b f) d n') # [B, N, 3] --> [B, 3, N]
        y = self.act(self.conv_in(x))
        feat_list = []
        for i in range(self.num_layers):
            y = self.act(self.layers[i](y))
            y_global = y.max(-1, keepdim=True).values
            y = torch.cat([y, y_global.expand_as(y)], dim=1)
            y = self.act(self.global_layers[i](y))
            feat_list.append(y)
        x = torch.cat(feat_list, dim=1)
        x = self.conv_out(x)
        x = einops.rearrange(x, '(b f) d n -> b f n d', b=B)
        return x



class SpatialAttentionEncoder(nn.Module):
    """
    Encoder for Pointcloud

    Stolen from DP3 codebase
    """

    def __init__(self,
                 in_shape: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 block_channel=(64, 128),
                 hidden_dim=256,
                 num_spatial_features=64,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()

        num_points, in_channels = in_shape

        sizes = [in_channels] + list(block_channel) + [hidden_dim]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nn.LayerNorm(sizes[i+1]) if use_layernorm else nn.Identity(),)
            layers.append(nn.ReLU())
        layers = layers[:-2]
        # breakpoint()
        
        self.mlp = nn.Sequential(*layers)

        self.K_proj = nn.Linear(hidden_dim, hidden_dim)
        self.Q_proj = nn.Parameter(torch.empty((num_spatial_features, hidden_dim, hidden_dim)))
        torch.nn.init.kaiming_uniform_(self.Q_proj, a=math.sqrt(5))

        self.spatial_proj = nn.Linear(num_spatial_features * 3, out_features=out_channels // 2)
        self.global_proj = nn.Linear(hidden_dim, out_features=out_channels // 2)
        
       
        # if final_norm == 'layernorm':
        #     self.final_projection = nn.Sequential(
        #         nn.Linear(block_channel[-1], out_channels),
        #         nn.LayerNorm(out_channels)
        #     )
        # elif final_norm == 'none':
        #     self.final_projection = nn.Linear(block_channel[-1], out_channels)
        # else:
        #     raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, pc):
        B, F, _, _ = pc.shape

        xyz = pc[..., :3]
        x = self.mlp(pc) # B, frame_stack, n_point, hidden_dim

        K = self.K_proj(x) # B, frame_stack, n_point, hidden_dim
        Q_premax = torch.einsum("mij,bfnj->bfnmi", self.Q_proj, x) # B, frame_stack, n_point, n_spatial_feature, hidden_dim
        Q = Q_premax.max(dim=2)[0] # B, frame_stack, n_spatial_feature, hidden_dim
        attn_logits = torch.einsum("bfni,bfmi->bfmn", K, Q) # B, frame_stack, n_spatial_feature, n_point 
        attn_weights = torch.softmax(attn_logits, dim=-1) # B, frame_stack, n_spatial_feature, n_point

        weighted_xyz = attn_weights.unsqueeze(-1) * xyz.unsqueeze(2) # B, frame_stack, n_spa_feat, n_points, 3
        sum_xyz = weighted_xyz.sum(dim=3) # B, frame_stack, n_spa_feat, 3
        spatial_features = sum_xyz.view(B, F, -1) # B, frame_stack, n_spa_feat * 3
        out_spatial = self.spatial_proj(spatial_features) # B, frame_stack, out_channel / 2
        out_global = self.global_proj(x).max(dim=2)[1] # B, frame, out_channel / 2
        out = torch.cat((out_spatial, out_global), dim=-1)
        return out


    




# class DP3Encoder(nn.Module):
#     def __init__(self, 
#                  out_channel=256,
#                  pointcloud_encoder_cfg=None,
#                  use_pc_color=False,
#                  pointnet_type='pointnet',
#                  ):
#         super().__init__()
#         self.point_cloud_key = 'point_cloud'
#         self.n_output_channels = out_channel
        
#         self.use_pc_color = use_pc_color
#         self.pointnet_type = pointnet_type
#         if pointnet_type == "pointnet":
#             if use_pc_color:
#                 pointcloud_encoder_cfg.in_channels = 6
#                 self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
#             else:
#                 pointcloud_encoder_cfg.in_channels = 3
#                 self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
#         else:
#             raise NotImplementedError(f"pointnet_type: {pointnet_type}")



#     def forward(self, observations: Dict) -> torch.Tensor:
#         points = observations[self.point_cloud_key]
#         assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        
#         # points = torch.transpose(points, 1, 2)   # B * 3 * N
#         # points: B * 3 * (N + sum(Ni))
#         pn_feat = self.extractor(points)    # B * out_channel
            
#         return pn_feat


#     def output_shape(self):
#         return self.n_output_channels