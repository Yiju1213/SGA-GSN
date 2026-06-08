import torch
import torch.nn as nn
import torch.nn.functional as F

from .AdaPoinTr import PointTransformerEncoderEntry
from .AdaPoinTr import trunc_normal_
from models.Transformer_utils import *
from models.MultiResolutionCrossAttn import MultiResCrossAttnEntry

class PointEncoder(nn.Module):
    """
    Unified point cloud encoder for both visual and tactile features.
    Replaces the previously separate ShapeEncoder and ContactEncoder classes.
    """
    def __init__(self, config, shared_pos_embed=None):
        super().__init__()
        # for dgcnn grouper fps downsampling
        self.center_num = getattr(config, 'proxy_num', [512, 128])

        self.grouper = DGCNN_Grouper(
            config.input_dim,
            k=16
        )
        
        # Use shared position encoder or create independent one
        if shared_pos_embed is not None:
            self.pos_embed = shared_pos_embed
            self.use_shared_pos_embed = True
        else:
            self.pos_embed = nn.Sequential(
                nn.Linear(3, 128),  # xyz
                nn.GELU(),
                nn.Linear(128, config.transformer.embed_dim)
            )
            self.use_shared_pos_embed = False
            
        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),
            nn.GELU(),
            nn.Linear(512, config.transformer.embed_dim)
        )
        self.encoder = PointTransformerEncoderEntry(config.transformer)
        
        # Initialize only non-shared parameters
        if not self.use_shared_pos_embed:
            self.apply(self._init_weights)
        else:
            # Initialize modules except the shared position encoder
            self.input_proj.apply(self._init_weights)
            self.encoder.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, xyz):
        bs = xyz.size(0)
        coor, f = self.grouper(xyz, self.center_num)  # b n c
        pe = self.pos_embed(coor)
        x = self.input_proj(f)  # b n encoder_config.embed_dim

        x = self.encoder(x + pe, coor)  # b n c
        return x, coor


# Legacy aliases for backward compatibility (can be removed if not needed)
class ShapeEncoder(PointEncoder):
    """Legacy alias for PointEncoder - for backward compatibility"""
    pass


class ContactEncoder(PointEncoder):
    """Legacy alias for PointEncoder - for backward compatibility"""
    pass
    
class DGCNN_Grouper(nn.Module):
    """
    Modified from AdaPoinTr, enabling tactile point cloud with gel-dim
    """
    def __init__(self, input_dim, k = 16):
        super().__init__()
        '''
        K has to be 16
        '''
        self.k = k
        # self.knn = KNN(k=k, transpose_mode=False)
        self.input_trans = nn.Conv1d(input_dim, 8, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 32),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 128),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )
        self.num_features = 128
    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            # _, idx = self.knn(coor_k, coor_q)  # bs k np
            idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous()) # B G M
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1) # twice feature length
        return feature

    def forward(self, x, num):
        '''
            INPUT:
                x : bs N input_dim
                num : list e.g.[1024, 512]
            ----------------------
            OUTPUT:

                coor bs N 3
                f    bs N C(128) 
        '''
        x = x.transpose(-1, -2).contiguous() # bs, input_dim, N

        coor = x[:, :3, :] # bs, 3, N
        f = self.input_trans(x) # feature input_dim->8

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, num[0])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, num[1])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        coor = coor.transpose(-1, -2).contiguous()
        f = f.transpose(-1, -2).contiguous()

        return coor, f
    

def point_seq_drop_shuf(point_f, point_coor, drop_rate=0.1, training=True):
    """
    Random dropout + shuffle for point feature sequences, ensuring consistent length through fixed retention count
    Supports drop_rate interval random sampling to enhance training randomness
    
    Args:
        point_f: [B, N, D] - point feature sequence
        point_coor: [B, N, 3] - point coordinate sequence
        drop_rate: dropout ratio, supports the following formats:
                  - float: fixed dropout rate (e.g. 0.1)
                  - tuple/list: dropout rate interval (e.g. [0.05, 0.15] or (0.1, 0.3))
        training: whether in training mode
    
    Returns:
        point_f: [B, N_keep, D] - point features after dropout+shuffle, all batches have same length
        point_coor: [B, N_keep, 3] - corresponding point coordinates, all batches have same length
        keep_num: int - number of points retained (same for all batches)
    """
    # Apply dropout only during training, return original data during testing
    if not training:
        B, N = point_f.shape[:2]
        return point_f, point_coor, N
    
    B, N, D = point_f.shape
    
    # Handle drop_rate interval: randomly select a value from the interval
    if isinstance(drop_rate, (list, tuple)) and len(drop_rate) == 2:
        min_rate, max_rate = drop_rate
        actual_drop_rate = torch.rand(1).item() * (max_rate - min_rate) + min_rate
    else:
        actual_drop_rate = drop_rate if isinstance(drop_rate, (int, float)) else 0.1
    
    # Return directly if actual dropout rate is 0 or negative
    if actual_drop_rate <= 0:
        return point_f, point_coor, N
    
    # Calculate fixed number of points to keep (same for all batches)
    keep_num = max(1, int(N * (1 - actual_drop_rate)))  # Keep at least 1 point
    
    if keep_num >= N:
        # Return directly if retention count equals original count
        return point_f, point_coor, N
    
    # Randomly select keep_num points for each batch and shuffle (fully parallelized)
    # Generate random indices for all batches
    batch_indices = torch.stack([
        torch.randperm(N, device=point_f.device)[:keep_num] 
        for _ in range(B)
    ], dim=0)  # [B, keep_num]
    
    # Create batch indices for advanced indexing
    batch_idx = torch.arange(B, device=point_f.device).unsqueeze(1).expand(B, keep_num)  # [B, keep_num]
    
    # Extract selected points for all batches in parallel
    result_point_f = point_f[batch_idx, batch_indices]  # [B, keep_num, D]
    result_point_coor = point_coor[batch_idx, batch_indices]  # [B, keep_num, 3]
    
    return result_point_f, result_point_coor, keep_num

class PointMLP(nn.Module):
    def __init__(self, input_dim, compression_ratio=4, output_dim=None, dropout_rate=0.1):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
            
        # Bottleneck design: compress to 1/4 dimension
        bottleneck_dim = max(32, input_dim // compression_ratio)  # At least 32 dimensions
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),     # Significant compression
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(bottleneck_dim, bottleneck_dim), # Transform in compressed space
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(bottleneck_dim, output_dim),     # Restore dimension
            nn.LayerNorm(output_dim)
        )
        
        self.use_residual = (input_dim == output_dim)
        
    def forward(self, x):
        out = self.mlp(x)
        if self.use_residual:
            out = out + x
        return out