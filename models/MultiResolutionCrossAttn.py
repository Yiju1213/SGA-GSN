from numpy import isin
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath, trunc_normal_
from models.Transformer_utils import *
from pointnet2_ops import pointnet2_utils
from models.AdaPoinTr import CrossAttnBlockApi
import torch.nn.functional as F

class GroupingBasedDownsampler(nn.Module):
    """
    Grouping-based feature aggregation downsampler with EdgeConv/DynamicGraphAttention-style local structure modeling.
    """
    def __init__(self, input_dim, output_dim, target_points=None, downsample_ratio=None, k_neighbors=16, 
                 graph_aggregation_method='max', downsample_method='fps'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Support two modes: target_points first, or downsample_ratio for backward compatibility.
        assert target_points is not None or downsample_ratio is not None, \
            "Must provide either target_points or downsample_ratio"
        assert not (target_points is not None and downsample_ratio is not None), \
            "Cannot provide both target_points and downsample_ratio"
        
        self.target_points = target_points
        self.downsample_ratio = downsample_ratio
        self.k_neighbors = k_neighbors
        self.graph_aggregation_method = graph_aggregation_method
        self.downsample_method = downsample_method
        # EdgeConv-style feature aggregation network.
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        # self.pos_encoder = nn.Sequential(
        #     nn.Linear(3, output_dim // 4),
        #     nn.GELU(),
        #     nn.Linear(output_dim // 4, output_dim)
        # )
    def forward(self, features, positions):
        """
        Args:
            features: [B, N, input_dim]
            positions: [B, N, 3]
        Returns:
            aggregated_features: [B, N_new, output_dim]  
            sampled_positions: [B, N_new, 3]
        """
        B, N, input_C = features.shape
        assert input_C == self.input_dim, f"Input dim {input_C} != expected {self.input_dim}"
        
        # Compute target point count.
        if self.target_points is not None:
            assert self.target_points <= N, f"target_points {self.target_points} must be <= input points {N}"  # The caller should already have validated this.
            assert self.target_points > 0, f"target_points {self.target_points} must be > 0"
            target_num = self.target_points
        else:
            target_num = max(int(N * self.downsample_ratio), 32)
        
        # Early exit when the input point count already matches the target.
        # Handle stage-1 downsampler enabled with target_points == N.
        if N == target_num:
            # Adapt output dimensions if needed.
            if self.input_dim != self.output_dim:
                # Adjust dimensions with an MLP.
                adapted_features = self.edge_mlp(torch.cat([features, features], dim=-1))  # [B, N, output_dim]
            else:
                adapted_features = features
            return adapted_features, positions
        
        # Step 1: select sampled center points.
        if self.downsample_method == 'fps':
            center_idx = pointnet2_utils.furthest_point_sample(
                positions.contiguous(), target_num
            )
        else:  # random sampling
            center_idx = torch.randperm(N, device=features.device)[:target_num]
            center_idx = center_idx.unsqueeze(0).expand(B, -1)
        
        # Step 2: gather sampled centers.
        center_positions = pointnet2_utils.gather_operation(
            positions.transpose(1, 2).contiguous(), center_idx
        ).transpose(1, 2).contiguous()  # [B, target_num, 3]
        
        # Step 3: collect k neighbors for each center.
        neighbor_idx = knn_point(self.k_neighbors, positions, center_positions)  # [B, target_num, k]
        
        # Step 4: aggregate neighbor features.
        neighbor_features = index_points(features, neighbor_idx)  # [B, target_num, k, input_dim]
        center_feat = index_points(features, center_idx)  # [B, target_num, input_dim]
        center_feat = center_feat.unsqueeze(2).expand(-1, -1, self.k_neighbors, -1)  # [B, target_num, k, input_dim]
        # EdgeConv style: concatenate (neighbor - center, center).
        edge_feat = torch.cat([neighbor_features - center_feat, center_feat], dim=-1)  # [B, target_num, k, 2*input_dim]
        edge_feat = self.edge_mlp(edge_feat)  # [B, target_num, k, output_dim]
        # # Positional encoding.
        # neighbor_positions = index_points(positions, neighbor_idx)  # [B, target_num, k, 3]
        
        # # Step 5: relative positional encoding adapted to output_dim.
        # relative_pos = neighbor_positions - center_positions.unsqueeze(-2)  # [B, target_num, k, 3]
        # pos_encoding = self.pos_encoder(relative_pos)  # [B, target_num, k, output_dim]
        # edge_feat = edge_feat + pos_encoding
        # Aggregate.
        if self.graph_aggregation_method == 'max':
            aggregated_features = edge_feat.max(dim=-2)[0]  # [B, target_num, output_dim]
        elif self.graph_aggregation_method == 'mean':
            aggregated_features = edge_feat.mean(dim=-2)
        elif self.graph_aggregation_method == 'attention':
            # Attention-weighted aggregation.
            attention_weights = torch.softmax(
                torch.sum(edge_feat * edge_feat, dim=-1, keepdim=True), 
                dim=-2
            )  # [B, target_num, k, 1]
            aggregated_features = torch.sum(edge_feat * attention_weights, dim=-2)
        assert aggregated_features.shape[-1] == self.output_dim, f"Output dim {aggregated_features.shape[-1]} != expected {self.output_dim}"
        return aggregated_features, center_positions

class MultiScaleCrossAttnBlock(nn.Module):
    def __init__(
        self, 
        input_dim,  # Current layer input dimension for downsampling.
        embed_dim,  # Current layer dimension for cross-attention.
        num_heads, 
        mlp_ratio=4., 
        qkv_bias=False, 
        drop=0., 
        attn_drop=0.,
        init_values=None, 
        drop_path=0., 
        act_layer=nn.GELU, 
        norm_layer=nn.LayerNorm,
        self_attn_block_style='attn-deform', 
        self_attn_combine_style='concat',
        cross_attn_block_style='attn-deform', 
        cross_attn_combine_style='concat',
        k=10, 
        n_group=2,
        # Downsampling parameters.
        downsample_v=True, 
        downsample_q=False,  # Whether to downsample query points
        target_points=None,
        downsample_ratio=None,
        downsample_k=16, 
        graph_aggregation_method='max'
    ):
        super().__init__()
        
        # 
        self.cross_attn_block = CrossAttnBlockApi(
            dim=embed_dim,  # 
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            self_attn_block_style=self_attn_block_style,
            self_attn_combine_style=self_attn_combine_style,
            cross_attn_block_style=cross_attn_block_style,
            cross_attn_combine_style=cross_attn_combine_style,
            k=k, 
            n_group=n_group
        )
        # parameter checks
        assert input_dim <= embed_dim, \
            f"input_dim {input_dim} must be <= embed_dim {embed_dim} for feature expansion or identity"
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Downsampler supporting PVT-style downsampling plus dimension expansion, or downsampling only.
        # No downsampler is needed if consecutive point counts are the same.
        self.target_points = target_points  
        self.downsample_v = downsample_v
        self.downsample_q = downsample_q
        if downsample_v:
            self.v_downsampler = GroupingBasedDownsampler(
                input_dim=input_dim,       # input_dim of current block input
                output_dim=embed_dim,      # embed_dim for current cross attention
                target_points=target_points,
                downsample_ratio=downsample_ratio, 
                k_neighbors=downsample_k, 
                graph_aggregation_method=graph_aggregation_method
            )
        else:
            # No downsampling, but dimension expansion may be needed.
            if input_dim != embed_dim:
                # Add a simple linear layer for dimension projection.
                self.v_dim_expansion = nn.Linear(input_dim, embed_dim)
            # No extra processing is needed when dimensions match.

        # Q downsampler (uses same target_points as V)
        if downsample_q:
            self.q_downsampler = GroupingBasedDownsampler(
                input_dim=input_dim,       # input_dim of current block input
                output_dim=embed_dim,      # embed_dim for current cross attention
                target_points=target_points,
                downsample_ratio=downsample_ratio, 
                k_neighbors=downsample_k, 
                graph_aggregation_method=graph_aggregation_method
            )
        else:
            if embed_dim != input_dim:
                self.q_dim_expansion = nn.Linear(input_dim, embed_dim)
        # No extra parameters are needed when embed_dim == input_dim.
            
    def forward(self, q, v, q_pos, v_pos):
        """
        Args:
            q: [B, Nq, input_dim] - contact features
            v: [B, Nv, input_dim] - shape features
            q_pos: [B, Nq, 3] - contact positions
            v_pos: [B, Nv, 3] - shape positions
        Returns:
            q: [B, Nq_new, embed_dim] - contact features for CrossAttn, possibly downsampled and projected
            v: [B, Nv_new, embed_dim] - shape features for CrossAttn, possibly downsampled and projected
            q_pos: [B, Nq_new, 3] - contact positions after optional downsampling
            v_pos: [B, Nv_new, 3] - shape positions after downsampling
        """
        def process_branch(features, positions, use_downsampler, downsampler, dim_expansion):
            """Shared branch-processing logic."""
            if use_downsampler:
                # Downsample and optionally project dimensions.
                features_out, positions_out = downsampler(features, positions)
            else:
                # Only project dimensions if needed.
                if self.embed_dim > self.input_dim:
                    features_out = dim_expansion(features)
                    # Residual connection: zero-pad original features to the target dimension.
                    residual = F.pad(features, (0, self.embed_dim - self.input_dim), 'constant', 0)
                    features_out = features_out + residual
                else:
                    features_out = features
                positions_out = positions
            return features_out, positions_out
        
        # Step 1: process Q and V branches consistently.
        v_ds, v_pos_ds = process_branch(
            v, v_pos, self.downsample_v, 
            getattr(self, 'v_downsampler', None), 
            getattr(self, 'v_dim_expansion', None)
        )
        
        q_ds, q_pos_ds = process_branch(
            q, q_pos, self.downsample_q,
            getattr(self, 'q_downsampler', None),
            getattr(self, 'q_dim_expansion', None)
        )
        
        # Step 2: cross-attention.
        q_attn = self.cross_attn_block(q_ds, v_ds, q_pos_ds, v_pos_ds)
        
        # Step 3: residual connection. q_ds and q_attn have the same sequence length after process_branch.
        q = q_attn + q_ds
        
        return q, v_ds, q_pos_ds, v_pos_ds

class MultiResCrossAttn(nn.Module):
    def __init__(
        self, 
        input_dim,  # need as embed_dim[-1]
        # transformer parameters:
        embed_dims=[384, 480, 576, 576],  # dim of current layer SA & CA features
        depth=4, 
        num_heads=6, 
        mlp_ratio=4.,
        qkv_bias=True,
        init_values=None,
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0.,
        norm_layer=None, 
        act_layer=None,
        # kv downsample parameters:
        enable_downsample_q=False,  # Whether to downsample query points
        downsample_points=None,  # [N1, N2, N3, ...], can only be used if downsample_ratios is None
        downsample_ratios=None,  # [r1, r2, r3, ...], can only be used if downsample_points is None
        downsample_k=8, 
        graph_aggregation_method='max',
        # transformer block styles:
        self_attn_block_style_list=None,
        self_attn_combine_style='concat',
        cross_attn_block_style_list=None,
        cross_attn_combine_style='concat',
        k=10, 
        n_group=2,
        # Multi-Scale Feature Aggregation(only calculate if all embed_dims are the same):
        multi_scale_aggregation=False,
        multi_scale_aggregation_method='concat',  # 'concat', 'weighted_sum', 'attention'
        multi_scale_aggregation_layers=[1, 3]
    ):
        super().__init__()
        # ======= robust parameter checking =======
        # embed_dims process, enabling single and multiple
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * depth
        elif isinstance(embed_dims, list):
            assert len(embed_dims) == depth, f"embed_dims length {len(embed_dims)} != depth {depth}"
        # head process, enabling single and multiple
        if isinstance(num_heads, int):
            num_heads = [num_heads] * depth
        elif isinstance(num_heads, list):
            assert len(num_heads) == depth, f"num_heads length {len(num_heads)} != depth {depth}"
        # downsample_points/downsample_ratios process, enabling single and multiple
        assert downsample_points is not None or downsample_ratios is not None, \
            "Must provide either downsample_points or downsample_ratios"
        assert not (downsample_points is not None and downsample_ratios is not None), \
            "Cannot provide both downsample_points and downsample_ratios"
        
        if downsample_points is not None:
            if isinstance(downsample_points, int):
                downsample_points = [downsample_points] * depth
            elif isinstance(downsample_points, list):
                assert len(downsample_points) == depth, \
                    f"downsample_points length {len(downsample_points)} != depth {depth}"
                # Check that all point counts are positive integers.
                assert all(isinstance(p, int) and p > 0 for p in downsample_points), \
                    f"All downsample_points must be positive integers, got {downsample_points}"
            self.downsample_points = downsample_points
            self.downsample_ratios = None
        else:
            assert isinstance(downsample_ratios, list), \
                "downsample_ratios must be a list of floats"
            assert all(isinstance(r, float) and r > 0 and r <= 1 for r in downsample_ratios), \
                f"All downsample_ratios must be between 0 and 1, got {downsample_ratios}"
            assert len(downsample_ratios) == depth, \
                f"downsample_ratios length {len(downsample_ratios)} != depth {depth}"
            self.downsample_points = None
            self.downsample_ratios = downsample_ratios
            
        # self_attn_block_style_list and cross_attn_block_style_list process
        assert self_attn_block_style_list is not None and cross_attn_block_style_list is not None, \
            "Must provide both self_attn_block_style_list and cross_attn_block_style_list"
        if isinstance(self_attn_block_style_list, str):
            self_attn_block_style_list = [self_attn_block_style_list] * depth
        if isinstance(cross_attn_block_style_list, str):
            cross_attn_block_style_list = [cross_attn_block_style_list] * depth
        assert isinstance(self_attn_block_style_list, list) and isinstance(cross_attn_block_style_list, list), \
            "self_attn_block_style_list and cross_attn_block_style_list must be lists"
        assert len(self_attn_block_style_list) == depth, \
            f"self_attn_block_style_list length {len(self_attn_block_style_list)} != depth {depth}"
        assert len(cross_attn_block_style_list) == depth, \
            f"cross_attn_block_style_list length {len(cross_attn_block_style_list)} != depth {depth}"
        
        # connection checking
        self.multi_scale_aggregation = multi_scale_aggregation
        if multi_scale_aggregation is True:
            # check if all embed_dims are the same
            assert len(set(embed_dims)) == 1, "All embed_dims must be the same for skip connection"
            assert multi_scale_aggregation_method in ['concat', 'weighted_sum', 'attention'], \
                f"Invalid connection method: {multi_scale_aggregation_method}"
            assert isinstance(multi_scale_aggregation_layers, list) and all(isinstance(x, int) for x in multi_scale_aggregation_layers), \
                "connection_layers must be a list of integers"
            assert all(0 <= x < depth for x in multi_scale_aggregation_layers), \
                f"connection_layers must be within range [0, {depth - 1}]"
            
            self.multi_scale_aggregation_layers = multi_scale_aggregation_layers
            self.multi_scale_aggregation_method = multi_scale_aggregation_method

        # ====== initialize network submodule ======
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.embed_dims = embed_dims
        self.depth = depth
        # Drop path rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # 
        self.blocks = nn.ModuleList()
        for i in range(depth):
            # problem: stage 1 downsampler will always be True, but target_points can be same
            enable_downsample_v = \
                (self.downsample_points[i] < self.downsample_points[i - 1] if i > 0 else True) \
                if self.downsample_points is not None \
                else self.downsample_ratios[i] < 1.0
            self.blocks.append(MultiScaleCrossAttnBlock(
                input_dim=input_dim if i == 0 else embed_dims[i - 1],  # Input dimension before scale transition.
                embed_dim=embed_dims[i],  # Dimension passed into CrossAttn after scale transition.
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                self_attn_block_style=self_attn_block_style_list[i],
                self_attn_combine_style=self_attn_combine_style,
                cross_attn_block_style=cross_attn_block_style_list[i],
                cross_attn_combine_style=cross_attn_combine_style,
                k=k, 
                n_group=n_group,
                downsample_v=enable_downsample_v,
                downsample_q=enable_downsample_v and enable_downsample_q,
                target_points=self.downsample_points[i] if self.downsample_points is not None else None,
                downsample_ratio=self.downsample_ratios[i] if self.downsample_ratios is not None else None,
                downsample_k=downsample_k,
                graph_aggregation_method=graph_aggregation_method
            ))

        # Skip-connection fusion module.
        if multi_scale_aggregation is True:
            if multi_scale_aggregation_method == 'concat':
                self.connection_proj = nn.Sequential(
                    nn.Linear(embed_dims[-1] * len(multi_scale_aggregation_layers), embed_dims[-1] * 2),
                    nn.LayerNorm(embed_dims[-1] * 2),
                    nn.GELU(),
                    nn.Dropout(drop_rate),
                    nn.Linear(embed_dims[-1] * 2, embed_dims[-1])
                )
            elif multi_scale_aggregation_method == 'weighted_sum':
                self.layer_weights = nn.Parameter(torch.ones(len(multi_scale_aggregation_layers)) / len(multi_scale_aggregation_layers))
            elif multi_scale_aggregation_method == 'attention':
                self.attention_weights = nn.MultiheadAttention(
                    embed_dims[-1], num_heads=8, dropout=drop_rate
                )

        # Final normalization.
        self.norm = norm_layer(embed_dims[-1])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, q, v, q_pos, v_pos):
        """
        Args:
            q: [B, Nq, embed_dims[0]] - contact features
            v: [B, Nv, embed_dims[0]] - shape features
            q_pos: [B, Nq, 3] - contact positions
            v_pos: [B, Nv, 3] - shape positions
        Returns:
            q: [B, Nq_final, embed_dims[-1]] - fused contact features, possibly downsampled
        """
        B, Nv, _ = v.shape
        
        # Runtime check: when using downsample_points, ensure targets do not exceed input counts.
        if self.downsample_points is not None:
            for i, target_pts in enumerate(self.downsample_points):
                if target_pts is not None and target_pts > 0:
                    # Compute the input point count for layer i, accounting for previous downsampling.
                    current_v_points = Nv
                    for j in range(i):
                        if (self.downsample_points[j] > 0):
                            current_v_points = min(self.downsample_points[j], current_v_points)
                    
                    if target_pts > current_v_points:
                        print(f"Warning: Layer {i} downsample_points={target_pts} > current_v_points={current_v_points}, "
                              f"will be clipped to {current_v_points}")
        
        layer_outputs = []

        for i, block in enumerate(self.blocks):
            # 
            q, v, q_pos, v_pos = block(q, v, q_pos, v_pos)
            if self.multi_scale_aggregation is True and i in self.multi_scale_aggregation_layers:
                # Save current-layer output for later fusion.
                layer_outputs.append(q)
        
        if self.multi_scale_aggregation is True:
            # Fuse outputs from all selected layers.
            if self.multi_scale_aggregation_method == 'concat':
                # Directly concatenate outputs with different dimensions: [B, Nq, sum(embed_dims[connection_layers])].
                cat_q = torch.cat(layer_outputs, dim=-1)
                q = self.connection_proj(cat_q)
                
            elif self.multi_scale_aggregation_method == 'weighted_sum':
                weights = torch.softmax(self.layer_weights, dim=0)
                q = sum(w * layer_out for w, layer_out in zip(weights, layer_outputs))
                
            elif self.multi_scale_aggregation_method == 'attention':
                # Attention fusion: align dimensions first.
                B, Nq, dim = layer_outputs[0].shape
                stacked_outputs = torch.stack(layer_outputs, dim=0)  # [len(connection_layers), B, Nq, final_dim]
                stacked_outputs = stacked_outputs.view(len(self.multi_scale_aggregation_layers), B*Nq, dim)

                # Use the last layer as query and all selected layers as key/value.
                query = stacked_outputs[-1:].transpose(0, 1)  # [B*Nq, 1, final_dim]
                key_value = stacked_outputs.transpose(0, 1)    # [B*Nq, len(connection_layers), final_dim]

                attended_output, _ = self.attention_weights(query, key_value, key_value)
                q = attended_output.squeeze(1).view(B, Nq, dim)  # [B, Nq, final_dim]
        
        # Final normalization.
        q = self.norm(q)
        return q
    
class MultiResCrossAttnEntry(MultiResCrossAttn):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))