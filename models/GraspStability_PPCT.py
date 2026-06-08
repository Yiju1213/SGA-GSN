"""
Spatial Grasp Stability Network (SGSNet) - PPCT Model

This module implements a point-based grasp stability prediction model using:
- Point-Point Cross-modal Transformer (PPCT) architecture
- Dual-stream processing for visual and tactile features

Key Components:
- SGSNet: Main model for grasp stability prediction
- TrainingManager: Unified loss management with structured breakdown
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import MODELS

from models.PPCT_utils import PointEncoder, point_seq_drop_shuf
from models.MultiResolutionCrossAttn import MultiResCrossAttnEntry

def compute_local_density_knn(xyz, k=16):
    """
    xyz: (B, N, 3) torch tensor on GPU
    return: (B, N, 1) local density feature
    """
    dist = torch.cdist(xyz, xyz, p=2)
    knn_dist, _ = dist.topk(k=k + 1, largest=False)
    knn_dist = knn_dist[:, :, 1:]
    local_density = knn_dist.mean(dim=-1, keepdim=True)
    return local_density

@MODELS.register_module()
class SGSNet(nn.Module): # SpatialGraspStabilityNet
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # loss manager
        self.loss_manager = LossManager(config.loss_manager)  # Initialize loss manager with config

        #
        vis_cfg = config.visual_encoder
        tac_cfg = config.tactile_encoder
        
        # Shared position encoder configuration
        self.use_shared_pos_embed = getattr(config, 'use_shared_pos_embed', False)
        if self.use_shared_pos_embed:
            # Create shared position encoder with unified output dimension
            assert vis_cfg.transformer.embed_dim == tac_cfg.transformer.embed_dim, \
                "embed_dim must be equal to share pos_embed parameters"
            shared_embed_dim = vis_cfg.transformer.embed_dim  # 
            self.shared_pos_embed = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(), 
                nn.Linear(128, shared_embed_dim)
            )
        else:
            self.shared_pos_embed = None
        
        # Point sequence dropout configuration for both visual and contact features
        self.use_vis_seq_shuf_drop = getattr(config, 'use_vis_seq_shuf_drop', True)
        self.vis_seq_drop_rate = getattr(config, 'vis_seq_drop_rate', 0.1)
        self.use_tac_seq_shuf_drop = getattr(config, 'use_tac_seq_shuf_drop', False)  # Contact dropout disabled by default
        self.tac_seq_drop_rate = getattr(config, 'tac_seq_drop_rate', 0.1)
        
        # Support interval configuration validation for visual dropout
        if isinstance(self.vis_seq_drop_rate, (list, tuple)):
            assert len(self.vis_seq_drop_rate) == 2, f"vis_seq_drop_rate interval should contain 2 values: {self.vis_seq_drop_rate}"
            min_rate, max_rate = self.vis_seq_drop_rate
            assert 0 <= min_rate <= max_rate <= 0.5, f"vis_seq_drop_rate interval should be within [0, 0.5]: {self.vis_seq_drop_rate}"
        else: # int
            assert 0 <= self.vis_seq_drop_rate <= 0.5, f"vis_seq_drop_rate {self.vis_seq_drop_rate} should be in [0, 0.5]"

        # Support interval configuration validation for contact dropout
        if isinstance(self.tac_seq_drop_rate, (list, tuple)):
            assert len(self.tac_seq_drop_rate) == 2, f"tac_seq_drop_rate interval should contain 2 values: {self.tac_seq_drop_rate}"
            min_rate, max_rate = self.tac_seq_drop_rate
            assert 0 <= min_rate <= max_rate <= 0.5, f"tac_seq_drop_rate interval should be within [0, 0.5]: {self.tac_seq_drop_rate}"
        else: # int
            assert 0 <= self.tac_seq_drop_rate <= 0.5, f"tac_seq_drop_rate {self.tac_seq_drop_rate} should be in [0, 0.5]"

        # Individual feature aggregation
        self.contact_encoder = PointEncoder(tac_cfg, shared_pos_embed=self.shared_pos_embed)
        if getattr(config, 'without_shape_completion', False): # ablation study
            self.shape_encoder = PointEncoder(vis_cfg, shared_pos_embed=self.shared_pos_embed) # only encoder, and need train
        self.use_vis_encoder = config.use_vis_encoder
        if self.use_vis_encoder:
            self.shape_encoder = PointEncoder(vis_cfg, shared_pos_embed=self.shared_pos_embed) # test

        if isinstance(config.feature_fusion.embed_dims, list):
            feature_fusion_input_dim = config.feature_fusion.embed_dims[0]
            feature_fusion_output_dim = config.feature_fusion.embed_dims[-1]
        elif isinstance(config.feature_fusion.embed_dims, int):
            feature_fusion_input_dim = config.feature_fusion.embed_dims
            feature_fusion_output_dim = config.feature_fusion.embed_dims
        else:
            raise ValueError("feature_fusion.embed_dims must be a list or an int")
        
        # mem_link preparation for cross-attention feature injection
        if vis_cfg.transformer.embed_dim == feature_fusion_input_dim: # glob + local
            self.vis_mem_link = nn.Identity() # keep same
        else:
            self.vis_mem_link = nn.Linear(vis_cfg.transformer.embed_dim, feature_fusion_input_dim)
        if tac_cfg.transformer.embed_dim == feature_fusion_input_dim:
            self.tac_mem_link = nn.Identity()
        else:
            self.tac_mem_link = nn.Linear(tac_cfg.transformer.embed_dim, feature_fusion_input_dim)

        # cross-attention feature injection
        self.feature_fusion = MultiResCrossAttnEntry(config.feature_fusion)

        # prediction head
        self.predict_head = nn.Sequential(
            nn.LayerNorm(feature_fusion_output_dim),
            nn.Dropout(config.pred_head_drop_rate),
            nn.Linear(feature_fusion_output_dim, config.pred_head_mid_feature),
            nn.GELU(),
            nn.Dropout(config.pred_head_drop_rate),
            nn.Linear(config.pred_head_mid_feature, 1)
        )

        if getattr(config, 'direct_concat', False): # ablation study
            self.predict_head_concat_ablation = nn.Sequential(
                nn.LayerNorm(feature_fusion_output_dim * 2), # 2C
                nn.Dropout(config.pred_head_drop_rate),
                nn.Linear(feature_fusion_output_dim * 2, config.pred_head_mid_feature),
                nn.GELU(),
                nn.Dropout(config.pred_head_drop_rate),
                nn.Linear(config.pred_head_mid_feature, 1)
            )

    def forward(self, vis_f:torch.Tensor, vis_coor:torch.Tensor, tac_xyzc:torch.Tensor, obj_ids:torch.Tensor=None): 
        # v_xyz for visual coordinate, t_xyzc for tactile coordinate+contact(bool)
        ### ablation study
        if getattr(self.config, 'direct_concat', False):
            return self.forward_ablation_direct_concat(vis_f, vis_coor, tac_xyzc, obj_ids)
        elif getattr(self.config, 'without_shape_completion', False): 
            return self.forward_ablation_without_shape_completion(vis_f, vis_coor, tac_xyzc, obj_ids)

        ### normal forward
        # encoder for each input
        if self.use_vis_encoder:
            if self.config.visual_encoder.input_dim == 4:
                local_density = compute_local_density_knn(vis_coor)
                vis_coor = torch.cat([vis_coor, local_density], dim=-1)
            vis_f, vis_coor = self.shape_encoder(vis_coor) # using shape encoder to encoder coor of adapointr input
        tac_f, tac_coor = self.contact_encoder(tac_xyzc) # B, center_num(default 128), transformer_embed_dim(default 384)
        
        # Apply point sequence dropout+shuffle for both visual and contact features (only during training)
        if self.use_vis_seq_shuf_drop:
            vis_f, vis_coor, vis_keep_num = point_seq_drop_shuf( # avoid full shape overfitting
                vis_f, vis_coor, 
                drop_rate=self.vis_seq_drop_rate,
                training=self.training
            )

        if self.use_tac_seq_shuf_drop:
            tac_f, tac_coor, tac_keep_num = point_seq_drop_shuf( # avoid contact overfitting
                tac_f, tac_coor, 
                drop_rate=self.tac_seq_drop_rate,
                training=self.training
            )

        # cross-attention feature injection
        vis_f = self.vis_mem_link(vis_f) 
        tac_f = self.tac_mem_link(tac_f) 
        out_f = self.feature_fusion(q=tac_f, v=vis_f, q_pos=tac_coor, v_pos=vis_coor) # B, center_num(default 128), transformer_embed_dim(default 384)
        # max-pooling operation
        out_f = out_f.max(dim=1, keepdim=False)[0]  # (B, C)

        # prediction head
        pred = self.predict_head(out_f)
        return pred
    
    def forward_ablation_direct_concat(self, vis_f, vis_coor, tac_xyzc, obj_ids:torch.Tensor=None):
        # Feature extraction remains unchanged
        if self.use_vis_encoder:
            vis_f, vis_coor = self.shape_encoder(vis_coor)
        tac_f, tac_coor = self.contact_encoder(tac_xyzc)
        
        # Apply point sequence dropout+shuffle for both visual and contact features (only during training)
        if self.use_vis_seq_shuf_drop:
            vis_f, vis_coor, vis_keep_num = point_seq_drop_shuf(
                vis_f, vis_coor, 
                drop_rate=self.vis_seq_drop_rate,
                training=self.training
            )

        if self.use_tac_seq_shuf_drop:
            tac_f, tac_coor, tac_keep_num = point_seq_drop_shuf(
                tac_f, tac_coor, 
                drop_rate=self.tac_seq_drop_rate,
                training=self.training
            )
        
        vis_f = self.vis_mem_link(vis_f) 
        tac_f = self.tac_mem_link(tac_f)
        
        # Ablation study: direct concatenation instead of feature_fusion
        # Approach 1: simple concatenation followed by max pooling
        vis_f_pooled = vis_f.max(dim=1, keepdim=False)[0]  # (B, C)
        tac_f_pooled = tac_f.max(dim=1, keepdim=False)[0]  # (B, C)
        out_f = torch.cat([vis_f_pooled, tac_f_pooled], dim=-1)  # (B, 2C) TODO: verify this direct-concat ablation path before using it publicly.

        # Need to adjust the input dimension of predict_head
        pred = self.predict_head_concat_ablation(out_f)
        return pred
    
    def forward_ablation_without_shape_completion(self, vis_f, vis_coor, tac_xyzc, obj_ids:torch.Tensor=None): # vis_f should be None
        # feature extraction: shape using ShapeEncoder, not AdaPoinTr
        tac_f, tac_coor = self.contact_encoder(tac_xyzc)
        vis_f, vis_coor = self.shape_encoder(vis_coor)  # Use ShapeEncoder

        # Apply point sequence dropout+shuffle for both visual and contact features (only during training)
        if self.use_vis_seq_shuf_drop:
            vis_f, vis_coor, vis_keep_num = point_seq_drop_shuf(
                vis_f, vis_coor, 
                drop_rate=self.vis_seq_drop_rate,
                training=self.training
            )

        if self.use_tac_seq_shuf_drop:
            tac_f, tac_coor, tac_keep_num = point_seq_drop_shuf(
                tac_f, tac_coor, 
                drop_rate=self.tac_seq_drop_rate,
                training=self.training
            )

        # cross-attention feature injection
        vis_f = self.vis_mem_link(vis_f) 
        tac_f = self.tac_mem_link(tac_f) 
        out_f = self.feature_fusion(q=tac_f, v=vis_f, q_pos=tac_coor, v_pos=vis_coor) 
        out_f = out_f.max(dim=1, keepdim=False)[0]  # (B, C)

        # prediction head
        pred = self.predict_head(out_f)
        return pred

class LossManager:
    """
    Training Manager: Unified management of grasp stability loss.
    """
    def __init__(self, config):
        self.config = config
    
    def get_loss(self, predictions, targets, epoch):
        """Alias for compute_total_loss to maintain backward compatibility"""
        return self.compute_total_loss(predictions, targets, epoch)
    
    def compute_total_loss(self, predictions, targets, epoch):
        """
        Args:
            predictions: [B, 1] Model predictions for grasp stability
            targets: [B] Ground truth grasp stability labels
            epoch: int Current epoch for weight scheduling
            
        Returns:
            total_loss: BCE loss for backpropagation
            loss_info: Dictionary with loss breakdown
        """
        assert predictions.dim() == 1, \
            f"Predictions should be [B], got {predictions.shape}"
        assert targets.dim() == 1, f"Targets should be [B], got {targets.shape}"
        
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets.float())
        total_loss = bce_loss
        loss_info = {
            'epoch': epoch,
            'bce_loss': bce_loss.item(),
            'total_loss': total_loss.item(),
        }
        return total_loss, loss_info
