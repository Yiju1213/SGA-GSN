#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import MODELS
from .PPCT_utils import DGCNN_Grouper


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points # global offset

    idx = idx + idx_base  # idx_base broadcast to (batch_size, num_points, k)， offseting to each batch in idx

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


@MODELS.register_module()
class DGCNN(nn.Module):
    def __init__(self, config, output_channels=None):
        super().__init__()
        self.config = config
        self.args = config
        self.k = getattr(config, 'k', 20)
        self.emb_dims = getattr(config, 'emb_dims', 1024)
        self.dropout = getattr(config, 'dropout', 0.5)
        self.input_dim = getattr(config, 'input_dim', 3)
        self.conv_channels = getattr(config, 'conv_channels', [64, 64, 128, 256])
        if not isinstance(self.conv_channels, (list, tuple)) or len(self.conv_channels) != 4:
            raise ValueError('conv_channels must be a list/tuple of length 4')
        if output_channels is None:
            output_channels = getattr(config, 'output_channels', None)
        if output_channels is None:
            output_channels = getattr(config, 'num_classes', 40)
        self.output_channels = output_channels

        c1, c2, c3, c4 = self.conv_channels
        self.bn1 = nn.BatchNorm2d(c1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.bn3 = nn.BatchNorm2d(c3)
        self.bn4 = nn.BatchNorm2d(c4)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(self.input_dim * 2, c1, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(c1*2, c2, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(c2*2, c3, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(c3*2, c4, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(c1 + c2 + c3 + c4, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.out_dim = self.emb_dims * 2

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        return torch.cat((x_max, x_avg), 1)

@MODELS.register_module()
class DGCNNGraspStability(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_manager = LossManager(getattr(config, "loss_manager", {}))

        self.vis_grouper_cfg = getattr(config, "VIS_GROUPER", {})
        self.tac_grouper_cfg = getattr(config, "TAC_GROUPER", {})
        self.vis_proxy_num = getattr(self.vis_grouper_cfg, "proxy_num", [256,128])
        self.tac_proxy_num = getattr(self.tac_grouper_cfg, "proxy_num", [256,128])

        self.vis_enc_cfg = getattr(config, "VIS_BACKBONE", None)
        self.tac_enc_cfg = getattr(config, "TAC_BACKBONE", None)
        if self.vis_enc_cfg is None or self.tac_enc_cfg is None:
            raise ValueError("VIS_BACKBONE and TAC_BACKBONE are required.")
        
        hidden_dim = getattr(config, "HIDDEN_DIM", 256)
        out_dim = getattr(config, "OUT_DIM", 1)
        dropout_rate = float(getattr(config, "DROPOUT", 0.0))

        self.vis_raw_dim = getattr(self.vis_grouper_cfg, "input_dim", 3)
        self.tac_raw_dim = getattr(self.tac_grouper_cfg, "input_dim", 3)
        self.vis_grouper = DGCNN_Grouper(input_dim=self.vis_raw_dim)
        self.tac_grouper = DGCNN_Grouper(input_dim=self.tac_raw_dim)

        self.vis_enc_cfg.input_dim = self.vis_grouper.num_features 
        self.tac_enc_cfg.input_dim = self.tac_grouper.num_features
        self.vis_encoder = DGCNN(self.vis_enc_cfg)
        self.tac_encoder = DGCNN(self.tac_enc_cfg)

        vis_out_dim = self.vis_encoder.out_dim
        tac_out_dim = self.tac_encoder.out_dim
        self.predict_head = nn.Sequential(
            nn.Linear(vis_out_dim + tac_out_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, out_dim),
        )

    def _to_channel_first(self, points, input_dim, name=None):
        if points is None or points.dim() != 3:
            raise ValueError(f"{name} must be a 3D tensor, got {type(points)}.")
        if points.size(1) == input_dim:
            return points
        if points.size(-1) == input_dim:
            return points.transpose(1, 2).contiguous()
        raise ValueError(f"{name} last/second dim must be {input_dim}, got {points.shape}.")

    def forward(self, vis_f, vis_coor, gs_input, obj_ids=None):
        if vis_coor is None or gs_input is None:
            raise ValueError("vis_coor and gs_input are required inputs.")

        # grouper insider will handle channel position
        vis_coor_agg, vis_f = self.vis_grouper(vis_coor, self.vis_proxy_num)
        tac_coor_agg, tac_f = self.tac_grouper(gs_input, self.tac_proxy_num)
        vis_points = self._to_channel_first(vis_f, self.vis_grouper.num_features , "vis_f")
        tac_points = self._to_channel_first(tac_f, self.tac_grouper.num_features, "tac_f")

        vis_feat = self.vis_encoder(vis_points)
        tac_feat = self.tac_encoder(tac_points)
        fused = torch.cat([vis_feat, tac_feat], dim=1)
        return self.predict_head(fused)


class LossManager:
    def __init__(self, config):
        self.config = config

    def get_loss(self, predictions, targets, epoch):
        return self.compute_total_loss(predictions, targets, epoch)

    def compute_total_loss(self, predictions, targets, epoch):
        if predictions.dim() == 2 and predictions.size(1) == 1:
            predictions = predictions.squeeze(1)
        if targets.dim() == 2 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets.float())
        total_loss = bce_loss
        loss_info = {
            "epoch": epoch,
            "bce_loss": bce_loss.item(),
            "total_loss": total_loss.item(),
        }
        return total_loss, loss_info
