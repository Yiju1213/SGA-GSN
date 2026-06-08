"""
CNN-based Grasp Stability Prediction Network
Migrated from /the-feeling-of-success/grasp_net.py

Author: Assistant
Date: 2025-07-09
Content: CNN-based network for visual and tactile sensor fusion for grasp stability prediction
"""
from .build import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


@MODELS.register_module()
class GraspStability_CNN(nn.Module):
    """
    CNN-based Grasp Stability Prediction Network
    
    This model processes RGB camera images at two time points and tactile sensor data
    from two GelSight sensors, using ResNet-50 backbone for feature extraction.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Extract configuration parameters
        pretrained = getattr(config, 'pretrained', True)
        freeze_backbone = getattr(config, 'freeze_backbone', False)
        hidden_dim = getattr(config, 'hidden_dim', 1024)
        dropout_rate = getattr(config, 'dropout', 0.5)
        num_classes = getattr(config, 'num_classes', 1)
        self.lack_cam_bef = getattr(config, 'lack_cam_bef', False)
        
        # ResNet for RGB camera (processes images at two time points)
        self.resnet50_cam = models.resnet50(pretrained=pretrained)
        self.resnet50_cam = nn.Sequential(*list(self.resnet50_cam.children())[:-1])
        
        # ResNet for tactile sensors (two sensors share parameters)
        self.resnet50_gel = models.resnet50(pretrained=pretrained) 
        self.resnet50_gel = nn.Sequential(*list(self.resnet50_gel.children())[:-1])
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.resnet50_cam.parameters():
                param.requires_grad = False
            for param in self.resnet50_gel.parameters():
                param.requires_grad = False
        
        # Fully connected layers: 6 inputs × 2048 features = 12288
        if self.lack_cam_bef:
            # If cam_bef is not used, we have 5 inputs × 2048 features = 10240
            input_dim = 2048 * 5
        else:
            input_dim = 2048 * 6
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Store parameters for forward compatibility
        self.num_classes = num_classes
        
    def forward(self, cam_bef, cam_dur, lgel_bef, lgel_dur, rgel_bef, rgel_dur):
        """
        Forward pass of the network
        
        Args:
            cam_bef: [B, 3, 224, 224] RGB camera image before grasp
            cam_dur: [B, 3, 224, 224] RGB camera image during grasp
            lgel_bef: [B, 3, 224, 224] Left GelSight tactile image before grasp
            lgel_dur: [B, 3, 224, 224] Left GelSight tactile image during grasp
            rgel_bef: [B, 3, 224, 224] Right GelSight tactile image before grasp
            rgel_dur: [B, 3, 224, 224] Right GelSight tactile image during grasp
            
        Returns:
            output: [B, num_classes] Grasp stability prediction
        """
        # Extract RGB features
        if self.lack_cam_bef:
            cam_bef_feat = self.resnet50_cam(cam_bef).flatten(1)  # [B, 2048]
        cam_dur_feat = self.resnet50_cam(cam_dur).flatten(1)  # [B, 2048]
        
        # Compute tactile difference
        lgel_diff = lgel_dur - lgel_bef
        rgel_diff = rgel_dur - rgel_bef
        
        # Extract tactile features (shared parameters)
        lgel_feat_diff = self.resnet50_gel(lgel_diff).flatten(1)  # [B, 2048]
        rgel_feat_diff = self.resnet50_gel(rgel_diff).flatten(1)  # [B, 2048]
        
        lgel_feat = self.resnet50_gel(lgel_dur).flatten(1)  # [B, 2048]
        rgel_feat = self.resnet50_gel(rgel_dur).flatten(1)  # [B, 2048]
        
        # Feature fusion
        if not self.lack_cam_bef:
            x = torch.cat([cam_bef_feat, cam_dur_feat, 
                       lgel_feat, rgel_feat, 
                       lgel_feat_diff, rgel_feat_diff], dim=1)  # [B, 2048 * 6]
        else:
            x = torch.cat([cam_dur_feat, 
                       lgel_feat, rgel_feat, 
                       lgel_feat_diff, rgel_feat_diff], dim=1)
        
        # Fully connected layers with BatchNorm and Dropout
        x = self.fc1(x)
        
        # Handle BatchNorm for different batch sizes
        if x.size(0) == 1 and self.training:
            # For batch size 1 in training mode, use eval mode temporarily for BatchNorm
            self.bn1.eval()
            x = self.bn1(x)
            self.bn1.train()
        else:
            x = self.bn1(x)
            
        x = F.relu(x)
        x = self.dropout(x)
        
        # Final output
        x = self.fc2(x)
        
        # Return logits for BCEWithLogitsLoss (do not apply sigmoid here)
        return x


def create_grasp_stability_cnn_model(config=None):
    """
    Factory function for creating GraspStability_CNN model
    
    Args:
        config: Configuration object or None for default config
        
    Returns:
        model: GraspStability_CNN model instance
    """
    if config is None:
        # Default configuration template
        template = {
            'pretrained': True,
            'freeze_backbone': False,
            'hidden_dim': 1024,
            'dropout': 0.5,
            'num_classes': 1,
        }
        
        # Create config object from template
        config = type('Config', (), template)()
    
    model = GraspStability_CNN(config)
    return model


if __name__ == "__main__":
    # Example usage and testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with default config
    class TestConfig:
        def __init__(self):
            self.pretrained = True
            self.freeze_backbone = False
            self.hidden_dim = 1024
            self.dropout = 0.5
            self.num_classes = 1
    
    config = TestConfig()
    model = GraspStability_CNN(config).to(device)
    
    # Test forward pass
    batch_size = 2
    cam_bef = torch.randn(batch_size, 3, 224, 224).to(device)
    cam_dur = torch.randn(batch_size, 3, 224, 224).to(device)
    lgel_bef = torch.randn(batch_size, 3, 224, 224).to(device)
    lgel_dur = torch.randn(batch_size, 3, 224, 224).to(device)
    rgel_bef = torch.randn(batch_size, 3, 224, 224).to(device)
    rgel_dur = torch.randn(batch_size, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(cam_bef, cam_dur, lgel_bef, lgel_dur, rgel_bef, rgel_dur)
        print(f"Model output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test with factory function
        model_factory = create_grasp_stability_cnn_model()
        output_factory = model_factory(cam_bef, cam_dur, lgel_bef, lgel_dur, rgel_bef, rgel_dur)
        print(f"Factory model output shape: {output_factory.shape}")
        
        print("✅ GraspStability_CNN model test passed!")
