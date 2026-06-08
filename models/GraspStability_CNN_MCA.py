"""
Dual-modal Fusion Network for Grasp Stability Prediction
Based on Cross-Modal Attention Mechanism

Author: Assistant
Date: 2025-07-07
Content: CNN + Multi-Modal Cross Attention (MCA) for visual and tactile image fusion
"""
from .build import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from .Transformer_utils import Mlp


class PositionalAttention(nn.Module):
    """Multi-head Self-Attention with positional encoding support"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define separate projection layers for Q, K, and V.
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, pos_embed=None, mask=None):
        """
        Args:
            x: [B, N, D] input features
            pos_embed: [B, N, D] or [1, N, D] positional encoding
            mask: attention mask
        Returns:
            output: [B, N, D]
        """
        B, N, C = x.shape
        
        # Correct implementation: add positional encoding to Q and K, but not V.
        if pos_embed is not None:
            q = self.q_proj(x + pos_embed)  # Q is generated from x + pos.
            k = self.k_proj(x + pos_embed)  # K is generated from x + pos.
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
        v = self.v_proj(x)  # V is generated from the original x.
        
        # Reshape for multi-head attention
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Compute attention.
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            mask_value = -torch.finfo(attn.dtype).max
            mask = (mask > 0)
            attn = attn.masked_fill(mask, mask_value)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PositionalCrossAttention(nn.Module):
    """Cross-Attention with positional encoding support"""
    
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q_feat, kv_feat, q_pos=None, kv_pos=None):
        """
        Args:
            q_feat: [B, N_q, D] query features
            kv_feat: [B, N_kv, D] key-value features  
            q_pos: [B, N_q, D] or [1, N_q, D] query positional encoding
            kv_pos: [B, N_kv, D] or [1, N_kv, D] key-value positional encoding
        Returns:
            output: [B, N_q, out_dim]
        """
        B, N_q, _ = q_feat.shape
        N_kv = kv_feat.size(1)
        C = self.out_dim
        
        # Apply positional encoding for Q and K computation
        if q_pos is not None:
            q_with_pos = q_feat + q_pos
        else:
            q_with_pos = q_feat
            
        if kv_pos is not None:
            kv_with_pos = kv_feat + kv_pos
        else:
            kv_with_pos = kv_feat

        # Compute Q from query+pos, K from key+pos, V from key (no pos)
        q = self.q_map(q_with_pos).view(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(kv_with_pos).view(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(kv_feat).view(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PositionalEncoding2D(nn.Module):
    """2D Sinusoidal Positional Encoding for image features"""
    
    def __init__(self, embed_dim, height=7, width=7, temperature=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.height = height
        self.width = width
        self.temperature = temperature
        
        # Create 2D position embedding
        pe = torch.zeros(height, width, embed_dim)
        
        y_pos = torch.arange(0, height).unsqueeze(1).repeat(1, width)
        x_pos = torch.arange(0, width).unsqueeze(0).repeat(height, 1)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * 
                           -(math.log(temperature) / embed_dim))
        
        pe[:, :, 0::2] = torch.sin(y_pos.unsqueeze(-1) * div_term[:(embed_dim//2)])
        pe[:, :, 1::2] = torch.cos(y_pos.unsqueeze(-1) * div_term[:(embed_dim//2)])
        
        if embed_dim % 2 == 1:
            pe[:, :, -1] = torch.sin(x_pos.unsqueeze(-1) * div_term[0])
        
        # Register as buffer (non-trainable)
        self.register_buffer('pe', pe.flatten(0, 1).unsqueeze(0))  # [1, H*W, D]
    
    def forward(self):
        """
        Returns:
            positional encoding: [1, H*W, D]
        """
        return self.pe


class FeatureExtractor(nn.Module):
    """ResNet-50 based feature extractor with dimension reduction"""
    
    def __init__(self, config):
        super().__init__()
        # Extract config parameters - now using hierarchical config
        embed_dim = getattr(config, 'embed_dim', 512)
        pretrained = getattr(config, 'pretrained', True)
        freeze_backbone = getattr(config, 'freeze_backbone', False)
        
        # Load ResNet-50 backbone
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove last two layers (avgpool and fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Dimension reduction: 2048 -> embed_dim
        self.dim_reduction = nn.Conv2d(2048, embed_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224]
        Returns:
            features: [B, 49, embed_dim] (flattened spatial features)
        """
        # Feature extraction
        feat = self.backbone(x)  # [B, 2048, 7, 7]
        
        # Dimension reduction
        feat = self.dim_reduction(feat)  # [B, embed_dim, 7, 7]
        
        # Flatten spatial dimensions
        B, C, H, W = feat.shape
        feat = feat.view(B, C, -1).transpose(1, 2)  # [B, H*W, embed_dim]
        
        # Layer normalization
        feat = self.norm(feat)
        
        return feat


class FusionLayer(nn.Module):
    """Single fusion layer with MSA + MCA + FFN"""
    
    def __init__(self, config):
        super().__init__()
        # Extract config parameters - now using hierarchical config  
        embed_dim = getattr(config, 'embed_dim', 512)
        num_heads = getattr(config, 'num_heads', 8)
        mlp_ratio = getattr(config, 'mlp_ratio', 4.0)
        dropout = getattr(config, 'dropout', 0.1)
        
        # Multi-head Self-Attention for each modality
        self.visual_msa = PositionalAttention(
            dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout
        )
        
        self.tactile_msa = PositionalAttention(
            dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout
        )
        
        # Cross-Modal Attention
        self.visual_to_tactile_ca = PositionalCrossAttention(
            dim=embed_dim,
            out_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout
        )
        
        self.tactile_to_visual_ca = PositionalCrossAttention(
            dim=embed_dim,
            out_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout
        )
        
        # Feed Forward Networks
        self.visual_ffn = Mlp(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            drop=dropout
        )
        
        self.tactile_ffn = Mlp(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            drop=dropout
        )
        
        # Layer normalization
        self.norm1_v = nn.LayerNorm(embed_dim)
        self.norm1_t = nn.LayerNorm(embed_dim)
        self.norm2_v = nn.LayerNorm(embed_dim)
        self.norm2_t = nn.LayerNorm(embed_dim)
        self.norm3_v = nn.LayerNorm(embed_dim)
        self.norm3_t = nn.LayerNorm(embed_dim)
    
    def forward(self, visual_feat, tactile_feat, visual_pos=None, tactile_pos=None):
        """
        Args:
            visual_feat: [B, N, D]
            tactile_feat: [B, N, D]
            visual_pos: [B, N, D] or [1, N, D] visual positional encoding
            tactile_pos: [B, N, D] or [1, N, D] tactile positional encoding
        Returns:
            enhanced_visual: [B, N, D]
            enhanced_tactile: [B, N, D]
        """
        # Self-attention for each modality with positional encoding
        visual_msa_out = self.visual_msa(self.norm1_v(visual_feat), visual_pos)
        tactile_msa_out = self.tactile_msa(self.norm1_t(tactile_feat), tactile_pos)
        
        # Residual connection
        visual_feat = visual_feat + visual_msa_out
        tactile_feat = tactile_feat + tactile_msa_out
        
        # Cross-modal attention with positional encoding
        visual_ca_out = self.visual_to_tactile_ca(
            self.norm2_v(visual_feat), 
            self.norm2_t(tactile_feat),
            visual_pos,
            tactile_pos
        )
        tactile_ca_out = self.tactile_to_visual_ca(
            self.norm2_t(tactile_feat), 
            self.norm2_v(visual_feat),
            tactile_pos,
            visual_pos
        )
        
        # Residual connection
        visual_feat = visual_feat + visual_ca_out
        tactile_feat = tactile_feat + tactile_ca_out
        
        # Feed forward networks
        visual_ffn_out = self.visual_ffn(self.norm3_v(visual_feat))
        tactile_ffn_out = self.tactile_ffn(self.norm3_t(tactile_feat))
        
        # Final residual connection
        enhanced_visual = visual_feat + visual_ffn_out
        enhanced_tactile = tactile_feat + tactile_ffn_out
        
        return enhanced_visual, enhanced_tactile


class CoAttention(nn.Module):
    """Co-attention mechanism for final fusion using self-attention on concatenated features"""
    
    def __init__(self, config):
        super().__init__()
        embed_dim = getattr(config, 'embed_dim', 512)
        num_heads = getattr(config, 'num_heads', 8)
        dropout = getattr(config, 'dropout', 0.1)
        
        # Self-attention for concatenated features
        self.self_attn = PositionalAttention(
            dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Positional encoding for concatenated features (2*49=98 tokens)
        self.concat_pos_encoding = PositionalEncoding2D(embed_dim, height=14, width=7)  # 98 = 14*7
        
    def forward(self, visual_feat, tactile_feat):
        """
        Args:
            visual_feat: [B, N, D] where N=49
            tactile_feat: [B, N, D] where N=49
        Returns:
            fused_feat: [B, D]
        """
        # Concatenate features
        concat_feat = torch.cat([visual_feat, tactile_feat], dim=1)  # [B, 2*N, D] = [B, 98, D]
        
        # Get positional encoding for concatenated features
        concat_pos = self.concat_pos_encoding()  # [1, 98, D]
        
        # Self-attention on concatenated features with positional encoding
        attended_feat = self.self_attn(self.norm(concat_feat), concat_pos)
        
        # Residual connection
        fused_feat = concat_feat + attended_feat
        
        # Global pooling to get final representation
        pooled_feat = self.global_pool(fused_feat.transpose(1, 2)).squeeze(-1)  # [B, D]
        
        return pooled_feat

class CrossModalityFusionTransformer(nn.Module):
    """Cross-modality fusion transformer with multiple layers"""
    
    def __init__(self, config):
        super().__init__()
        num_layers = getattr(config, 'num_layers', 4)
        embed_dim = getattr(config, 'embed_dim', 512)
        
        # Multiple fusion layers
        self.fusion_layers = nn.ModuleList([
            FusionLayer(config) for _ in range(num_layers)
        ])
        
        # Final co-attention mechanism
        self.co_attention = CoAttention(config)
        
        # Positional encoding for 7x7 spatial features (49 tokens)
        self.pos_encoding = PositionalEncoding2D(embed_dim, height=7, width=7)
    
    def forward(self, visual_features, tactile_features):
        """
        Args:
            visual_features: [B, N, D] where N=49
            tactile_features: [B, N, D] where N=49
        Returns:
            fused_features: [B, D]
        """
        # Get positional encodings for 7x7 spatial layout
        pos_embed = self.pos_encoding()  # [1, 49, D]
        
        # Pass through multiple fusion layers
        for layer in self.fusion_layers:
            visual_features, tactile_features = layer(
                visual_features, tactile_features, pos_embed, pos_embed
            )
        
        # Final co-attention fusion (now returns [B, D] instead of [B, 2*D])
        fused_features = self.co_attention(visual_features, tactile_features)
        
        return fused_features

@MODELS.register_module()
class GraspStability_CNNMCA(nn.Module):
    """Main network for grasp stability prediction using CNN + MCA"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Extract hierarchical configs similar to SGSNet
        visual_cfg = config.visual_extractor
        tactile_cfg = config.tactile_extractor  
        fusion_cfg = config.fusion_transformer
        classifier_cfg = config.classifier
        
        # Store important configs
        self.num_classes = getattr(classifier_cfg, 'num_classes', 2)
        self.embed_dim = getattr(fusion_cfg, 'embed_dim', 512)
        self.dropout = getattr(classifier_cfg, 'dropout', 0.1)
        
        # Feature extractors for both modalities
        self.visual_extractor = FeatureExtractor(visual_cfg)
        self.tactile_extractor = FeatureExtractor(tactile_cfg)
        
        # Cross-modality fusion transformer
        self.fusion_transformer = CrossModalityFusionTransformer(fusion_cfg)
        
        # Classification head configuration
        classifier_hidden_dim = getattr(classifier_cfg, 'hidden_dim', self.embed_dim // 2)
        
        # Classification head (now input is [B, D] instead of [B, 2*D])
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),  
            nn.Linear(classifier_hidden_dim, self.num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, visual_img, tactile_img):
        """
        Args:
            visual_img: [B, 3, 224, 224]
            tactile_img: [B, 3, 224, 224]
        Returns:
            logits: [B, num_classes]
        """
        # Extract features from both modalities
        visual_features = self.visual_extractor(visual_img)      # [B, 49, embed_dim]
        tactile_features = self.tactile_extractor(tactile_img)   # [B, 49, embed_dim]
        
        # Cross-modal fusion
        fused_features = self.fusion_transformer(visual_features, tactile_features)  # [B, embed_dim]
        
        # Classification
        logits = self.classifier(fused_features)  # [B, num_classes]
        
        return logits


# Factory function for easy model creation
def create_grasp_stability_model(config=None):
    """
    Factory function to create the grasp stability prediction model
    
    Args:
        config: Hierarchical configuration dictionary. If None, uses default config.
    
    Returns:
        model: GraspStability_CNNMCA model
    """
    if config is None:
        # Create a simple namespace-like object for default config
        class DefaultConfig:
            def __init__(self):
                template = {
                    'visual_extractor': {
                        'embed_dim': 512,
                        'pretrained': True,
                        'freeze_backbone': False,
                    },
                    'tactile_extractor': {
                        'embed_dim': 512,
                        'pretrained': True,
                        'freeze_backbone': False,
                    },
                    'fusion_transformer': {
                        'embed_dim': 512,
                        'num_heads': 8,
                        'num_layers': 4,
                        'mlp_ratio': 4.0,
                        'dropout': 0.1,
                        'pos_encoding_temp': 10000,
                    },
                    'classifier': {
                        'num_classes': 2,
                        'hidden_dim': 256,
                        'dropout': 0.1,
                    },
                }
                
                # Create namespace objects for each config section
                self.visual_extractor = type('Config', (), template['visual_extractor'])()
                self.tactile_extractor = type('Config', (), template['tactile_extractor'])()  
                self.fusion_transformer = type('Config', (), template['fusion_transformer'])()
                self.classifier = type('Config', (), template['classifier'])()
        
        model = GraspStability_CNNMCA(DefaultConfig())
    else:
        model = GraspStability_CNNMCA(config)
    
    return model


if __name__ == "__main__":
    # Example usage and testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with hierarchical config structure following SGSNet pattern
    class TestConfig:
        def __init__(self):
            # Visual extractor config
            self.visual_extractor = type('Config', (), {
                'embed_dim': 512,
                'pretrained': True,
                'freeze_backbone': False,
            })()
            
            # Tactile extractor config
            self.tactile_extractor = type('Config', (), {
                'embed_dim': 512,
                'pretrained': True,
                'freeze_backbone': False,
            })()
            
            # Fusion transformer config
            self.fusion_transformer = type('Config', (), {
                'embed_dim': 512,
                'num_heads': 8,
                'num_layers': 4,
                'mlp_ratio': 4.0,
                'dropout': 0.1,
            })()
            
            # Classifier config
            self.classifier = type('Config', (), {
                'num_classes': 2,
                'hidden_dim': 256,
                'dropout': 0.1,
            })()
    
    config = TestConfig()
    model = GraspStability_CNNMCA(config).to(device)
    
    # Test forward pass
    batch_size = 2
    visual_img = torch.randn(batch_size, 3, 224, 224).to(device)
    tactile_img = torch.randn(batch_size, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(visual_img, tactile_img)
        print(f"Model output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test with factory function
        model_factory = create_grasp_stability_model()
        output_factory = model_factory(visual_img, tactile_img)
        print(f"Factory model output shape: {output_factory.shape}")
        
        print("✅ CNN_MCA model with hierarchical config test passed!")
