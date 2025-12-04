"""
Face encoder for extracting identity features.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class FaceEncoder(nn.Module):
    """Encodes face images to extract identity and expression features."""
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        output_dim: int = 512,
        freeze_backbone: bool = False
    ):
        """
        Initialize face encoder.
        
        Args:
            backbone: Backbone architecture
            pretrained: Use pretrained weights
            output_dim: Output feature dimension
            freeze_backbone: Freeze backbone weights
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.output_dim = output_dim
        
        # Load backbone
        if backbone == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = resnet50(weights=weights)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            feature_dim = 2048
        elif backbone == 'resnet34':
            from torchvision.models import resnet34, ResNet34_Weights
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = resnet34(weights=weights)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            feature_dim = 512
        elif backbone == 'mobilenet_v2':
            from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
            weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            mobilenet = mobilenet_v2(weights=weights)
            self.backbone = mobilenet.features
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True)
        )
        
        # Identity branch
        self.identity_branch = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim // 2, output_dim)
        )
        
        # Expression branch
        self.expression_branch = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Face images of shape (batch, 3, H, W)
            
        Returns:
            Face features of shape (batch, output_dim)
        """
        # Extract features with backbone
        features = self.backbone(x)
        
        # Pool
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        
        # Project
        features = self.projection(features)
        
        return features
    
    def encode_with_branches(
        self,
        x: torch.Tensor
    ) -> tuple:
        """
        Encode with separate identity and expression features.
        
        Args:
            x: Face images of shape (batch, 3, H, W)
            
        Returns:
            Tuple of (identity_features, expression_features)
        """
        # Get base features
        features = self.forward(x)
        
        # Split into identity and expression
        identity = self.identity_branch(features)
        expression = self.expression_branch(features)
        
        return identity, expression


class LightweightFaceEncoder(nn.Module):
    """Lightweight face encoder for real-time inference."""
    
    def __init__(self, output_dim: int = 512):
        """
        Initialize lightweight face encoder.
        
        Args:
            output_dim: Output feature dimension
        """
        super().__init__()
        
        # Efficient convolutional layers
        self.conv_layers = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 128 -> 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64 -> 32
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 32 -> 16
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 16 -> 8
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Face images of shape (batch, 3, H, W)
            
        Returns:
            Face features of shape (batch, output_dim)
        """
        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
