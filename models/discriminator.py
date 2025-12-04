"""
Discriminator network for adversarial training.
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Discriminates between real and generated mouth regions."""
    
    def __init__(
        self,
        input_channels: int = 3,
        hidden_dims: list = [64, 128, 256, 512]
    ):
        """
        Initialize discriminator.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        in_channels = input_channels
        
        for i, out_channels in enumerate(hidden_dims):
            layers.extend([
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels) if i > 0 else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images of shape (batch, 3, H, W)
            
        Returns:
            Discrimination scores of shape (batch, 1)
        """
        features = self.conv_layers(x)
        output = self.classifier(features)
        return output.view(output.size(0), -1)


class SyncDiscriminator(nn.Module):
    """Discriminator for audio-visual synchronization."""
    
    def __init__(
        self,
        visual_channels: int = 3,
        audio_dim: int = 512
    ):
        """
        Initialize sync discriminator.
        
        Args:
            visual_channels: Number of visual input channels
            audio_dim: Dimension of audio features
        """
        super().__init__()
        
        # Visual encoder
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(visual_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Audio encoder
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(512 * 6 * 6 + 512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            visual: Visual input of shape (batch, 3, H, W)
            audio: Audio features of shape (batch, audio_dim)
            
        Returns:
            Sync scores of shape (batch, 1)
        """
        # Encode visual
        visual_features = self.visual_encoder(visual)
        visual_features = visual_features.view(visual_features.size(0), -1)
        
        # Encode audio
        audio_features = self.audio_encoder(audio)
        
        # Concatenate and classify
        combined = torch.cat([visual_features, audio_features], dim=1)
        output = self.fusion(combined)
        
        return output


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for better quality."""
    
    def __init__(
        self,
        input_channels: int = 3,
        num_scales: int = 3
    ):
        """
        Initialize multi-scale discriminator.
        
        Args:
            input_channels: Number of input channels
            num_scales: Number of scales
        """
        super().__init__()
        
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList([
            Discriminator(input_channels) for _ in range(num_scales)
        ])
        
        # Downsampling for multi-scale
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> list:
        """
        Forward pass.
        
        Args:
            x: Input images of shape (batch, 3, H, W)
            
        Returns:
            List of discrimination scores at different scales
        """
        outputs = []
        
        for i, discriminator in enumerate(self.discriminators):
            outputs.append(discriminator(x))
            if i < self.num_scales - 1:
                x = self.downsample(x)
        
        return outputs
