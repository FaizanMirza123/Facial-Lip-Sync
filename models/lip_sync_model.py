"""
Lip-sync model architecture.
Generates lip movements synchronized with audio features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AudioEncoder(nn.Module):
    """Encodes audio mel-spectrograms into feature vectors."""
    
    def __init__(
        self,
        input_channels: int = 1,
        hidden_dims: list = [64, 128, 256, 512],
        mel_bins: int = 80,
        window_size: int = 16
    ):
        """
        Initialize audio encoder.
        
        Args:
            input_channels: Number of input channels
            hidden_dims: List of hidden layer dimensions
            mel_bins: Number of mel frequency bins
            window_size: Number of frames in audio window
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_dims = hidden_dims
        
        # Convolutional layers
        layers = []
        in_channels = input_channels
        
        for out_channels in hidden_dims:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate flattened size
        self.flat_size = hidden_dims[-1] * (mel_bins // (2 ** len(hidden_dims))) * \
                        (window_size // (2 ** len(hidden_dims)))
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Audio features of shape (batch, mel_bins, window_size)
            
        Returns:
            Audio embeddings of shape (batch, 512)
        """
        # Add channel dimension
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Convolutional encoding
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc(x)
        
        return x


class FaceEncoder(nn.Module):
    """Encodes face images to extract identity features."""
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        output_dim: int = 512
    ):
        """
        Initialize face encoder.
        
        Args:
            backbone: Backbone architecture name
            pretrained: Whether to use pretrained weights
            output_dim: Output feature dimension
        """
        super().__init__()
        
        # Load backbone
        if backbone == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = resnet50(weights=weights)
            self.features = nn.Sequential(*list(resnet.children())[:-1])
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Face images of shape (batch, 3, H, W)
            
        Returns:
            Face embeddings of shape (batch, output_dim)
        """
        # Extract features
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        # Project
        x = self.projection(x)
        
        return x


class LipSyncGenerator(nn.Module):
    """Generates lip-synced face images from audio and face features."""
    
    def __init__(
        self,
        audio_dim: int = 512,
        face_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 6,
        attention_heads: int = 8
    ):
        """
        Initialize lip-sync generator.
        
        Args:
            audio_dim: Dimension of audio features
            face_dim: Dimension of face features
            hidden_dim: Hidden layer dimension
            num_layers: Number of transformer layers
            attention_heads: Number of attention heads
        """
        super().__init__()
        
        # Feature fusion
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.face_proj = nn.Linear(face_dim, hidden_dim)
        
        # Transformer for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder - generates mouth region
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256 * 6 * 6),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling convolutions
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(
        self,
        audio_features: torch.Tensor,
        face_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            audio_features: Audio features of shape (batch, audio_dim)
            face_features: Face features of shape (batch, face_dim)
            
        Returns:
            Generated mouth region of shape (batch, 3, 96, 96)
        """
        # Project features
        audio_emb = self.audio_proj(audio_features)
        face_emb = self.face_proj(face_features)
        
        # Combine features
        combined = audio_emb + face_emb
        
        # Add sequence dimension for transformer
        combined = combined.unsqueeze(1)
        
        # Temporal modeling
        features = self.transformer(combined)
        
        # Remove sequence dimension
        features = features.squeeze(1)
        
        # Decode
        x = self.decoder(features)
        x = x.view(x.size(0), 256, 6, 6)
        
        # Upsample to mouth region size
        mouth_region = self.upsample(x)
        
        return mouth_region


class LipSyncModel(nn.Module):
    """Complete lip-sync model combining all components."""
    
    def __init__(
        self,
        audio_config: dict,
        face_config: dict,
        generator_config: dict
    ):
        """
        Initialize lip-sync model.
        
        Args:
            audio_config: Configuration for audio encoder
            face_config: Configuration for face encoder
            generator_config: Configuration for lip-sync generator
        """
        super().__init__()
        
        self.audio_encoder = AudioEncoder(**audio_config)
        self.face_encoder = FaceEncoder(**face_config)
        self.generator = LipSyncGenerator(**generator_config)
    
    def forward(
        self,
        audio: torch.Tensor,
        face: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            audio: Audio features of shape (batch, mel_bins, window_size)
            face: Face images of shape (batch, 3, H, W)
            
        Returns:
            Generated mouth region of shape (batch, 3, 96, 96)
        """
        # Encode inputs
        audio_features = self.audio_encoder(audio)
        face_features = self.face_encoder(face)
        
        # Generate lip-synced mouth
        mouth_region = self.generator(audio_features, face_features)
        
        return mouth_region
    
    def inference(
        self,
        audio: torch.Tensor,
        face: torch.Tensor
    ) -> torch.Tensor:
        """
        Inference mode forward pass.
        
        Args:
            audio: Audio features
            face: Face images
            
        Returns:
            Generated mouth region
        """
        with torch.no_grad():
            return self.forward(audio, face)


def create_lip_sync_model(config: dict) -> LipSyncModel:
    """
    Create lip-sync model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LipSyncModel instance
    """
    audio_config = config['model']['audio_encoder']
    face_config = config['model']['face_encoder']
    generator_config = config['model']['generator']
    
    model = LipSyncModel(audio_config, face_config, generator_config)
    
    return model


def load_checkpoint(
    model: LipSyncModel,
    checkpoint_path: str,
    device: str = 'cuda'
) -> LipSyncModel:
    """
    Load model from checkpoint.
    
    Args:
        model: LipSyncModel instance
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model
