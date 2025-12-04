"""
Loss functions for lip-sync model training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


class ReconstructionLoss(nn.Module):
    """L1 reconstruction loss for mouth region."""
    
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
    
    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate reconstruction loss.
        
        Args:
            generated: Generated mouth regions
            target: Ground truth mouth regions
            
        Returns:
            Loss value
        """
        return self.l1_loss(generated, target)


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features."""
    
    def __init__(self, layers: list = [3, 8, 15, 22]):
        """
        Initialize perceptual loss.
        
        Args:
            layers: VGG layer indices to use for feature extraction
        """
        super().__init__()
        
        # Load pretrained VGG
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        
        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Extract feature layers
        self.feature_extractors = nn.ModuleList()
        prev_layer = 0
        
        for layer_idx in layers:
            self.feature_extractors.append(
                nn.Sequential(*list(vgg.children())[prev_layer:layer_idx+1])
            )
            prev_layer = layer_idx + 1
        
        self.l1_loss = nn.L1Loss()
    
    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate perceptual loss.
        
        Args:
            generated: Generated images
            target: Ground truth images
            
        Returns:
            Loss value
        """
        loss = 0.0
        
        gen_input = generated
        target_input = target
        
        for extractor in self.feature_extractors:
            gen_features = extractor(gen_input)
            target_features = extractor(target_input)
            
            loss += self.l1_loss(gen_features, target_features)
            
            gen_input = gen_features
            target_input = target_features
        
        return loss


class SyncLoss(nn.Module):
    """Audio-visual synchronization loss."""
    
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()
    
    def forward(
        self,
        sync_scores: torch.Tensor,
        is_synced: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate sync loss.
        
        Args:
            sync_scores: Predicted sync scores from discriminator
            is_synced: Ground truth sync labels (1 for synced, 0 for not)
            
        Returns:
            Loss value
        """
        return self.bce_loss(sync_scores, is_synced)


class AdversarialLoss(nn.Module):
    """Adversarial loss for GAN training."""
    
    def __init__(self, loss_type: str = 'lsgan'):
        """
        Initialize adversarial loss.
        
        Args:
            loss_type: Type of loss ('vanilla', 'lsgan', 'hinge')
        """
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'vanilla':
            self.criterion = nn.BCELoss()
        elif loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif loss_type == 'hinge':
            self.criterion = None
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward_generator(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """
        Generator adversarial loss.
        
        Args:
            fake_pred: Discriminator predictions for fake samples
            
        Returns:
            Loss value
        """
        if self.loss_type == 'hinge':
            return -fake_pred.mean()
        else:
            target = torch.ones_like(fake_pred)
            return self.criterion(fake_pred, target)
    
    def forward_discriminator(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Discriminator adversarial loss.
        
        Args:
            real_pred: Discriminator predictions for real samples
            fake_pred: Discriminator predictions for fake samples
            
        Returns:
            Loss value
        """
        if self.loss_type == 'hinge':
            real_loss = F.relu(1.0 - real_pred).mean()
            fake_loss = F.relu(1.0 + fake_pred).mean()
            return real_loss + fake_loss
        else:
            real_target = torch.ones_like(real_pred)
            fake_target = torch.zeros_like(fake_pred)
            
            real_loss = self.criterion(real_pred, real_target)
            fake_loss = self.criterion(fake_pred, fake_target)
            
            return (real_loss + fake_loss) * 0.5


class CombinedLoss(nn.Module):
    """Combined loss for lip-sync model."""
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        sync_weight: float = 10.0,
        adversarial_weight: float = 0.1
    ):
        """
        Initialize combined loss.
        
        Args:
            reconstruction_weight: Weight for reconstruction loss
            perceptual_weight: Weight for perceptual loss
            sync_weight: Weight for sync loss
            adversarial_weight: Weight for adversarial loss
        """
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.sync_weight = sync_weight
        self.adversarial_weight = adversarial_weight
        
        self.reconstruction_loss = ReconstructionLoss()
        self.perceptual_loss = PerceptualLoss()
        self.sync_loss = SyncLoss()
        self.adversarial_loss = AdversarialLoss()
    
    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
        sync_scores: torch.Tensor,
        is_synced: torch.Tensor,
        fake_pred: torch.Tensor
    ) -> dict:
        """
        Calculate combined loss.
        
        Args:
            generated: Generated mouth regions
            target: Ground truth mouth regions
            sync_scores: Sync discriminator predictions
            is_synced: Ground truth sync labels
            fake_pred: Discriminator predictions for generated samples
            
        Returns:
            Dictionary with individual losses and total loss
        """
        # Calculate individual losses
        recon_loss = self.reconstruction_loss(generated, target)
        percep_loss = self.perceptual_loss(generated, target)
        sync_l = self.sync_loss(sync_scores, is_synced)
        adv_loss = self.adversarial_loss.forward_generator(fake_pred)
        
        # Weighted sum
        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.perceptual_weight * percep_loss +
            self.sync_weight * sync_l +
            self.adversarial_weight * adv_loss
        )
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'perceptual': percep_loss,
            'sync': sync_l,
            'adversarial': adv_loss
        }


class LandmarkLoss(nn.Module):
    """Loss for facial landmark consistency."""
    
    def __init__(self):
        super().__init__()
        self.l2_loss = nn.MSELoss()
    
    def forward(
        self,
        generated_landmarks: torch.Tensor,
        target_landmarks: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate landmark loss.
        
        Args:
            generated_landmarks: Landmarks from generated image
            target_landmarks: Ground truth landmarks
            
        Returns:
            Loss value
        """
        return self.l2_loss(generated_landmarks, target_landmarks)
