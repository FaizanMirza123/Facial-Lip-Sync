"""
Training script for lip-sync model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import yaml
from tqdm import tqdm
import argparse

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lip_sync_model import LipSyncModel
from models.discriminator import Discriminator, SyncDiscriminator
from training.losses import CombinedLoss, AdversarialLoss
from data.dataset import create_dataloaders
from utils.audio_processing import AudioProcessor
from utils.face_detection import FaceDetector


class LipSyncTrainer:
    """Trainer for lip-sync model."""
    
    def __init__(self, config: dict, device: str = 'cuda'):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
            device: Device to train on
        """
        self.config = config
        self.device = device
        
        # Create models
        self.generator = self._create_generator()
        self.discriminator = Discriminator(**config['model']['discriminator'])
        self.sync_discriminator = SyncDiscriminator()
        
        # Move to device
        self.generator.to(device)
        self.discriminator.to(device)
        self.sync_discriminator.to(device)
        
        # Create optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config['training']['learning_rate'],
            betas=(config['training']['beta1'], config['training']['beta2'])
        )
        
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config['training']['learning_rate'],
            betas=(config['training']['beta1'], config['training']['beta2'])
        )
        
        self.sync_d_optimizer = optim.Adam(
            self.sync_discriminator.parameters(),
            lr=config['training']['learning_rate'],
            betas=(config['training']['beta1'], config['training']['beta2'])
        )
        
        # Create loss functions
        self.combined_loss = CombinedLoss(**config['training']['loss_weights'])
        self.adversarial_loss = AdversarialLoss()
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create tensorboard writer
        self.writer = SummaryWriter(config['training']['tensorboard_dir'])
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        # Load checkpoint if resuming
        if config['training']['resume_from']:
            self.load_checkpoint(config['training']['resume_from'])
    
    def _create_generator(self) -> LipSyncModel:
        """Create generator model."""
        return LipSyncModel(
            audio_config=self.config['model']['audio_encoder'],
            face_config=self.config['model']['face_encoder'],
            generator_config=self.config['model']['generator']
        )
    
    def train_epoch(self, train_loader, epoch: int):
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        self.sync_discriminator.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            face = batch['face'].to(self.device)
            audio = batch['audio'].to(self.device)
            mouth = batch['mouth'].to(self.device)
            is_synced = batch['is_synced'].to(self.device)
            
            # Train discriminators
            self.d_optimizer.zero_grad()
            self.sync_d_optimizer.zero_grad()
            
            with torch.no_grad():
                fake_mouth = self.generator(audio, face)
            
            # Discriminator loss
            real_pred = self.discriminator(mouth)
            fake_pred = self.discriminator(fake_mouth.detach())
            d_loss = self.adversarial_loss.forward_discriminator(real_pred, fake_pred)
            
            # Sync discriminator loss
            audio_features = self.generator.audio_encoder(audio)
            real_sync = self.sync_discriminator(mouth, audio_features)
            
            # Create unsynced samples
            batch_size = face.size(0)
            perm = torch.randperm(batch_size).to(self.device)
            wrong_audio = audio[perm]
            wrong_audio_features = self.generator.audio_encoder(wrong_audio)
            fake_sync = self.sync_discriminator(mouth, wrong_audio_features)
            
            sync_d_loss = (
                self.adversarial_loss.forward_discriminator(
                    real_sync,
                    fake_sync
                )
            )
            
            # Backprop discriminators
            d_loss.backward()
            sync_d_loss.backward()
            self.d_optimizer.step()
            self.sync_d_optimizer.step()
            
            # Train generator
            self.g_optimizer.zero_grad()
            
            # Generate mouth
            fake_mouth = self.generator(audio, face)
            audio_features = self.generator.audio_encoder(audio)
            
            # Discriminator predictions
            fake_pred = self.discriminator(fake_mouth)
            sync_pred = self.sync_discriminator(fake_mouth, audio_features)
            
            # Combined loss
            losses = self.combined_loss(
                generated=fake_mouth,
                target=mouth,
                sync_scores=sync_pred,
                is_synced=is_synced,
                fake_pred=fake_pred
            )
            
            g_loss = losses['total']
            
            # Backprop generator
            g_loss.backward()
            self.g_optimizer.step()
            
            # Log losses
            if batch_idx % self.config['training']['log_frequency'] == 0:
                self.writer.add_scalar('Loss/Generator', g_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/Discriminator', d_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/SyncDiscriminator', sync_d_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/Reconstruction', losses['reconstruction'].item(), self.global_step)
                self.writer.add_scalar('Loss/Perceptual', losses['perceptual'].item(), self.global_step)
                self.writer.add_scalar('Loss/Sync', losses['sync'].item(), self.global_step)
                
                pbar.set_postfix({
                    'G': f"{g_loss.item():.4f}",
                    'D': f"{d_loss.item():.4f}",
                    'Sync': f"{sync_d_loss.item():.4f}"
                })
            
            self.global_step += 1
    
    def validate(self, val_loader):
        """Validate model."""
        self.generator.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                face = batch['face'].to(self.device)
                audio = batch['audio'].to(self.device)
                mouth = batch['mouth'].to(self.device)
                
                # Generate
                fake_mouth = self.generator(audio, face)
                
                # Calculate reconstruction loss
                loss = nn.L1Loss()(fake_mouth, mouth)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.writer.add_scalar('Loss/Validation', avg_loss, self.current_epoch)
        
        print(f"Validation Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'sync_discriminator_state_dict': self.sync_discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'sync_d_optimizer_state_dict': self.sync_d_optimizer.state_dict(),
            'config': self.config
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.sync_discriminator.load_state_dict(checkpoint['sync_discriminator_state_dict'])
        
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.sync_d_optimizer.load_state_dict(checkpoint['sync_d_optimizer_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"Checkpoint loaded: {checkpoint_path}")
    
    def train(self, train_loader, val_loader, num_epochs: int):
        """Train model."""
        best_val_loss = float('inf')
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['save_frequency'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train lip-sync model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create audio processor and face detector
    audio_processor = AudioProcessor(**config['data']['audio'])
    face_detector = FaceDetector(
        detector_type=config['inference']['face_detection']['detector'],
        device=args.device
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        config, audio_processor, face_detector
    )
    
    # Create trainer
    trainer = LipSyncTrainer(config, device=args.device)
    
    # Train
    trainer.train(train_loader, val_loader, config['training']['num_epochs'])


if __name__ == '__main__':
    main()
