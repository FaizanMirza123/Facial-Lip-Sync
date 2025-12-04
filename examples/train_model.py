"""
Example script for training the lip-sync model.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train import LipSyncTrainer
from data.dataset import create_dataloaders
from utils.audio_processing import AudioProcessor
from utils.face_detection import FaceDetector


def main():
    parser = argparse.ArgumentParser(description='Train lip-sync model')
    parser.add_argument(
        '--config',
        type=str,
        default='../configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing training videos'
    )
    parser.add_argument(
        '--val_dir',
        type=str,
        help='Directory containing validation videos (optional)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to train on'
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update data directories
    config['data']['train_data_dir'] = args.data_dir
    if args.val_dir:
        config['data']['val_data_dir'] = args.val_dir
    else:
        config['data']['val_data_dir'] = args.data_dir  # Use same as train
    
    # Update resume path if provided
    if args.resume:
        config['training']['resume_from'] = args.resume
    
    print("\n" + "="*50)
    print("Training Configuration:")
    print("="*50)
    print(f"Training data: {config['data']['train_data_dir']}")
    print(f"Validation data: {config['data']['val_data_dir']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Device: {args.device}")
    print("="*50 + "\n")
    
    # Create audio processor
    print("Initializing audio processor...")
    audio_processor = AudioProcessor(**config['data']['audio'])
    
    # Create face detector
    print("Initializing face detector...")
    face_detector = FaceDetector(
        detector_type=config['inference']['face_detection']['detector'],
        device=args.device
    )
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        config, audio_processor, face_detector
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = LipSyncTrainer(config, device=args.device)
    
    # Start training
    print("\nStarting training...")
    print("="*50)
    
    try:
        trainer.train(
            train_loader,
            val_loader,
            config['training']['num_epochs']
        )
        
        print("\n" + "="*50)
        print("Training completed successfully!")
        print(f"Model saved to: {config['training']['checkpoint_dir']}")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint('interrupted_checkpoint.pth')
        print("Checkpoint saved. You can resume training with --resume flag")
    
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        print("\nSaving checkpoint before exit...")
        trainer.save_checkpoint('error_checkpoint.pth')


if __name__ == '__main__':
    main()
