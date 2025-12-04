"""
Preprocessing utilities for data preparation.
"""

import os
import cv2
import numpy as np
from pathlib import Path


def validate_video_file(video_path: str) -> dict:
    """
    Validate video file and return metadata.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with validation results
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {
            'valid': False,
            'error': 'Cannot open video file'
        }
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    # Validation checks
    issues = []
    if fps < 15:
        issues.append(f'Low FPS: {fps}')
    if width < 256 or height < 256:
        issues.append(f'Low resolution: {width}x{height}')
    if duration < 1.0:
        issues.append(f'Too short: {duration:.1f}s')
    if duration > 30.0:
        issues.append(f'Too long: {duration:.1f}s')
    
    return {
        'valid': len(issues) == 0,
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration,
        'issues': issues
    }


def preprocess_image(image: np.ndarray, target_size: int = 256) -> np.ndarray:
    """
    Preprocess image: resize and normalize.
    
    Args:
        image: Input image
        target_size: Target size
        
    Returns:
        Preprocessed image
    """
    # Resize
    if image.shape[0] != target_size or image.shape[1] != target_size:
        image = cv2.resize(image, (target_size, target_size))
    
    # Normalize to [-1, 1]
    image = image.astype(np.float32) / 127.5 - 1.0
    
    return image


def augment_image(image: np.ndarray, augmentation_type: str = 'random') -> np.ndarray:
    """
    Apply data augmentation to image.
    
    Args:
        image: Input image
        augmentation_type: Type of augmentation
        
    Returns:
        Augmented image
    """
    if augmentation_type == 'brightness':
        factor = np.random.uniform(0.8, 1.2)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
    
    elif augmentation_type == 'contrast':
        factor = np.random.uniform(0.8, 1.2)
        mean = image.mean()
        image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    elif augmentation_type == 'flip':
        image = cv2.flip(image, 1)
    
    elif augmentation_type == 'random':
        # Apply random combination
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
    
    return image


def split_dataset(
    data_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
):
    """
    Split dataset into train/val/test sets.
    
    Args:
        data_dir: Directory with all data
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
    """
    import shutil
    import random
    
    data_path = Path(data_dir)
    
    # Find all video files
    video_files = []
    for ext in ['.mp4', '.avi', '.mov']:
        video_files.extend(list(data_path.glob(f'*{ext}')))
    
    # Shuffle
    random.shuffle(video_files)
    
    # Calculate splits
    total = len(video_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        'train': video_files[:train_end],
        'val': video_files[train_end:val_end],
        'test': video_files[val_end:]
    }
    
    # Create directories and copy files
    for split_name, files in splits.items():
        split_dir = data_path.parent / split_name
        split_dir.mkdir(exist_ok=True)
        
        print(f"\n{split_name}: {len(files)} videos")
        for i, file in enumerate(files):
            dest = split_dir / f"{split_name}_{i:05d}{file.suffix}"
            shutil.copy2(file, dest)
        
        print(f"  Copied to {split_dir}")
    
    print(f"\nDataset split complete!")


def calculate_dataset_statistics(data_dir: str):
    """
    Calculate statistics for dataset.
    
    Args:
        data_dir: Directory with video files
    """
    from utils.video_utils import VideoReader
    
    data_path = Path(data_dir)
    video_files = list(data_path.glob('*.mp4')) + list(data_path.glob('*.avi'))
    
    if len(video_files) == 0:
        print("No video files found")
        return
    
    total_duration = 0
    total_frames = 0
    resolutions = []
    fps_list = []
    
    print(f"Analyzing {len(video_files)} videos...")
    
    for video_file in video_files:
        try:
            reader = VideoReader(str(video_file))
            duration = reader.frame_count / reader.fps
            
            total_duration += duration
            total_frames += reader.frame_count
            resolutions.append((reader.width, reader.height))
            fps_list.append(reader.fps)
            
            reader.close()
        except:
            pass
    
    # Print statistics
    print("\n" + "="*50)
    print("Dataset Statistics")
    print("="*50)
    print(f"Number of videos: {len(video_files)}")
    print(f"Total duration: {total_duration/3600:.2f} hours")
    print(f"Total frames: {total_frames:,}")
    print(f"Average FPS: {np.mean(fps_list):.1f}")
    print(f"Most common resolution: {max(set(resolutions), key=resolutions.count)}")
    print("="*50)
