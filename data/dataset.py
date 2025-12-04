"""
Dataset classes for loading and preprocessing training data.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from typing import Tuple, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio_processing import AudioProcessor
from utils.face_detection import FaceDetector
from utils.video_utils import VideoReader


class LipSyncDataset(Dataset):
    """Dataset for lip-sync training."""
    
    def __init__(
        self,
        data_dir: str,
        audio_processor: AudioProcessor,
        face_detector: FaceDetector,
        img_size: int = 256,
        mouth_size: int = 96,
        fps: int = 25,
        window_size: int = 16,
        augment: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing video files
            audio_processor: AudioProcessor instance
            face_detector: FaceDetector instance
            img_size: Size of face images
            mouth_size: Size of mouth region crops
            fps: Frames per second
            window_size: Audio window size in frames
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.audio_processor = audio_processor
        self.face_detector = face_detector
        self.img_size = img_size
        self.mouth_size = mouth_size
        self.fps = fps
        self.window_size = window_size
        self.augment = augment
        
        # Find all video files
        self.video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov']:
            self.video_files.extend(list(self.data_dir.glob(ext)))
        
        if len(self.video_files) == 0:
            raise ValueError(f"No video files found in {data_dir}")
        
        print(f"Found {len(self.video_files)} videos in {data_dir}")
    
    def __len__(self) -> int:
        return len(self.video_files)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a training sample.
        
        Returns:
            Dictionary with:
                - face: Face image tensor (3, img_size, img_size)
                - audio: Audio features (mel_bins, window_size)
                - mouth: Mouth region tensor (3, mouth_size, mouth_size)
                - is_synced: Whether audio and video are synced (1) or not (0)
        """
        video_path = str(self.video_files[idx])
        
        try:
            # Read video
            reader = VideoReader(video_path)
            frames = reader.read_all_frames()
            reader.close()
            
            if len(frames) == 0:
                return self.__getitem__((idx + 1) % len(self))
            
            # Extract audio
            audio = self.audio_processor.extract_audio_from_video(video_path)
            
            # Align audio to video length
            audio = self.audio_processor.align_audio_to_video(audio, len(frames))
            
            # Extract mel-spectrogram
            mel_spec = self.audio_processor.extract_mel_spectrogram(audio)
            
            # Get audio windows
            audio_windows = self.audio_processor.get_audio_windows(mel_spec, len(frames))
            
            # Select random frame
            frame_idx = random.randint(0, len(frames) - 1)
            frame = frames[frame_idx]
            audio_window = audio_windows[frame_idx]
            
            # Detect face and landmarks
            landmarks = self.face_detector.get_landmarks(frame)
            
            if landmarks is None:
                # Skip this sample if no face detected
                return self.__getitem__((idx + 1) % len(self))
            
            # Crop face
            face_crop, _ = self.face_detector.crop_face(frame, landmarks, self.img_size)
            
            # Crop mouth region
            mouth_crop = self.face_detector.crop_mouth_region(frame, landmarks, self.mouth_size)
            
            # Random sync/unsync
            is_synced = random.random() > 0.5
            
            if not is_synced:
                # Use audio from different frame for negative samples
                wrong_idx = random.randint(0, len(frames) - 1)
                while wrong_idx == frame_idx:
                    wrong_idx = random.randint(0, len(frames) - 1)
                audio_window = audio_windows[wrong_idx]
            
            # Data augmentation
            if self.augment:
                face_crop = self._augment_image(face_crop)
                mouth_crop = self._augment_image(mouth_crop)
            
            # Convert to tensors
            face_tensor = self._to_tensor(face_crop)
            mouth_tensor = self._to_tensor(mouth_crop)
            audio_tensor = torch.FloatTensor(audio_window)
            
            return {
                'face': face_tensor,
                'audio': audio_tensor,
                'mouth': mouth_tensor,
                'is_synced': torch.FloatTensor([1.0 if is_synced else 0.0])
            }
            
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
    
    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """Apply random augmentation to image."""
        # Random brightness
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
        
        # Random contrast
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            mean = img.mean()
            img = np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        # Random horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        
        return img
    
    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert image to tensor and normalize."""
        # Normalize to [-1, 1]
        img = img.astype(np.float32) / 127.5 - 1.0
        
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        return torch.FloatTensor(img)


def create_dataloaders(
    config: dict,
    audio_processor: AudioProcessor,
    face_detector: FaceDetector
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        config: Configuration dictionary
        audio_processor: AudioProcessor instance
        face_detector: FaceDetector instance
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = LipSyncDataset(
        data_dir=config['data']['train_data_dir'],
        audio_processor=audio_processor,
        face_detector=face_detector,
        img_size=config['data']['img_size'],
        mouth_size=config['data']['crop_size'],
        fps=config['data']['fps'],
        window_size=config['data']['audio']['window_size'],
        augment=True
    )
    
    val_dataset = LipSyncDataset(
        data_dir=config['data']['val_data_dir'],
        audio_processor=audio_processor,
        face_detector=face_detector,
        img_size=config['data']['img_size'],
        mouth_size=config['data']['crop_size'],
        fps=config['data']['fps'],
        window_size=config['data']['audio']['window_size'],
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    return train_loader, val_loader
