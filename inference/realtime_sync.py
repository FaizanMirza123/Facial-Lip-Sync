"""
Real-time lip-sync inference engine.
"""

import torch
import cv2
import numpy as np
from typing import Optional, Tuple
import queue
import threading

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lip_sync_model import LipSyncModel, load_checkpoint
from utils.audio_processing import AudioProcessor
from utils.face_detection import FaceDetector
from utils.video_utils import VideoWriter


class RealtimeLipSync:
    """Real-time lip synchronization engine."""
    
    def __init__(
        self,
        model_path: str,
        config: dict,
        device: str = 'cuda'
    ):
        """
        Initialize real-time lip sync.
        
        Args:
            model_path: Path to trained model checkpoint
            config: Configuration dictionary
            device: Device to run on
        """
        self.device = device
        self.config = config
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Create processors
        self.audio_processor = AudioProcessor(**config['data']['audio'])
        self.face_detector = FaceDetector(
            detector_type=config['inference']['face_detection']['detector'],
            device=device
        )
        
        # Buffers for streaming
        self.audio_buffer = queue.Queue(maxsize=100)
        self.frame_buffer = queue.Queue(maxsize=30)
        
        # Face cache
        self.face_image = None
        self.face_landmarks = None
        self.face_features = None
    
    def _load_model(self, model_path: str) -> LipSyncModel:
        """Load trained model."""
        # Create model
        model = LipSyncModel(
            audio_config=self.config['model']['audio_encoder'],
            face_config=self.config['model']['face_encoder'],
            generator_config=self.config['model']['generator']
        )
        
        # Load weights
        model = load_checkpoint(model, model_path, self.device)
        model.eval()
        
        return model
    
    def set_face_image(self, image_path: str):
        """
        Set the face image to animate.
        
        Args:
            image_path: Path to face image
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face
        landmarks = self.face_detector.get_landmarks(image)
        
        if landmarks is None:
            raise ValueError("No face detected in image")
        
        # Crop and resize face
        face_crop, _ = self.face_detector.crop_face(
            image, landmarks, self.config['data']['img_size']
        )
        
        # Store face
        self.face_image = face_crop
        self.face_landmarks = landmarks
        
        # Extract face features
        with torch.no_grad():
            face_tensor = self._preprocess_image(face_crop)
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
            self.face_features = self.model.face_encoder(face_tensor)
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Normalize to [-1, 1]
        image = image.astype(np.float32) / 127.5 - 1.0
        
        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        
        return torch.FloatTensor(image)
    
    def _postprocess_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Postprocess model output to image."""
        # CHW to HWC
        image = tensor.cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize from [-1, 1] to [0, 255]
        image = ((image + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        
        return image
    
    def generate_frame(self, audio_features: np.ndarray) -> np.ndarray:
        """
        Generate a single lip-synced frame.
        
        Args:
            audio_features: Audio features for current frame
            
        Returns:
            Generated frame as RGB image
        """
        if self.face_features is None:
            raise ValueError("Face image not set. Call set_face_image() first.")
        
        with torch.no_grad():
            # Prepare audio features
            audio_tensor = torch.FloatTensor(audio_features)
            audio_tensor = audio_tensor.unsqueeze(0).to(self.device)
            
            # Extract audio features
            audio_emb = self.model.audio_encoder(audio_tensor)
            
            # Generate mouth region
            mouth_region = self.model.generator(audio_emb, self.face_features)
            
            # Postprocess
            mouth_image = self._postprocess_image(mouth_region.squeeze(0))
            
            # Composite mouth onto face
            output_frame = self._composite_mouth(
                self.face_image.copy(),
                mouth_image,
                self.face_landmarks
            )
            
            return output_frame
    
    def _composite_mouth(
        self,
        face_image: np.ndarray,
        mouth_image: np.ndarray,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Composite generated mouth onto face image.
        
        Args:
            face_image: Base face image
            mouth_image: Generated mouth region
            landmarks: Facial landmarks
            
        Returns:
            Composited image
        """
        # Get mouth bounding box
        x1, y1, x2, y2 = self.face_detector.get_mouth_bbox(landmarks)
        
        # Resize mouth to fit bbox
        mouth_h = y2 - y1
        mouth_w = x2 - x1
        mouth_resized = cv2.resize(mouth_image, (mouth_w, mouth_h))
        
        # Create smooth blend mask
        mask = np.ones((mouth_h, mouth_w), dtype=np.float32)
        blur_size = int(min(mouth_h, mouth_w) * 0.2)
        if blur_size % 2 == 0:
            blur_size += 1
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        mask = mask[:, :, np.newaxis]
        
        # Blend mouth into face
        face_image[y1:y2, x1:x2] = (
            mouth_resized * mask + face_image[y1:y2, x1:x2] * (1 - mask)
        ).astype(np.uint8)
        
        return face_image
    
    def process_audio_file(
        self,
        audio_path: str,
        output_path: str
    ):
        """
        Process audio file and generate lip-synced video.
        
        Args:
            audio_path: Path to audio file
            output_path: Path to output video file
        """
        if self.face_features is None:
            raise ValueError("Face image not set. Call set_face_image() first.")
        
        # Process audio
        mel_tensor, num_frames = self.audio_processor.process_audio_for_inference(
            audio_path
        )
        
        # Create video writer
        writer = VideoWriter(
            output_path,
            fps=self.config['data']['fps'],
            frame_size=(self.config['data']['img_size'], self.config['data']['img_size'])
        )
        
        print(f"Generating {num_frames} frames...")
        
        # Generate frames
        for i in range(num_frames):
            audio_features = mel_tensor[i].numpy()
            frame = self.generate_frame(audio_features)
            writer.write_frame(frame)
            
            if (i + 1) % 25 == 0:
                print(f"Generated {i + 1}/{num_frames} frames")
        
        writer.close()
        
        # Combine with audio
        from utils.video_utils import combine_video_audio
        temp_output = output_path.replace('.mp4', '_temp.mp4')
        os.rename(output_path, temp_output)
        combine_video_audio(temp_output, audio_path, output_path)
        os.remove(temp_output)
        
        print(f"Video saved to {output_path}")
    
    def process_audio_stream(
        self,
        audio_stream,
        output_stream
    ):
        """
        Process real-time audio stream.
        
        Args:
            audio_stream: Audio input stream
            output_stream: Video output stream
        """
        # TODO: Implement streaming processing
        # This would require buffering audio, extracting features in real-time,
        # and generating frames with minimal latency
        pass


class BatchLipSync:
    """Batch processing for multiple videos."""
    
    def __init__(
        self,
        model_path: str,
        config: dict,
        device: str = 'cuda'
    ):
        """Initialize batch processor."""
        self.lip_sync = RealtimeLipSync(model_path, config, device)
    
    def process_batch(
        self,
        face_image_path: str,
        audio_paths: list,
        output_dir: str
    ):
        """
        Process multiple audio files with same face.
        
        Args:
            face_image_path: Path to face image
            audio_paths: List of audio file paths
            output_dir: Directory for output videos
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Set face once
        self.lip_sync.set_face_image(face_image_path)
        
        # Process each audio file
        for i, audio_path in enumerate(audio_paths):
            output_path = os.path.join(
                output_dir,
                f"output_{i:04d}.mp4"
            )
            
            print(f"\nProcessing {i+1}/{len(audio_paths)}: {audio_path}")
            self.lip_sync.process_audio_file(audio_path, output_path)
