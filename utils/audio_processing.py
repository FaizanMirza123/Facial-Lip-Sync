"""
Audio processing utilities for lip-sync model.
Extracts mel-spectrograms and other audio features from speech signals.
"""

import torch
import librosa
import numpy as np
from scipy import signal
from typing import Tuple, Optional


class AudioProcessor:
    """Processes audio signals for lip-sync model input."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 800,
        hop_length: int = 200,
        n_mels: int = 80,
        win_length: int = 800,
        window_size: int = 16,
        fps: int = 25
    ):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Audio sampling rate in Hz
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mels: Number of mel filterbanks
            win_length: Window length for STFT
            window_size: Number of frames for audio context window
            fps: Frames per second for video sync
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.win_length = win_length
        self.window_size = window_size
        self.fps = fps
        
        # Calculate samples per frame for video sync
        self.samples_per_frame = sample_rate // fps
        
    def load_audio(self, audio_path: str, sr: Optional[int] = None) -> np.ndarray:
        """
        Load audio file and resample if needed.
        
        Args:
            audio_path: Path to audio file
            sr: Target sample rate (uses default if None)
            
        Returns:
            Audio waveform as numpy array
        """
        sr = sr or self.sample_rate
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        return audio
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio: Input audio waveform
            
        Returns:
            Normalized audio
        """
        audio = audio.astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel-spectrogram from audio.
        
        Args:
            audio: Input audio waveform
            
        Returns:
            Mel-spectrogram of shape (n_mels, time_steps)
        """
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmax=8000
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Input audio waveform
            n_mfcc: Number of MFCCs to extract
            
        Returns:
            MFCC features of shape (n_mfcc, time_steps)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        
        return mfcc
    
    def get_audio_windows(
        self, 
        mel_spec: np.ndarray, 
        num_frames: int
    ) -> np.ndarray:
        """
        Create sliding windows of mel-spectrogram for each video frame.
        
        Args:
            mel_spec: Mel-spectrogram of shape (n_mels, time_steps)
            num_frames: Number of video frames
            
        Returns:
            Audio windows of shape (num_frames, n_mels, window_size)
        """
        # Calculate frames in mel-spectrogram
        mel_frames = mel_spec.shape[1]
        
        # Create windows
        windows = []
        for i in range(num_frames):
            # Calculate center frame in mel-spectrogram
            center_frame = int((i / num_frames) * mel_frames)
            
            # Get window around center frame
            start_frame = max(0, center_frame - self.window_size // 2)
            end_frame = min(mel_frames, start_frame + self.window_size)
            
            # Extract window
            window = mel_spec[:, start_frame:end_frame]
            
            # Pad if necessary
            if window.shape[1] < self.window_size:
                padding = self.window_size - window.shape[1]
                window = np.pad(window, ((0, 0), (0, padding)), mode='edge')
            
            windows.append(window)
        
        return np.array(windows)
    
    def process_audio_for_inference(
        self, 
        audio_path: str
    ) -> Tuple[torch.Tensor, int]:
        """
        Process audio file for inference.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (mel_spectrogram_tensor, num_frames)
        """
        # Load and normalize audio
        audio = self.load_audio(audio_path)
        audio = self.normalize_audio(audio)
        
        # Extract mel-spectrogram
        mel_spec = self.extract_mel_spectrogram(audio)
        
        # Calculate number of frames based on audio duration
        num_frames = int(len(audio) / self.samples_per_frame)
        
        # Get audio windows
        windows = self.get_audio_windows(mel_spec, num_frames)
        
        # Convert to tensor
        mel_tensor = torch.FloatTensor(windows)
        
        return mel_tensor, num_frames
    
    def align_audio_to_video(
        self,
        audio: np.ndarray,
        num_video_frames: int
    ) -> np.ndarray:
        """
        Align audio length to match video frame count.
        
        Args:
            audio: Input audio waveform
            num_video_frames: Number of frames in video
            
        Returns:
            Aligned audio waveform
        """
        target_length = num_video_frames * self.samples_per_frame
        
        if len(audio) < target_length:
            # Pad audio
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        elif len(audio) > target_length:
            # Trim audio
            audio = audio[:target_length]
        
        return audio
    
    def extract_audio_from_video(self, video_path: str) -> np.ndarray:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Audio waveform
        """
        import subprocess
        import tempfile
        import os
        
        # Create temporary audio file
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        try:
            # Extract audio using ffmpeg
            command = [
                'ffmpeg',
                '-i', video_path,
                '-ac', '1',  # Mono
                '-ar', str(self.sample_rate),  # Sample rate
                '-f', 'wav',
                '-y',  # Overwrite
                temp_audio_path
            ]
            
            subprocess.run(command, check=True, capture_output=True)
            
            # Load extracted audio
            audio = self.load_audio(temp_audio_path)
            
            return audio
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)


def preprocess_audio_batch(
    audio_paths: list,
    processor: AudioProcessor
) -> torch.Tensor:
    """
    Preprocess a batch of audio files.
    
    Args:
        audio_paths: List of paths to audio files
        processor: AudioProcessor instance
        
    Returns:
        Batch tensor of mel-spectrograms
    """
    batch_mels = []
    
    for audio_path in audio_paths:
        mel_tensor, _ = processor.process_audio_for_inference(audio_path)
        batch_mels.append(mel_tensor)
    
    # Stack into batch
    return torch.stack(batch_mels)
