"""
Video utilities for reading, writing, and processing video files.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Generator
import imageio
from pathlib import Path


class VideoReader:
    """Read video files frame by frame."""
    
    def __init__(self, video_path: str):
        """
        Initialize video reader.
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def __iter__(self):
        """Iterate over frames."""
        return self
    
    def __next__(self) -> np.ndarray:
        """Get next frame."""
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Read specific frame by index.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Frame as RGB numpy array, or None if failed
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def read_all_frames(self) -> List[np.ndarray]:
        """
        Read all frames from video.
        
        Returns:
            List of frames as RGB numpy arrays
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        return frames
    
    def read_frames_batch(
        self,
        start_idx: int,
        end_idx: int
    ) -> List[np.ndarray]:
        """
        Read a batch of frames.
        
        Args:
            start_idx: Starting frame index
            end_idx: Ending frame index (exclusive)
            
        Returns:
            List of frames
        """
        frames = []
        for idx in range(start_idx, end_idx):
            frame = self.read_frame(idx)
            if frame is not None:
                frames.append(frame)
        return frames
    
    def close(self):
        """Release video capture."""
        self.cap.release()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class VideoWriter:
    """Write video files frame by frame."""
    
    def __init__(
        self,
        output_path: str,
        fps: float = 25.0,
        frame_size: Optional[Tuple[int, int]] = None,
        codec: str = 'mp4v'
    ):
        """
        Initialize video writer.
        
        Args:
            output_path: Path to output video file
            fps: Frames per second
            frame_size: (width, height) of output frames
            codec: Video codec fourcc code
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.writer = None
        self.frame_count = 0
    
    def write_frame(self, frame: np.ndarray):
        """
        Write a single frame to video.
        
        Args:
            frame: Frame as RGB numpy array
        """
        # Initialize writer on first frame
        if self.writer is None:
            if self.frame_size is None:
                self.frame_size = (frame.shape[1], frame.shape[0])
            
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                self.frame_size
            )
        
        # Resize frame if needed
        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        
        # Convert RGB to BGR and write
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame_bgr)
        self.frame_count += 1
    
    def write_frames(self, frames: List[np.ndarray]):
        """
        Write multiple frames to video.
        
        Args:
            frames: List of frames as RGB numpy arrays
        """
        for frame in frames:
            self.write_frame(frame)
    
    def close(self):
        """Release video writer."""
        if self.writer is not None:
            self.writer.release()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


def create_video_from_images(
    image_paths: List[str],
    output_path: str,
    fps: float = 25.0
):
    """
    Create video from a list of image files.
    
    Args:
        image_paths: List of paths to image files
        output_path: Path to output video file
        fps: Frames per second
    """
    writer = VideoWriter(output_path, fps=fps)
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        writer.write_frame(img)
    
    writer.close()


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    frame_interval: int = 1
) -> List[str]:
    """
    Extract frames from video and save as images.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every N-th frame
        
    Returns:
        List of saved frame paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    reader = VideoReader(video_path)
    saved_paths = []
    
    for idx, frame in enumerate(reader):
        if idx % frame_interval == 0:
            output_path = output_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(
                str(output_path),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            )
            saved_paths.append(str(output_path))
    
    reader.close()
    return saved_paths


def combine_video_audio(
    video_path: str,
    audio_path: str,
    output_path: str
):
    """
    Combine video and audio files.
    
    Args:
        video_path: Path to video file (without audio)
        audio_path: Path to audio file
        output_path: Path to output video file
    """
    import subprocess
    
    command = [
        'ffmpeg',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-y',
        output_path
    ]
    
    subprocess.run(command, check=True, capture_output=True)


def resize_video(
    input_path: str,
    output_path: str,
    target_size: Tuple[int, int]
):
    """
    Resize video to target size.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        target_size: (width, height)
    """
    reader = VideoReader(input_path)
    writer = VideoWriter(output_path, fps=reader.fps, frame_size=target_size)
    
    for frame in reader:
        resized = cv2.resize(frame, target_size)
        writer.write_frame(resized)
    
    reader.close()
    writer.close()


class VideoStreamProcessor:
    """Process video in real-time streaming fashion."""
    
    def __init__(self, fps: float = 25.0):
        """
        Initialize video stream processor.
        
        Args:
            fps: Target frames per second
        """
        self.fps = fps
        self.frame_buffer = []
        self.processing_callback = None
    
    def set_processing_callback(self, callback):
        """
        Set callback function for processing frames.
        
        Args:
            callback: Function that takes frame and returns processed frame
        """
        self.processing_callback = callback
    
    def process_stream(
        self,
        input_stream: Generator[np.ndarray, None, None],
        output_writer: VideoWriter
    ):
        """
        Process video stream frame by frame.
        
        Args:
            input_stream: Generator yielding frames
            output_writer: VideoWriter instance
        """
        for frame in input_stream:
            if self.processing_callback is not None:
                frame = self.processing_callback(frame)
            output_writer.write_frame(frame)
    
    def add_to_buffer(self, frame: np.ndarray):
        """Add frame to buffer."""
        self.frame_buffer.append(frame)
    
    def get_from_buffer(self) -> Optional[np.ndarray]:
        """Get frame from buffer."""
        if len(self.frame_buffer) > 0:
            return self.frame_buffer.pop(0)
        return None
    
    def clear_buffer(self):
        """Clear frame buffer."""
        self.frame_buffer.clear()
