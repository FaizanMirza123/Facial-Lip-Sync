"""
Utility script to prepare training data from videos.
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.video_utils import VideoReader, extract_frames_from_video
from utils.face_detection import FaceDetector


def validate_video(video_path: str, face_detector: FaceDetector) -> bool:
    """
    Check if video contains clear face.
    
    Args:
        video_path: Path to video
        face_detector: FaceDetector instance
        
    Returns:
        True if video is valid
    """
    try:
        reader = VideoReader(video_path)
        
        # Check first, middle, and last frames
        indices = [0, reader.frame_count // 2, reader.frame_count - 1]
        face_count = 0
        
        for idx in indices:
            frame = reader.read_frame(idx)
            if frame is not None:
                landmarks = face_detector.get_landmarks(frame)
                if landmarks is not None:
                    face_count += 1
        
        reader.close()
        
        # Require face in at least 2 out of 3 sampled frames
        return face_count >= 2
        
    except Exception as e:
        print(f"Error validating {video_path}: {e}")
        return False


def prepare_dataset(
    input_dir: str,
    output_dir: str,
    min_duration: float = 2.0,
    max_duration: float = 10.0,
    validate_faces: bool = True
):
    """
    Prepare training dataset from raw videos.
    
    Args:
        input_dir: Directory with raw videos
        output_dir: Output directory for processed videos
        min_duration: Minimum video duration in seconds
        max_duration: Maximum video duration in seconds
        validate_faces: Whether to validate face presence
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.rglob(f'*{ext}'))
    
    print(f"Found {len(video_files)} video files")
    
    # Initialize face detector if validating
    if validate_faces:
        print("Initializing face detector...")
        face_detector = FaceDetector(device='cuda')
    else:
        face_detector = None
    
    # Process videos
    valid_count = 0
    invalid_count = 0
    
    for i, video_file in enumerate(video_files):
        print(f"\nProcessing {i+1}/{len(video_files)}: {video_file.name}")
        
        try:
            # Check duration
            reader = VideoReader(str(video_file))
            duration = reader.frame_count / reader.fps
            reader.close()
            
            if duration < min_duration:
                print(f"  Skipped: Too short ({duration:.1f}s)")
                invalid_count += 1
                continue
            
            if duration > max_duration:
                print(f"  Skipped: Too long ({duration:.1f}s)")
                invalid_count += 1
                continue
            
            # Validate face
            if validate_faces:
                if not validate_video(str(video_file), face_detector):
                    print(f"  Skipped: No clear face detected")
                    invalid_count += 1
                    continue
            
            # Copy to output
            output_file = output_path / f"video_{valid_count:05d}{video_file.suffix}"
            shutil.copy2(video_file, output_file)
            
            print(f"  ✓ Valid ({duration:.1f}s)")
            valid_count += 1
            
        except Exception as e:
            print(f"  Error: {e}")
            invalid_count += 1
    
    print("\n" + "="*50)
    print(f"Dataset preparation complete!")
    print(f"Valid videos: {valid_count}")
    print(f"Invalid videos: {invalid_count}")
    print(f"Output directory: {output_dir}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare training dataset from videos'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory with raw videos'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for processed videos'
    )
    parser.add_argument(
        '--min-duration',
        type=float,
        default=2.0,
        help='Minimum video duration in seconds'
    )
    parser.add_argument(
        '--max-duration',
        type=float,
        default=10.0,
        help='Maximum video duration in seconds'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip face validation'
    )
    
    args = parser.parse_args()
    
    prepare_dataset(
        args.input,
        args.output,
        args.min_duration,
        args.max_duration,
        validate_faces=not args.no_validate
    )


if __name__ == '__main__':
    main()
