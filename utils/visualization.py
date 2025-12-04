"""
Visualization utilities for monitoring and debugging.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import torch


def visualize_audio_features(
    mel_spectrogram: np.ndarray,
    save_path: str = None
):
    """
    Visualize mel-spectrogram.
    
    Args:
        mel_spectrogram: Mel-spectrogram array
        save_path: Path to save visualization
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_landmarks(
    image: np.ndarray,
    landmarks: np.ndarray,
    save_path: str = None
) -> np.ndarray:
    """
    Visualize facial landmarks on image.
    
    Args:
        image: Input image
        landmarks: Facial landmarks
        save_path: Path to save visualization
        
    Returns:
        Image with landmarks drawn
    """
    vis_image = image.copy()
    
    # Draw all landmarks
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(vis_image, (int(x), int(y)), 2, (0, 255, 0), -1)
    
    # Highlight mouth landmarks (48-68)
    if len(landmarks) >= 68:
        mouth_landmarks = landmarks[48:68]
        for i, (x, y) in enumerate(mouth_landmarks):
            cv2.circle(vis_image, (int(x), int(y)), 3, (0, 0, 255), -1)
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    return vis_image


def create_comparison_grid(
    images: List[np.ndarray],
    titles: List[str] = None,
    save_path: str = None
) -> np.ndarray:
    """
    Create a grid comparison of images.
    
    Args:
        images: List of images
        titles: Titles for each image
        save_path: Path to save grid
        
    Returns:
        Grid image
    """
    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    for idx, (ax, img) in enumerate(zip(axes.flat, images)):
        ax.imshow(img)
        ax.axis('off')
        if titles and idx < len(titles):
            ax.set_title(titles[idx])
    
    # Hide empty subplots
    for idx in range(len(images), rows * cols):
        axes.flat[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    else:
        # Convert to numpy array
        fig.canvas.draw()
        grid = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        grid = grid.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return grid


def visualize_training_progress(
    generated: torch.Tensor,
    target: torch.Tensor,
    face: torch.Tensor,
    epoch: int,
    save_path: str
):
    """
    Visualize training progress.
    
    Args:
        generated: Generated mouth regions
        target: Target mouth regions
        face: Input face images
        epoch: Current epoch
        save_path: Path to save visualization
    """
    # Convert to numpy
    def tensor_to_numpy(t):
        t = t.detach().cpu()
        # Denormalize from [-1, 1] to [0, 255]
        t = ((t + 1.0) * 127.5).clamp(0, 255).byte()
        t = t.permute(0, 2, 3, 1).numpy()
        return t
    
    gen_np = tensor_to_numpy(generated[:4])
    tgt_np = tensor_to_numpy(target[:4])
    face_np = tensor_to_numpy(face[:4])
    
    # Create comparison
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    
    for i in range(min(4, len(gen_np))):
        axes[0, i].imshow(face_np[i])
        axes[0, i].set_title(f'Face {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(tgt_np[i])
        axes[1, i].set_title('Target')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(gen_np[i])
        axes[2, i].set_title('Generated')
        axes[2, i].axis('off')
    
    plt.suptitle(f'Training Progress - Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_curves(
    losses: dict,
    save_path: str
):
    """
    Plot training loss curves.
    
    Args:
        losses: Dictionary of loss lists
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flat
    
    for idx, (name, values) in enumerate(losses.items()):
        if idx < 4:
            axes[idx].plot(values)
            axes[idx].set_title(f'{name} Loss')
            axes[idx].set_xlabel('Iteration')
            axes[idx].set_ylabel('Loss')
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_inference_video(
    frames: List[np.ndarray],
    audio_path: str,
    output_path: str,
    fps: int = 25
):
    """
    Create video from frames with audio.
    
    Args:
        frames: List of video frames
        audio_path: Path to audio file
        output_path: Output video path
        fps: Frames per second
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from utils.video_utils import VideoWriter, combine_video_audio
    
    # Write frames to temporary video
    temp_video = output_path.replace('.mp4', '_temp.mp4')
    writer = VideoWriter(temp_video, fps=fps)
    
    for frame in frames:
        writer.write_frame(frame)
    
    writer.close()
    
    # Combine with audio
    combine_video_audio(temp_video, audio_path, output_path)
    
    # Cleanup
    if os.path.exists(temp_video):
        os.remove(temp_video)


def visualize_mouth_movement(
    mouth_regions: List[np.ndarray],
    save_path: str = None
):
    """
    Visualize mouth movement over time.
    
    Args:
        mouth_regions: List of mouth region images
        save_path: Path to save visualization
    """
    n_frames = len(mouth_regions)
    display_frames = min(16, n_frames)
    indices = np.linspace(0, n_frames-1, display_frames, dtype=int)
    
    cols = 4
    rows = (display_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = axes.flat if rows > 1 else [axes]
    
    for idx, frame_idx in enumerate(indices):
        axes[idx].imshow(mouth_regions[frame_idx])
        axes[idx].set_title(f'Frame {frame_idx}')
        axes[idx].axis('off')
    
    # Hide empty subplots
    for idx in range(display_frames, rows * cols):
        if idx < len(axes):
            axes[idx].axis('off')
    
    plt.suptitle('Mouth Movement Sequence')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_debug_overlay(
    frame: np.ndarray,
    info: dict
) -> np.ndarray:
    """
    Create debug overlay on frame with info.
    
    Args:
        frame: Input frame
        info: Dictionary with debug info
        
    Returns:
        Frame with overlay
    """
    overlay = frame.copy()
    
    # Add semi-transparent background
    h, w = overlay.shape[:2]
    cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # Add text
    y_offset = 30
    for key, value in info.items():
        text = f"{key}: {value}"
        cv2.putText(
            frame, text, (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        y_offset += 20
    
    return frame
