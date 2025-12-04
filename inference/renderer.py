"""
Face rendering utilities for compositing and blending.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import torch


class FaceRenderer:
    """Renders lip-synced faces with smooth blending."""
    
    def __init__(self, blend_method: str = 'poisson'):
        """
        Initialize face renderer.
        
        Args:
            blend_method: Blending method ('alpha', 'poisson', 'multiband')
        """
        self.blend_method = blend_method
    
    def composite_mouth(
        self,
        base_face: np.ndarray,
        mouth_region: np.ndarray,
        mouth_bbox: Tuple[int, int, int, int],
        blend_margin: float = 0.2
    ) -> np.ndarray:
        """
        Composite mouth region onto base face.
        
        Args:
            base_face: Base face image
            mouth_region: Generated mouth region
            mouth_bbox: Mouth bounding box (x1, y1, x2, y2)
            blend_margin: Margin for blending (fraction of mouth size)
            
        Returns:
            Composited face image
        """
        x1, y1, x2, y2 = mouth_bbox
        
        # Resize mouth to bbox
        mouth_h = y2 - y1
        mouth_w = x2 - x1
        mouth_resized = cv2.resize(mouth_region, (mouth_w, mouth_h))
        
        if self.blend_method == 'alpha':
            return self._alpha_blend(
                base_face, mouth_resized, (x1, y1, x2, y2), blend_margin
            )
        elif self.blend_method == 'poisson':
            return self._poisson_blend(
                base_face, mouth_resized, (x1, y1, x2, y2)
            )
        else:
            return self._alpha_blend(
                base_face, mouth_resized, (x1, y1, x2, y2), blend_margin
            )
    
    def _alpha_blend(
        self,
        base: np.ndarray,
        overlay: np.ndarray,
        bbox: Tuple[int, int, int, int],
        margin: float
    ) -> np.ndarray:
        """Alpha blending with smooth transitions."""
        x1, y1, x2, y2 = bbox
        h, w = overlay.shape[:2]
        
        # Create blend mask
        mask = self._create_blend_mask(h, w, margin)
        
        # Apply mask
        result = base.copy()
        result[y1:y2, x1:x2] = (
            overlay * mask + base[y1:y2, x1:x2] * (1 - mask)
        ).astype(np.uint8)
        
        return result
    
    def _create_blend_mask(
        self,
        height: int,
        width: int,
        margin: float
    ) -> np.ndarray:
        """Create smooth blend mask with margins."""
        # Create base mask
        mask = np.ones((height, width), dtype=np.float32)
        
        # Calculate margin size
        margin_h = int(height * margin)
        margin_w = int(width * margin)
        
        # Create gradient on edges
        for i in range(margin_h):
            alpha = i / margin_h
            mask[i, :] *= alpha
            mask[-(i+1), :] *= alpha
        
        for j in range(margin_w):
            alpha = j / margin_w
            mask[:, j] *= alpha
            mask[:, -(j+1)] *= alpha
        
        # Apply Gaussian blur for smoothness
        blur_size = int(min(height, width) * 0.15)
        if blur_size % 2 == 0:
            blur_size += 1
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        
        return mask[:, :, np.newaxis]
    
    def _poisson_blend(
        self,
        base: np.ndarray,
        overlay: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Poisson blending for seamless compositing."""
        x1, y1, x2, y2 = bbox
        
        # Create mask for poisson blending
        mask = np.ones(overlay.shape[:2], dtype=np.uint8) * 255
        
        # Calculate center point
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # Poisson blend
        result = cv2.seamlessClone(
            overlay,
            base,
            mask,
            center,
            cv2.NORMAL_CLONE
        )
        
        return result
    
    def color_correction(
        self,
        source: np.ndarray,
        target: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Match color statistics of source to target.
        
        Args:
            source: Source image
            target: Target image to match
            mask: Optional mask for region of interest
            
        Returns:
            Color-corrected source image
        """
        if mask is None:
            mask = np.ones(source.shape[:2], dtype=bool)
        
        result = source.copy().astype(np.float32)
        
        for i in range(3):  # RGB channels
            source_channel = source[:, :, i][mask]
            target_channel = target[:, :, i][mask]
            
            source_mean = source_channel.mean()
            source_std = source_channel.std()
            target_mean = target_channel.mean()
            target_std = target_channel.std()
            
            # Match statistics
            result[:, :, i] = (
                (result[:, :, i] - source_mean) * (target_std / source_std) +
                target_mean
            )
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def enhance_quality(
        self,
        image: np.ndarray,
        sharpen: bool = True,
        denoise: bool = False
    ) -> np.ndarray:
        """
        Enhance image quality.
        
        Args:
            image: Input image
            sharpen: Apply sharpening
            denoise: Apply denoising
            
        Returns:
            Enhanced image
        """
        result = image.copy()
        
        if denoise:
            result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)
        
        if sharpen:
            # Unsharp mask
            blurred = cv2.GaussianBlur(result, (0, 0), 3)
            result = cv2.addWeighted(result, 1.5, blurred, -0.5, 0)
        
        return result
    
    def stabilize_mouth(
        self,
        current_mouth: np.ndarray,
        previous_mouth: Optional[np.ndarray],
        alpha: float = 0.7
    ) -> np.ndarray:
        """
        Temporal stabilization of mouth movements.
        
        Args:
            current_mouth: Current frame's mouth
            previous_mouth: Previous frame's mouth
            alpha: Blending factor (0=all previous, 1=all current)
            
        Returns:
            Stabilized mouth region
        """
        if previous_mouth is None:
            return current_mouth
        
        # Blend with previous frame
        stabilized = cv2.addWeighted(
            current_mouth, alpha,
            previous_mouth, 1 - alpha,
            0
        )
        
        return stabilized.astype(np.uint8)


class VideoCompositor:
    """Composites face and mouth regions into video frames."""
    
    def __init__(self, renderer: FaceRenderer):
        """Initialize compositor."""
        self.renderer = renderer
        self.previous_mouth = None
        self.frame_count = 0
    
    def composite_frame(
        self,
        base_face: np.ndarray,
        mouth_region: np.ndarray,
        mouth_bbox: Tuple[int, int, int, int],
        stabilize: bool = True
    ) -> np.ndarray:
        """
        Composite a single video frame.
        
        Args:
            base_face: Base face image
            mouth_region: Generated mouth region
            mouth_bbox: Mouth bounding box
            stabilize: Apply temporal stabilization
            
        Returns:
            Composited frame
        """
        # Stabilize if requested
        if stabilize and self.previous_mouth is not None:
            mouth_region = self.renderer.stabilize_mouth(
                mouth_region, self.previous_mouth
            )
        
        # Composite
        frame = self.renderer.composite_mouth(
            base_face, mouth_region, mouth_bbox
        )
        
        # Update previous
        self.previous_mouth = mouth_region
        self.frame_count += 1
        
        return frame
    
    def reset(self):
        """Reset compositor state."""
        self.previous_mouth = None
        self.frame_count = 0
