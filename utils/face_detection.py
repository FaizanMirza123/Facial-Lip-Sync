"""
Face detection and facial landmark extraction utilities.
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Optional, List
import face_alignment
from scipy.spatial import ConvexHull


class FaceDetector:
    """Detects faces and extracts facial landmarks."""
    
    def __init__(
        self,
        detector_type: str = 'sfd',
        device: str = 'cuda',
        landmark_type: str = '2D'
    ):
        """
        Initialize face detector.
        
        Args:
            detector_type: Type of face detector ('sfd', 'dlib', 'mtcnn')
            device: Device to run on ('cuda' or 'cpu')
            landmark_type: Type of landmarks ('2D' or '3D')
        """
        self.device = device
        self.detector_type = detector_type
        
        # Initialize face alignment detector
        if landmark_type == '3D':
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.THREE_D,
                device=device,
                face_detector=detector_type
            )
        else:
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                device=device,
                face_detector=detector_type
            )
    
    def detect_faces(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect faces in image.
        
        Args:
            image: Input image (RGB)
            
        Returns:
            List of face bounding boxes [x1, y1, x2, y2]
        """
        faces = self.fa.face_detector.detect_from_image(image)
        return faces
    
    def get_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract facial landmarks from image.
        
        Args:
            image: Input image (RGB)
            
        Returns:
            Landmarks array of shape (68, 2) or (68, 3) for 3D
            Returns None if no face detected
        """
        landmarks = self.fa.get_landmarks(image)
        
        if landmarks is None or len(landmarks) == 0:
            return None
        
        # Return first face's landmarks
        return landmarks[0]
    
    def get_mouth_landmarks(
        self, 
        landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract mouth region landmarks.
        
        Args:
            landmarks: Full facial landmarks (68 points)
            
        Returns:
            Tuple of (inner_mouth_landmarks, outer_mouth_landmarks)
        """
        # Mouth landmarks indices (68-point model)
        outer_mouth = landmarks[48:60]  # Outer lip contour
        inner_mouth = landmarks[60:68]  # Inner lip contour
        
        return inner_mouth, outer_mouth
    
    def get_face_bbox(
        self, 
        landmarks: np.ndarray, 
        margin: float = 0.2
    ) -> Tuple[int, int, int, int]:
        """
        Get face bounding box from landmarks.
        
        Args:
            landmarks: Facial landmarks
            margin: Margin to add around face (fraction of face size)
            
        Returns:
            Bounding box (x1, y1, x2, y2)
        """
        x_min = landmarks[:, 0].min()
        x_max = landmarks[:, 0].max()
        y_min = landmarks[:, 1].min()
        y_max = landmarks[:, 1].max()
        
        # Add margin
        width = x_max - x_min
        height = y_max - y_min
        margin_x = width * margin
        margin_y = height * margin
        
        x1 = int(max(0, x_min - margin_x))
        y1 = int(max(0, y_min - margin_y))
        x2 = int(x_max + margin_x)
        y2 = int(y_max + margin_y)
        
        return x1, y1, x2, y2
    
    def get_mouth_bbox(
        self, 
        landmarks: np.ndarray,
        margin: float = 0.5
    ) -> Tuple[int, int, int, int]:
        """
        Get mouth region bounding box from landmarks.
        
        Args:
            landmarks: Facial landmarks
            margin: Margin to add around mouth (fraction of mouth size)
            
        Returns:
            Bounding box (x1, y1, x2, y2)
        """
        # Get mouth landmarks
        _, outer_mouth = self.get_mouth_landmarks(landmarks)
        
        x_min = outer_mouth[:, 0].min()
        x_max = outer_mouth[:, 0].max()
        y_min = outer_mouth[:, 1].min()
        y_max = outer_mouth[:, 1].max()
        
        # Add margin
        width = x_max - x_min
        height = y_max - y_min
        margin_x = width * margin
        margin_y = height * margin
        
        x1 = int(max(0, x_min - margin_x))
        y1 = int(max(0, y_min - margin_y))
        x2 = int(x_max + margin_x)
        y2 = int(y_max + margin_y)
        
        return x1, y1, x2, y2
    
    def crop_face(
        self, 
        image: np.ndarray, 
        landmarks: np.ndarray,
        output_size: int = 256
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crop and resize face region from image.
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            output_size: Size of output face crop
            
        Returns:
            Tuple of (cropped_face, transform_matrix)
        """
        x1, y1, x2, y2 = self.get_face_bbox(landmarks)
        
        # Crop face
        face = image[y1:y2, x1:x2]
        
        # Resize to output size
        face_resized = cv2.resize(face, (output_size, output_size))
        
        # Calculate transform matrix for later use
        scale_x = output_size / (x2 - x1)
        scale_y = output_size / (y2 - y1)
        transform = np.array([
            [scale_x, 0, -x1 * scale_x],
            [0, scale_y, -y1 * scale_y],
            [0, 0, 1]
        ])
        
        return face_resized, transform
    
    def crop_mouth_region(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        output_size: int = 96
    ) -> np.ndarray:
        """
        Crop mouth region from image.
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            output_size: Size of output mouth crop
            
        Returns:
            Cropped mouth region
        """
        x1, y1, x2, y2 = self.get_mouth_bbox(landmarks)
        
        # Crop mouth
        mouth = image[y1:y2, x1:x2]
        
        # Resize to output size
        mouth_resized = cv2.resize(mouth, (output_size, output_size))
        
        return mouth_resized
    
    def align_face(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        output_size: int = 256
    ) -> np.ndarray:
        """
        Align face based on eye positions.
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            output_size: Size of output aligned face
            
        Returns:
            Aligned face image
        """
        # Get eye centers
        left_eye = landmarks[36:42].mean(axis=0)
        right_eye = landmarks[42:48].mean(axis=0)
        
        # Calculate angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Get face center
        face_center = landmarks.mean(axis=0)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(
            tuple(face_center.astype(int)),
            angle,
            1.0
        )
        
        # Rotate image
        aligned = cv2.warpAffine(
            image,
            rotation_matrix,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_CUBIC
        )
        
        # Crop and resize
        aligned_crop, _ = self.crop_face(aligned, landmarks, output_size)
        
        return aligned_crop
    
    def create_face_mask(
        self,
        image_shape: Tuple[int, int],
        landmarks: np.ndarray,
        blur_radius: int = 15
    ) -> np.ndarray:
        """
        Create a mask for the face region.
        
        Args:
            image_shape: Shape of image (height, width)
            landmarks: Facial landmarks
            blur_radius: Radius for Gaussian blur on mask edges
            
        Returns:
            Face mask (0-1 values)
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        # Get convex hull of face landmarks
        hull = ConvexHull(landmarks[:, :2])
        hull_points = landmarks[hull.vertices, :2].astype(np.int32)
        
        # Fill convex hull
        cv2.fillConvexPoly(mask, hull_points, 255)
        
        # Blur edges
        if blur_radius > 0:
            mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
        
        # Normalize to [0, 1]
        mask = mask.astype(np.float32) / 255.0
        
        return mask


def batch_detect_faces(
    images: List[np.ndarray],
    detector: FaceDetector
) -> List[Optional[np.ndarray]]:
    """
    Detect faces in a batch of images.
    
    Args:
        images: List of images
        detector: FaceDetector instance
        
    Returns:
        List of landmarks for each image (None if no face detected)
    """
    landmarks_list = []
    
    for image in images:
        landmarks = detector.get_landmarks(image)
        landmarks_list.append(landmarks)
    
    return landmarks_list
