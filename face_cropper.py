import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple

class FaceCropper:
    """
    Face cropping utility using MediaPipe Face Mesh for eye region extraction.
    Designed to preprocess images for eye segmentation model that expects specific face regions.
    """
    
    def __init__(self, static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5):
        """
        Initialize MediaPipe Face Mesh.
        
        Args:
            static_image_mode: Process images independently (True for single images)
            max_num_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum confidence for face detection
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,  # Includes iris landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
        # Define landmark indices for regions of interest
        # Upper lip upper limit
        self.upper_lip_top = [13, 14, 269, 270, 267, 271, 272]
        
        # Forehead upper limit (hairline approximation)
        self.forehead_top = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 340, 346, 347, 348, 349, 350, 451, 452, 453]
        
        # Ear regions for lateral limits
        self.right_ear_landmarks = [234, 93, 132, 361, 340, 346, 347, 348, 349, 350]
        self.left_ear_landmarks = [454, 323, 361, 340, 346, 347, 348, 349, 350, 127, 162, 21, 54, 103, 67, 109]
        
    def get_landmark_coordinates(self, landmarks, indices, image_shape):
        """
        Extract pixel coordinates for specified landmark indices.
        
        Args:
            landmarks: MediaPipe normalized landmarks
            indices: List of landmark indices
            image_shape: Tuple of (height, width) of the image
            
        Returns:
            List of (x, y) coordinates
        """
        h, w = image_shape[:2]
        coords = []
        for idx in indices:
            if idx < len(landmarks):
                landmark = landmarks[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                coords.append((x, y))
        return coords
    
    def calculate_crop_boundaries(self, landmarks, image_shape, ear_margin_ratio=0.15):
        """
        Calculate crop boundaries based on facial landmarks.
        
        Args:
            landmarks: MediaPipe face landmarks
            image_shape: Tuple of (height, width) of the image
            ear_margin_ratio: Ratio of face width to use as margin from ears (0.15 = ~2 fingers)
            
        Returns:
            Tuple of (top, bottom, left, right) crop coordinates
        """
        h, w = image_shape[:2]
        
        # Get upper lip coordinates (bottom boundary)
        upper_lip_coords = self.get_landmark_coordinates(landmarks, self.upper_lip_top, image_shape)
        bottom = min([y for x, y in upper_lip_coords])
        
        # Get forehead coordinates (top boundary)
        forehead_coords = self.get_landmark_coordinates(landmarks, self.forehead_top, image_shape)
        top = min([y for x, y in forehead_coords])
        
        # Get ear coordinates for lateral boundaries
        right_ear_coords = self.get_landmark_coordinates(landmarks, self.right_ear_landmarks, image_shape)
        left_ear_coords = self.get_landmark_coordinates(landmarks, self.left_ear_landmarks, image_shape)
        
        # Calculate face width at ear level
        if right_ear_coords and left_ear_coords:
            rightmost_x = max([x for x, y in right_ear_coords])
            leftmost_x = min([x for x, y in left_ear_coords])
            face_width = leftmost_x - rightmost_x
            
            # Add margin (approximately 2 fingers width)
            margin = int(face_width * ear_margin_ratio)
            right = rightmost_x + margin
            left = leftmost_x - margin
        else:
            # Fallback to face bounding box
            all_x = [landmark.x * w for landmark in landmarks]
            all_y = [landmark.y * h for landmark in landmarks]
            margin = int((max(all_x) - min(all_x)) * ear_margin_ratio)
            right = int(max(all_x)) + margin
            left = int(min(all_x)) - margin
        
        # Ensure boundaries are within image
        top = max(0, top)
        bottom = min(h, bottom)
        left = max(0, left)
        right = min(w, right)
        
        return top, bottom, left, right
    
    def process_image(self, image: np.ndarray, target_height: int = 4000, target_width: int = 6000) -> Optional[np.ndarray]:
        """
        Process image to extract and resize face region for eye segmentation model.
        
        Args:
            image: Input image (BGR format)
            target_height: Target height for the output image
            target_width: Target width for the output image
            
        Returns:
            Processed image ready for eye segmentation model, or None if no face detected
        """
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            print("No face detected in the image")
            return None
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # Calculate crop boundaries
        top, bottom, left, right = self.calculate_crop_boundaries(face_landmarks, image.shape)
        
        # Crop the image
        cropped = image[top:bottom, left:right]
        
        # Check if crop is valid
        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            print("Invalid crop dimensions")
            return None
        
        # Resize to target dimensions while maintaining aspect ratio
        # Calculate scaling factors
        scale_h = target_height / cropped.shape[0]
        scale_w = target_width / cropped.shape[1]
        
        # Use the smaller scale to ensure the entire face fits
        scale = min(scale_h, scale_w)
        
        # Calculate new dimensions
        new_h = int(cropped.shape[0] * scale)
        new_w = int(cropped.shape[1] * scale)
        
        # Resize the cropped image
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create output image with padding to reach target size
        output = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate padding
        pad_h = (target_height - new_h) // 2
        pad_w = (target_width - new_w) // 2
        
        # Place resized image in center
        output[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        return output
    
    def preprocess_for_model(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image for the eye segmentation model.
        Splits image in half, mirrors left side, and resizes both to 256x256.
        
        Args:
            image: Processed image from process_image() method
            
        Returns:
            Tuple of (right_half, mirrored_left_half), both 256x256
        """
        if image is None:
            return None, None
        
        h, w = image.shape[:2]
        mid_w = w // 2
        
        # Split image
        right_half = image[:, :mid_w]
        left_half = image[:, mid_w:]
        
        # Mirror the left half horizontally
        left_half_mirrored = cv2.flip(left_half, 1)
        
        # Resize both to 256x256
        right_resized = cv2.resize(right_half, (256, 256), interpolation=cv2.INTER_LINEAR)
        left_resized = cv2.resize(left_half_mirrored, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        return right_resized, left_resized
    
    def close(self):
        """Release MediaPipe resources"""
        self.face_mesh.close()
