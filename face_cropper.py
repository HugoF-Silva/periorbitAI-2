import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple

class FaceCropper:
    """
    Face cropping utility using MediaPipe Face Mesh for eye region extraction.
    Designed to preprocess images for eye segmentation model that expects specific face regions.
    """
    
    def __init__(self, static_image_mode: bool = True,
                 max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5):
        """
        Initialize MediaPipe Face Mesh.
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
        self.upper_lip_top = [13, 14, 269, 270, 267, 271, 272]
        self.forehead_top = [10, 338, 297, 332, 284, 251, 389, 356, 454,
                             323, 361, 340, 346, 347, 348, 349, 350,
                             451, 452, 453]
        self.right_ear_landmarks = [234, 93, 132, 361, 340, 346, 347, 348, 349, 350]
        self.left_ear_landmarks  = [454, 323, 361, 340, 346, 347, 348,
                                    349, 350, 127, 162, 21, 54, 103, 67, 109]
    
    def get_landmark_coordinates(self, landmarks, indices, image_shape):
        """
        Extract pixel coordinates for specified landmark indices.
        """
        h, w = image_shape[:2]
        coords = []
        for idx in indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                coords.append((int(lm.x * w), int(lm.y * h)))
        return coords
    
    def calculate_crop_boundaries(self,
                                  landmarks,
                                  image_shape: Tuple[int, int, int],
                                  ear_margin_ratio: float = 0.15) -> Tuple[int,int,int,int]:
        """
        Calculate crop boundaries based on facial landmarks.
        Adds lateral margin of ~2 finger widths beyond each ear.
        """
        h, w = image_shape[:2]
        # Bottom: top of upper lip
        lower_coords = self.get_landmark_coordinates(landmarks,
                                                     self.upper_lip_top,
                                                     image_shape)
        bottom = min(y for x, y in lower_coords)

        # Top: highest forehead point
        forehead_coords = self.get_landmark_coordinates(landmarks,
                                                        self.forehead_top,
                                                        image_shape)
        top = min(y for x, y in forehead_coords)

        # Lateral: ears + margin
        right_ear = self.get_landmark_coordinates(landmarks,
                                                  self.right_ear_landmarks,
                                                  image_shape)
        left_ear  = self.get_landmark_coordinates(landmarks,
                                                  self.left_ear_landmarks,
                                                  image_shape)
        if right_ear and left_ear:
            rightmost_x = max(x for x, y in right_ear)
            leftmost_x  = min(x for x, y in left_ear)
            face_width = rightmost_x - leftmost_x
            margin = int(face_width * ear_margin_ratio)
            left = leftmost_x - margin
            right = rightmost_x + margin
        else:
            # fallback: full face bbox
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            left, right = int(min(xs)), int(max(xs))
            top_fb, bottom_fb = int(min(ys)), int(max(ys))
            face_width = right - left
            margin = int(face_width * ear_margin_ratio)
            left = left - margin
            right = right + margin
            top = top_fb  # keep original top if forehead missing
            bottom = bottom_fb  # fallback bottom if lips missing

        # Clamp to image
        top = max(0, top)
        bottom = min(h, bottom)
        left = max(0, left)
        right = min(w, right)
        return top, bottom, left, right
    
    def process_image(self,
                      image: np.ndarray,
                      target_height: int = 4000,
                      target_width: int = 6000) -> Optional[np.ndarray]:
        """
        Process image and crop to face region then resize.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            print("No face detected")
            return None
        lm = results.multi_face_landmarks[0].landmark
        top, bottom, left, right = self.calculate_crop_boundaries(lm, image.shape)
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            print("Invalid crop dimensions")
            return None
        # Resize with aspect ratio
        h_c, w_c = crop.shape[:2]
        scale = min(target_height/h_c, target_width/w_c)
        new_h, new_w = int(h_c * scale), int(w_c * scale)
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        output = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        pad_h = (target_height - new_h)//2
        pad_w = (target_width - new_w)//2
        output[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        return output
    
    def preprocess_for_model(self,
                             image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split and mirror for eye segmentation.
        """
        if image is None:
            return None, None
        h, w = image.shape[:2]
        mid = w // 2
        right = image[:, :mid]
        left = cv2.flip(image[:, mid:], 1)
        right_res = cv2.resize(right, (256,256), interpolation=cv2.INTER_LINEAR)
        left_res = cv2.resize(left, (256,256), interpolation=cv2.INTER_LINEAR)
        return right_res, left_res
    
    def close(self):
        self.face_mesh.close()
