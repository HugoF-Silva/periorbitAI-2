from face_parsing.model import BiSeNet  # make sure you have this!
from PIL import Image
import cv2
import mediapipe as mp
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import Tuple, Optional

class AspectRatioFaceCropper:
    """
    Face cropper that maintains 6000×4000 aspect ratio with intelligent boundary handling.
    """
    
    def __init__(self,
                 parsing_model_path: str,
                 static_image_mode: bool = True,
                 max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 device: str = 'cuda'):
        
        # Target dimensions and aspect ratio
        self.target_width = 6000
        self.target_height = 4000
        self.aspect_ratio = self.target_width / self.target_height  # 1.5
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
        # Load face parsing model
        self.device = device
        self.parsing_model = self._load_parsing_model(parsing_model_path)
        
        # Landmark indices
        self.upper_lip_indices = [13, 269, 272]
        self.left_eye_outer = 33
        self.right_eye_outer = 263
    
    def _load_parsing_model(self, checkpoint_path: str):
        """Load BiSeNet model."""
        from face_parsing.model import BiSeNet
        net = BiSeNet(n_classes=19)
        net.to(self.device)
        net.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=False))
        net.eval()
        return net
    
    def parse_face(self, img: np.ndarray):
        """Run face parsing to get segmentation map."""
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image = image.resize((512, 512), Image.BILINEAR)
        
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        tensor = to_tensor(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.parsing_model(tensor)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
        
        return parsing
    
    def get_two_finger_margin(self, landmarks, w: int, h: int, finger_factor: float = 0.4) -> int:
        """Calculate margin based on inter-ocular distance."""
        try:
            x_left = landmarks[self.left_eye_outer].x * w
            x_right = landmarks[self.right_eye_outer].x * w
            iod = abs(x_right - x_left)
            return int(finger_factor * iod)  # ~2 fingers total
        except:
            # Fallback to 10% of image width
            return int(w * 0.1)
    
    def get_hairline_y(self, img: np.ndarray, face_center_x: int) -> Tuple[Optional[int], bool]:
        """
        Get hairline Y coordinate at face center X.
        Returns the *lowest* hair pixel (max y) in that column, not the highest.
        """
        h, w = img.shape[:2]
        
        try:
            parsing_map = self.parse_face(img)
            hair_mask = (parsing_map == 17).astype(np.uint8)  # 17 = hair
            
            # Resize to original dimensions
            hair_full = cv2.resize(hair_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Clamp face_center_x into image bounds
            x = min(w - 1, max(0, int(face_center_x)))
            column = hair_full[:, x]
            ys = np.where(column)[0]
            
            if len(ys) > 0:
                # <-- here we pick the *lowest* hair pixel
                return int(ys.max()), True
            else:
                return None, False
        except Exception as e:
            print(f"Hairline detection failed: {e}")
            return None, False

    def get_upper_lip_y(self, landmarks, h: int) -> Tuple[Optional[int], bool]:
        """
        Get upper lip Y coordinate - uses outer upper edge only.
        Returns: (y_coordinate, success_flag)
        """
        try:
            # Use upper lip top edge landmarks (outer contour)
            # These are the landmarks that form the upper boundary of the upper lip
            # upper_edge_indices = [13, 312, 311, 310, 415, 308, 324, 318, 14, 87, 178, 88, 95]
            upper_edge_indices = [13, 269, 272]
            
            ys = []
            for idx in upper_edge_indices:
                if idx < len(landmarks):
                    ys.append(landmarks[idx].y * h)
            
            if ys:
                # Get the minimum y (topmost point) of upper lip outer edge
                return int(min(ys)), True
            else:
                # Fallback to original indices if new ones fail
                ys = [landmarks[i].y * h for i in self.upper_lip_indices if i < len(landmarks)]
                if ys:
                    return int(min(ys)), True
                return None, False
        except Exception as e:
            print(f"Upper lip detection failed: {e}")
            return None, False
        
    def process_image(self, img: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
        """
        Process image with aspect ratio constraints and boundary rules.
        
        Args:
            img: Input BGR image
            debug: Save debug visualizations
            
        Returns:
            Processed image (6000×4000) or None if should be skipped
        """
        h, w = img.shape[:2]
        
        # 1. Run MediaPipe face detection
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            print("No face detected - skipping image")
            return None
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Get face center for hairline detection
        xs_all = [lm.x * w for lm in landmarks]
        ys_all = [lm.y * h for lm in landmarks]
        face_center_x = int(np.mean(xs_all))
        
        # 2. Detect boundaries
        hairline_y, hairline_detected = self.get_hairline_y(img, face_center_x)
        upper_lip_y, lip_detected = self.get_upper_lip_y(landmarks, h)
        
        # 3. Apply boundary rules
        
        # Rule 3.3: Skip if BOTH landmarks not detected
        if not hairline_detected and not lip_detected:
            print("Both hairline and upper lip not detected - skipping image")
            return None
        
        # Rule 3.1: Use image top if hairline not detected
        if not hairline_detected:
            print("Hairline not detected - using image top boundary")
            top = 0
        else:
            top = hairline_y
        
        # Rule 3.2: Use image bottom if upper lip not detected
        if not lip_detected:
            print("Upper lip not detected - using image bottom boundary")
            bottom = h
        else:
            bottom = upper_lip_y
        
        # Validate vertical boundaries
        if bottom <= top:
            print(f"Invalid boundaries: top={top}, bottom={bottom} - skipping image")
            return None
        
        # 4. Calculate required dimensions
        crop_height = bottom - top
        print(f"upper lip - hairline = {bottom} - {top}")
        print(f"crop_height: {crop_height}")

        required_width = int(crop_height * self.aspect_ratio)
        
        # 5. Calculate lateral boundaries with 2-finger margin
        face_min_x = min(xs_all)
        face_max_x = max(xs_all)
        margin = self.get_two_finger_margin(landmarks, w, h)
        
        # Face width with margins
        face_width_with_margin = int((face_max_x - face_min_x) + 2 * margin)
        
        # 6. Determine crop/stretch strategy
        if w >= required_width:
            # Image is wide enough - center the crop
            crop_width = required_width
            center_x = int((face_min_x + face_max_x) / 2)
            left = center_x - crop_width // 2
            
            # Ensure crop stays within image bounds
            left = max(0, min(left, w - crop_width))
            right = left + crop_width
            
            # Crop the image
            cropped = img[top:bottom, left:right]
            stretch_applied = False
            
        else:
            # Image too narrow - use full width and stretch
            print(f"Image width {w} < required {required_width} - will stretch")
            left = 0
            right = w
            crop_width = w
            
            # Crop with full width
            cropped = img[top:bottom, left:right]
            
            # Stretch width to maintain aspect ratio
            stretch_width = required_width
            stretch_height = crop_height
            cropped = cv2.resize(cropped, (stretch_width, stretch_height), 
                               interpolation=cv2.INTER_LINEAR)
            stretch_applied = True
        
        # 7. Final resize to target dimensions
        final = cv2.resize(cropped, (self.target_width, self.target_height), 
                          interpolation=cv2.INTER_LINEAR)
        
        # 8. Debug visualization
        if debug:
            self._save_debug_visualization(
                img, top, bottom, left, right, 
                hairline_detected, lip_detected, 
                stretch_applied, face_center_x
            )
        
        return final
    
    def _save_debug_visualization(self, img, top, bottom, left, right, 
                                  hairline_ok, lip_ok, stretched, face_x):
        """Save debug visualization."""
        vis = img.copy()
        h, w = img.shape[:2]
        
        # Draw crop rectangle
        color = (0, 255, 0) if not stretched else (0, 165, 255)  # Green or Orange
        cv2.rectangle(vis, (left, top), (right, bottom), color, 3)
        
        # Draw detected lines
        if hairline_ok:
            cv2.line(vis, (0, top), (w, top), (0, 255, 0), 2)
            cv2.putText(vis, "Hairline", (10, top-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "Hairline: Using image top", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if lip_ok:
            cv2.line(vis, (0, bottom), (w, bottom), (0, 255, 0), 2)
            cv2.putText(vis, "Upper Lip", (10, bottom+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "Upper Lip: Using image bottom", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw face center line
        cv2.line(vis, (face_x, 0), (face_x, h), (255, 255, 0), 1)
        
        # Add status text
        status = "Stretched" if stretched else "Cropped"
        cv2.putText(vis, f"Status: {status}", (10, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imwrite('debug_face_crop.jpg', vis)
    
    def preprocess_for_model(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split and mirror for eye segmentation model."""
        if image is None:
            return None, None
        
        h, w = image.shape[:2]
        mid = w // 2
        
        # Split
        right = image[:, :mid]
        left = image[:, mid:]
        
        # Mirror left
        left_mirrored = cv2.flip(left, 1)
        
        # Resize to 256×256
        right_res = cv2.resize(right, (256, 256), interpolation=cv2.INTER_LINEAR)
        left_res = cv2.resize(left_mirrored, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        return right_res, left_res
    
    def close(self):
        """Release resources."""
        self.face_mesh.close()