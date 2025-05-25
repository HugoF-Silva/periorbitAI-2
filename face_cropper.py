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
        self.upper_lip_indices = [61, 146, 91, 181, 84, 17, 78, 95, 88, 178, 87, 14]
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
    
    # def get_hairline_y(self, img: np.ndarray, face_center_x: int) -> Tuple[Optional[int], bool]:
    #     """
    #     Get hairline Y coordinate at face center X.
    #     Returns: (y_coordinate, success_flag)
    #     """
    #     h, w = img.shape[:2]
        
    #     try:
    #         parsing_map = self.parse_face(img)
    #         hair_mask = (parsing_map == 17).astype(np.uint8)  # 17 = hair
            
    #         # Resize to original dimensions
    #         hair_full = cv2.resize(hair_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
    #         # Get hairline at face center
    #         face_center_x = min(w-1, max(0, int(face_center_x)))
    #         column = hair_full[:, face_center_x]
    #         ys = np.where(column)[0]
            
    #         if len(ys) > 0:
    #             return int(ys.min()), True
    #         else:
    #             return None, False
    #     except Exception as e:
    #         print(f"Hairline detection failed: {e}")
    #         return None, False

    def get_hairline_y_enhanced(self, img: np.ndarray, face_center_x: int) -> Tuple[Optional[int], bool]:
        """
        Enhanced hairline detection that finds the actual hair-forehead boundary.
        Uses multiple methods to ensure robust detection.
        """
        h, w = img.shape[:2]
        
        try:
            parsing_map = self.parse_face(img)
            
            # Get relevant masks
            hair_mask = (parsing_map == 17).astype(np.uint8)  # 17 = hair
            skin_mask = (parsing_map == 1).astype(np.uint8)   # 1 = skin/face
            forehead_mask = (parsing_map == 10).astype(np.uint8)  # 10 = forehead (if available)
            
            # Combine skin and forehead for better detection
            face_skin_mask = np.maximum(skin_mask, forehead_mask)
            
            # Resize to original dimensions
            hair_full = cv2.resize(hair_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            face_skin_full = cv2.resize(face_skin_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Ensure face_center_x is valid
            face_center_x = min(w-1, max(0, int(face_center_x)))
            
            # Method 1: Find skin-to-hair transition in a window around face center
            window_size = 50  # Look in a 50-pixel window
            x_start = max(0, face_center_x - window_size)
            x_end = min(w, face_center_x + window_size)
            
            best_hairline_y = None
            max_confidence = 0
            
            for x in range(x_start, x_end, 5):  # Check every 5 pixels
                # Scan from top to find transition
                for y in range(h - 1):
                    # Check if we're transitioning from skin/forehead to hair
                    if (face_skin_full[y, x] == 1 and hair_full[y + 1, x] == 1):
                        # Verify this is a substantial transition (not noise)
                        # Check if there's more hair below
                        hair_below = np.sum(hair_full[y+1:min(y+30, h), x])
                        skin_above = np.sum(face_skin_full[max(0, y-20):y, x])
                        
                        confidence = (hair_below + skin_above) / 50.0
                        
                        if confidence > max_confidence:
                            max_confidence = confidence
                            best_hairline_y = y + 1  # The first hair pixel
            
            # Method 2: If method 1 fails, use edge detection on hair mask
            if best_hairline_y is None:
                # Apply morphological operations to clean up the mask
                kernel = np.ones((5, 5), np.uint8)
                hair_cleaned = cv2.morphologyEx(hair_full, cv2.MORPH_CLOSE, kernel)
                hair_cleaned = cv2.morphologyEx(hair_cleaned, cv2.MORPH_OPEN, kernel)
                
                # Find edges
                edges = cv2.Canny(hair_cleaned * 255, 50, 150)
                
                # Look for horizontal edge near face center
                edge_column = edges[:, face_center_x]
                edge_indices = np.where(edge_column > 0)[0]
                
                if len(edge_indices) > 0:
                    # Find the topmost substantial edge
                    for y in edge_indices:
                        # Check if this edge represents a hairline (hair below, no hair above)
                        if (np.sum(hair_cleaned[y:min(y+20, h), face_center_x]) >= 15 and
                            np.sum(hair_cleaned[max(0, y-20):y, face_center_x]) <= 5):
                            best_hairline_y = y
                            break
            
            # Method 3: Statistical approach - find where hair density increases
            if best_hairline_y is None:
                # Calculate hair density in horizontal strips
                strip_height = 10
                densities = []
                
                for y in range(0, h - strip_height, 5):
                    strip = hair_full[y:y+strip_height, max(0, face_center_x-25):min(w, face_center_x+25)]
                    density = np.mean(strip)
                    densities.append((y, density))
                
                # Find where density significantly increases
                for i in range(1, len(densities)):
                    if densities[i][1] > 0.3 and densities[i][1] > densities[i-1][1] * 2:
                        best_hairline_y = densities[i][0]
                        break
            
            if best_hairline_y is not None:
                return int(best_hairline_y), True
            else:
                return None, False
                
        except Exception as e:
            print(f"Enhanced hairline detection failed: {e}")
            return None, False
    
    def get_upper_lip_y(self, landmarks, h: int) -> Tuple[Optional[int], bool]:
        """
        Get upper lip Y coordinate.
        Returns: (y_coordinate, success_flag)
        """
        try:
            ys = [landmarks[i].y * h for i in self.upper_lip_indices]
            return int(min(ys)), True
        except:
            return None, False
    
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