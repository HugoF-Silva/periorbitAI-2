from face_parsing.model import BiSeNet  # make sure you have this!
from PIL import Image
import cv2
import mediapipe as mp
import numpy as np
import torch
import torchvision.transforms as transforms

# ----- 1. Load Face Parsing Model -----
def load_face_parsing_model(checkpoint_path, device='cuda'):
    n_classes = 19  # For CelebAMask-HQ
    net = BiSeNet(n_classes=n_classes)
    net.to(device)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.eval()
    return net

def parse_face(img: np.ndarray, model, device='cuda'):
    # img: numpy array, BGR
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((512, 512), Image.BILINEAR)
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    tensor = to_tensor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)  # shape: (512, 512)
    return parsing

def get_hairline_from_parsing(parsing_map, orig_shape):
    # parsing_map: (512,512), values=label indices
    hair_mask = (parsing_map == 17).astype(np.uint8)  # 17=hair in CelebAMask-HQ

    # Resize mask back to original image shape
    hair_mask_full = cv2.resize(hair_mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
    # Find top-most (minimum y) hair pixel for each column
    hairline_points = []
    for x in range(hair_mask_full.shape[1]):
        ys = np.where(hair_mask_full[:, x])[0]
        if len(ys) > 0:
            hairline_points.append((x, ys[0]))  # Top-most
    return hair_mask_full, hairline_points



class FaceCropperCompatible:
    """
    Modified FaceCropper that maintains compatibility with the original model training format.
    This ensures faces appear at the same scale as in the training data.
    """
    
    def __init__(self,
                 model_path=None,  # Optional - only if using hairline detection
                 static_image_mode=True,
                 max_num_faces=1,
                 min_detection_confidence=0.5,
                 device='cuda',
                 use_hairline_detection=False):
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
        # Face parsing model (optional)
        self.use_hairline_detection = use_hairline_detection and model_path is not None
        if self.use_hairline_detection:
            self.device = device
            self.parsing_model = self._load_face_parsing_model(model_path)
    
    def _load_face_parsing_model(self, checkpoint_path):
        """Load face parsing model if available."""
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.to(self.device)
        net.load_state_dict(torch.load(checkpoint_path, 
                                      map_location=self.device, 
                                      weights_only=False))
        net.eval()
        return net
    
    def process_image_compatible(self, image: np.ndarray, 
                                target_height: int = 4000,
                                target_width: int = 6000,
                                face_size_ratio: float = 0.6,
                                debug: bool = False):
        """
        Process image to match the original training data format.
        This maintains the expected face scale that your model was trained on.
        
        Args:
            image: Input BGR image
            target_height: Target output height (4000)
            target_width: Target output width (6000)
            face_size_ratio: What fraction of the output the face should occupy 
                           (0.6 = face height is 60% of image height)
            debug: Save debug visualizations
        """
        # First, check if image is already in the expected format
        if image.shape == (4000, 6000, 3):
            if debug:
                print("Image already 4000x6000, using directly")
            return image
        
        h, w = image.shape[:2]
        
        # Simple approach: just resize if aspect ratio is close enough
        aspect_original = w / h
        aspect_target = 6000 / 4000  # 1.5
        
        if abs(aspect_original - aspect_target) < 0.1:
            if debug:
                print("Aspect ratio close to target, simple resize")
            return cv2.resize(image, (6000, 4000), interpolation=cv2.INTER_LINEAR)
        
        # Use face detection to center properly
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            if debug:
                print("No face detected, using center crop")
            return self._center_crop_resize(image, target_height, target_width)
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Get face bounding box from landmarks
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        
        face_left = int(min(x_coords))
        face_right = int(max(x_coords))
        face_top = int(min(y_coords))
        face_bottom = int(max(y_coords))
        
        face_width = face_right - face_left
        face_height = face_bottom - face_top
        face_center_x = (face_left + face_right) // 2
        face_center_y = (face_top + face_bottom) // 2
        
        # CRITICAL: Calculate scale to maintain face at training size
        # If model was trained with faces occupying 60% of height:
        target_face_height_in_output = target_height * face_size_ratio
        scale = target_face_height_in_output / face_height
        
        # Calculate what region to crop to achieve this after scaling
        crop_width = int(target_width / scale)
        crop_height = int(target_height / scale)
        
        # Center crop on face
        crop_left = face_center_x - crop_width // 2
        crop_top = face_center_y - crop_height // 2
        crop_right = crop_left + crop_width
        crop_bottom = crop_top + crop_height
        
        # Adjust if crop goes outside image bounds
        if crop_left < 0:
            crop_right = min(crop_width, w)
            crop_left = 0
        if crop_right > w:
            crop_left = max(0, w - crop_width)
            crop_right = w
        if crop_top < 0:
            crop_bottom = min(crop_height, h)
            crop_top = 0
        if crop_bottom > h:
            crop_top = max(0, h - crop_height)
            crop_bottom = h
        
        # Crop and resize
        cropped = image[crop_top:crop_bottom, crop_left:crop_right]
        
        # If crop is smaller than expected, pad it
        if cropped.shape[0] < crop_height or cropped.shape[1] < crop_width:
            padded = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)
            y_offset = (crop_height - cropped.shape[0]) // 2
            x_offset = (crop_width - cropped.shape[1]) // 2
            padded[y_offset:y_offset+cropped.shape[0], 
                   x_offset:x_offset+cropped.shape[1]] = cropped
            cropped = padded
        
        # Final resize to target
        output = cv2.resize(cropped, (target_width, target_height), 
                           interpolation=cv2.INTER_LINEAR)
        
        if debug:
            print(f"Face scale in output: {face_size_ratio*100:.0f}% of image height")
            print(f"Original face height: {face_height}px")
            print(f"Face height in output: ~{int(target_face_height_in_output)}px")
            
            # Save debug visualization
            vis_img = image.copy()
            cv2.rectangle(vis_img, (face_left, face_top), 
                         (face_right, face_bottom), (0, 255, 0), 2)
            cv2.rectangle(vis_img, (crop_left, crop_top), 
                         (crop_right, crop_bottom), (255, 0, 0), 3)
            cv2.putText(vis_img, f"Face: {face_width}x{face_height}", 
                       (face_left, face_top-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 255, 0), 2)
            cv2.imwrite('debug_face_scale_matching.jpg', vis_img)
        
        return output
    
    def _center_crop_resize(self, image, target_height, target_width):
        """Fallback: center crop and resize maintaining aspect ratio."""
        h, w = image.shape[:2]
        
        # Scale to fill at least one dimension
        scale = max(target_height / h, target_width / w)
        
        scaled_h = int(h * scale)
        scaled_w = int(w * scale)
        scaled = cv2.resize(image, (scaled_w, scaled_h), 
                           interpolation=cv2.INTER_LINEAR)
        
        # Center crop
        y_offset = max(0, (scaled_h - target_height) // 2)
        x_offset = max(0, (scaled_w - target_width) // 2)
        
        output = scaled[y_offset:y_offset+target_height, 
                       x_offset:x_offset+target_width]
        
        # Pad if needed
        if output.shape[0] < target_height or output.shape[1] < target_width:
            padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            y_off = (target_height - output.shape[0]) // 2
            x_off = (target_width - output.shape[1]) // 2
            padded[y_off:y_off+output.shape[0], 
                   x_off:x_off+output.shape[1]] = output
            output = padded
        
        return output
    
    def close(self):
        """Release resources."""
        self.face_mesh.close()


# Simple alternative if you don't want the full class
def simple_compatible_resize(image, face_size_ratio=0.6):
    """
    Simpler version - just scale and center the image appropriately.
    
    Args:
        image: Input image
        face_size_ratio: Expected ratio of face height to image height (0.6 = 60%)
    """
    if image.shape == (4000, 6000, 3):
        return image
    
    h, w = image.shape[:2]
    
    # Assume face takes up most of the image height
    # Scale so that current image height becomes face_size_ratio of output
    scale = (4000 * face_size_ratio) / h
    
    # But don't exceed width constraints
    if w * scale > 6000:
        scale = 6000 / w
    
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Center in 4000x6000
    output = np.zeros((4000, 6000, 3), dtype=np.uint8)
    y_offset = (4000 - new_h) // 2
    x_offset = (6000 - new_w) // 2
    output[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return output


# Usage in your pipeline:
if __name__ == "__main__":
    # Option 1: Use the full class
    face_cropper = FaceCropperCompatible(use_hairline_detection=False)
    img = cv2.imread("test.jpg")
    processed = face_cropper.process_image_compatible(img, face_size_ratio=0.6, debug=True)
    
    # Option 2: Use the simple function
    processed_simple = simple_compatible_resize(img, face_size_ratio=0.6)
    
    # Test with your model
    face_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    # seg_arr = ni.apply_net(net, face_rgb, argmax_output=True)[0,:,:]