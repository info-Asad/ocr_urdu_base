"""
Data Preprocessing Module for Urdu OCR
Handles image preprocessing, text normalization, and augmentation
"""
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config


class ImagePreprocessor:
    """Handles all image preprocessing operations"""
    
    def __init__(self, target_height=config.IMG_HEIGHT, target_width=config.IMG_WIDTH):
        self.target_height = target_height
        self.target_width = target_width
        
    def load_image(self, image_path):
        """Load image from file path"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def resize_image(self, image, maintain_aspect=True):
        """
        Resize image to target dimensions
        If maintain_aspect=True, pad the image to maintain aspect ratio
        """
        h, w = image.shape[:2]
        
        if maintain_aspect:
            # Calculate scaling factor
            scale = min(self.target_width / w, self.target_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create canvas and paste resized image
            canvas = np.ones((self.target_height, self.target_width, 3), dtype=np.uint8) * 255
            
            # Center the image
            y_offset = (self.target_height - new_h) // 2
            x_offset = (self.target_width - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas
        else:
            return cv2.resize(image, (self.target_width, self.target_height), 
                            interpolation=cv2.INTER_AREA)
    
    def normalize_image(self, image):
        """Normalize image to [0, 1] range"""
        return image.astype(np.float32) / 255.0
    
    def denormalize_image(self, image):
        """Convert normalized image back to [0, 255] range"""
        return (image * 255.0).astype(np.uint8)
    
    def preprocess(self, image_path):
        """Complete preprocessing pipeline"""
        image = self.load_image(image_path)
        if image is None:
            return None
        
        image = self.resize_image(image, maintain_aspect=True)
        image = self.normalize_image(image)
        
        return image
    
    def enhance_image(self, image):
        """Apply image enhancement techniques"""
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        enhanced = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        return enhanced


class TextPreprocessor:
    """Handles Urdu text preprocessing and normalization"""
    
    def __init__(self):
        self.char_to_idx = config.CHAR_TO_IDX
        self.idx_to_char = config.IDX_TO_CHAR
    
    def normalize_text(self, text):
        """Normalize Urdu text"""
        if text is None:
            return ""
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        # Remove characters not in vocabulary
        normalized = ''.join([char for char in text if char in self.char_to_idx])
        
        return normalized
    
    def encode_text(self, text):
        """Convert text to sequence of indices"""
        normalized = self.normalize_text(text)
        encoded = [self.char_to_idx.get(char, config.BLANK_INDEX) for char in normalized]
        return encoded
    
    def decode_text(self, indices):
        """Convert sequence of indices back to text"""
        # Remove blank tokens and consecutive duplicates (CTC decoding)
        decoded_chars = []
        prev_idx = None
        
        for idx in indices:
            if idx != config.BLANK_INDEX and idx != prev_idx:
                if idx in self.idx_to_char:
                    decoded_chars.append(self.idx_to_char[idx])
            prev_idx = idx
        
        return ''.join(decoded_chars)
    
    def get_max_length(self, texts):
        """Get maximum text length in a list"""
        return max([len(self.encode_text(text)) for text in texts])


class DataAugmentor:
    """Handles data augmentation for training"""
    
    def __init__(self, augmentation_config=config.AUGMENTATION):
        self.aug_config = augmentation_config
        
    def get_train_transforms(self):
        """Get training augmentation pipeline"""
        return A.Compose([
            A.OneOf([
                A.GaussNoise(var_limit=self.aug_config['noise_var'], p=0.3),
                A.GaussianBlur(blur_limit=self.aug_config['blur_limit'], p=0.3),
                A.MotionBlur(blur_limit=self.aug_config['blur_limit'], p=0.3),
            ], p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=self.aug_config['random_brightness'],
                contrast_limit=self.aug_config['random_contrast'],
                p=0.5
            ),
            A.RandomGamma(
                gamma_limit=self.aug_config['random_gamma'],
                p=0.3
            ),
            A.Rotate(
                limit=self.aug_config['rotation_limit'],
                border_mode=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
                p=0.3
            ),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                A.GridDistortion(p=0.3),
            ], p=0.3),
        ])
    
    def get_val_transforms(self):
        """Get validation transforms (no augmentation)"""
        return A.Compose([])
    
    def apply_transforms(self, image, transforms):
        """Apply transforms to image"""
        if transforms is not None:
            augmented = transforms(image=image)
            return augmented['image']
        return image


# Utility functions
def visualize_preprocessing(image_path, preprocessor):
    """Visualize preprocessing steps"""
    import matplotlib.pyplot as plt
    
    # Load original
    original = preprocessor.load_image(image_path)
    
    # Preprocessing steps
    resized = preprocessor.resize_image(original)
    normalized = preprocessor.normalize_image(resized)
    
    # Display
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(resized)
    axes[1].set_title('Resized')
    axes[1].axis('off')
    
    axes[2].imshow(normalized)
    axes[2].set_title('Normalized')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test preprocessing
    print("Image Preprocessor initialized")
    print(f"Target size: {config.IMG_WIDTH}x{config.IMG_HEIGHT}")
    print(f"Vocabulary size: {config.NUM_CLASSES}")
    
    # Test text preprocessing
    text_preprocessor = TextPreprocessor()
    sample_text = "یہ ایک ٹیسٹ ہے"
    encoded = text_preprocessor.encode_text(sample_text)
    decoded = text_preprocessor.decode_text(encoded)
    print(f"\nOriginal: {sample_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
