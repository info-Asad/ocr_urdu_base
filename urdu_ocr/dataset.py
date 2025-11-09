"""
Dataset Module for Urdu OCR
Handles loading and processing of training and validation data
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from tqdm import tqdm
import config
from data_preprocessing import ImagePreprocessor, TextPreprocessor, DataAugmentor


class UrduOCRDataset(Dataset):
    """
    Custom Dataset for Urdu OCR
    
    Expected data structure:
    data_dir/
        ├── images/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        └── labels.txt  (format: image_name.jpg<TAB>urdu_text)
    """
    
    def __init__(self, data_dir, is_training=True, augment=True):
        """
        Args:
            data_dir: Path to data directory
            is_training: Whether this is training data
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        self.labels_file = os.path.join(data_dir, 'labels.txt')
        self.is_training = is_training
        self.augment = augment and is_training
        
        # Initialize preprocessors
        self.image_preprocessor = ImagePreprocessor()
        self.text_preprocessor = TextPreprocessor()
        self.augmentor = DataAugmentor()
        
        # Get transforms
        if self.augment:
            self.transforms = self.augmentor.get_train_transforms()
        else:
            self.transforms = self.augmentor.get_val_transforms()
        
        # Load dataset
        self.samples = self._load_dataset()
        
        print(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def _load_dataset(self):
        """Load image paths and labels from labels file"""
        samples = []
        
        if not os.path.exists(self.labels_file):
            print(f"Warning: Labels file not found at {self.labels_file}")
            return samples
        
        with open(self.labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse line: image_name.jpg<TAB>text
                parts = line.split('\t')
                if len(parts) != 2:
                    print(f"Warning: Invalid line format: {line}")
                    continue
                
                image_name, text = parts
                image_path = os.path.join(self.images_dir, image_name)
                
                # Verify image exists
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    continue
                
                # Normalize text
                normalized_text = self.text_preprocessor.normalize_text(text)
                if not normalized_text:
                    print(f"Warning: Empty text after normalization for {image_name}")
                    continue
                
                samples.append({
                    'image_path': image_path,
                    'text': normalized_text,
                    'image_name': image_name
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        sample = self.samples[idx]
        
        # Load and preprocess image
        image = self.image_preprocessor.load_image(sample['image_path'])
        if image is None:
            # Return a blank image if loading fails
            image = np.ones((config.IMG_HEIGHT, config.IMG_WIDTH, 3), dtype=np.uint8) * 255
        
        # Resize image
        image = self.image_preprocessor.resize_image(image, maintain_aspect=True)
        
        # Apply augmentation
        if self.augment:
            image = self.augmentor.apply_transforms(image, self.transforms)
        
        # Normalize
        image = self.image_preprocessor.normalize_image(image)
        
        # Convert to tensor (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Encode text
        text = sample['text']
        encoded_text = self.text_preprocessor.encode_text(text)
        
        return {
            'image': image,
            'text': text,
            'encoded_text': encoded_text,
            'text_length': len(encoded_text),
            'image_name': sample['image_name']
        }


def collate_fn(batch):
    """
    Custom collate function for batching
    Handles variable length text sequences
    """
    images = []
    texts = []
    encoded_texts = []
    text_lengths = []
    image_names = []
    
    for sample in batch:
        images.append(sample['image'])
        texts.append(sample['text'])
        encoded_texts.append(sample['encoded_text'])
        text_lengths.append(sample['text_length'])
        image_names.append(sample['image_name'])
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    # Pad encoded texts to same length
    max_length = max(text_lengths)
    padded_texts = []
    for encoded_text in encoded_texts:
        padded = encoded_text + [config.BLANK_INDEX] * (max_length - len(encoded_text))
        padded_texts.append(padded)
    
    padded_texts = torch.LongTensor(padded_texts)
    text_lengths = torch.LongTensor(text_lengths)
    
    return {
        'images': images,
        'texts': texts,
        'encoded_texts': padded_texts,
        'text_lengths': text_lengths,
        'image_names': image_names
    }


def create_data_loaders(train_dir, val_dir, batch_size=config.BATCH_SIZE, num_workers=4):
    """
    Create training and validation data loaders
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        batch_size: Batch size
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = UrduOCRDataset(train_dir, is_training=True, augment=True)
    val_dataset = UrduOCRDataset(val_dir, is_training=False, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


def verify_dataset(data_dir, num_samples=5):
    """
    Verify dataset by loading and displaying samples
    
    Args:
        data_dir: Path to data directory
        num_samples: Number of samples to display
    """
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    
    dataset = UrduOCRDataset(data_dir, is_training=False, augment=False)
    
    if len(dataset) == 0:
        print("Dataset is empty!")
        return
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Displaying {min(num_samples, len(dataset))} samples...\n")
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        
        # Convert image tensor back to numpy
        image = sample['image'].permute(1, 2, 0).numpy()
        text = sample['text']
        
        axes[i].imshow(image)
        axes[i].set_title(f"Text: {text}", fontsize=12)
        axes[i].axis('off')
        
        print(f"Sample {i+1}:")
        print(f"  Image: {sample['image_name']}")
        print(f"  Text: {text}")
        print(f"  Encoded length: {sample['text_length']}")
        print()
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'dataset_samples.png'")
    plt.show()


def analyze_dataset(data_dir):
    """
    Analyze dataset statistics
    
    Args:
        data_dir: Path to data directory
    """
    dataset = UrduOCRDataset(data_dir, is_training=False, augment=False)
    
    if len(dataset) == 0:
        print("Dataset is empty!")
        return
    
    print(f"\n{'='*50}")
    print(f"DATASET ANALYSIS: {data_dir}")
    print(f"{'='*50}")
    
    # Basic statistics
    print(f"\nTotal samples: {len(dataset)}")
    
    # Text length statistics
    text_lengths = [sample['text_length'] for sample in dataset]
    print(f"\nText Length Statistics:")
    print(f"  Min: {min(text_lengths)}")
    print(f"  Max: {max(text_lengths)}")
    print(f"  Mean: {np.mean(text_lengths):.2f}")
    print(f"  Median: {np.median(text_lengths):.2f}")
    
    # Character frequency
    char_freq = {}
    for sample in dataset:
        for char in sample['text']:
            char_freq[char] = char_freq.get(char, 0) + 1
    
    print(f"\nCharacter Frequency (Top 20):")
    sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
    for char, freq in sorted_chars[:20]:
        print(f"  '{char}': {freq}")
    
    print(f"\n{'='*50}\n")


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset module...")
    
    # Check if data directories exist
    if os.path.exists(config.TRAIN_DIR):
        print(f"\nAnalyzing training data from: {config.TRAIN_DIR}")
        analyze_dataset(config.TRAIN_DIR)
    else:
        print(f"\nTraining directory not found: {config.TRAIN_DIR}")
        print("Please prepare your dataset according to DATASET_GUIDE.md")
    
    if os.path.exists(config.VAL_DIR):
        print(f"\nAnalyzing validation data from: {config.VAL_DIR}")
        analyze_dataset(config.VAL_DIR)
    else:
        print(f"\nValidation directory not found: {config.VAL_DIR}")
