#!/usr/bin/env python3
"""
Simple Urdu OCR Prediction Test Script
Use this to test your trained model after 1 epoch
"""

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
import argparse

# Add urdu_ocr to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'urdu_ocr'))

import config
from model import UrduOCRModel
from data_preprocessing import ImagePreprocessor, TextPreprocessor

def load_model(model_path):
    """Load the trained model"""
    print(f"Loading model from: {model_path}")
    
    # Create model
    model = UrduOCRModel(config.NUM_CLASSES)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully!")
        if 'best_cer' in checkpoint:
            print(f"Model CER: {checkpoint['best_cer']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("Model loaded (legacy format)")
    
    model.eval()
    return model

def preprocess_image(image_path):
    """Preprocess image for prediction"""
    preprocessor = ImagePreprocessor()
    
    # Load image
    image = preprocessor.load_image(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize maintaining aspect ratio
    image = preprocessor.resize_image(image, maintain_aspect=True)
    
    # Normalize
    image = preprocessor.normalize_image(image)
    
    # Convert to tensor (H, W, C) -> (C, H, W)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def predict_text(model, image_tensor):
    """Predict text from image"""
    text_preprocessor = TextPreprocessor()
    
    with torch.no_grad():
        # Get model output
        log_probs, sequence_lengths = model(image_tensor)
        
        # Convert to predictions (argmax)
        # log_probs is (seq_len, batch_size, num_classes) -> (batch_size, seq_len, num_classes)
        logits = log_probs.permute(1, 0, 2)
        predictions = torch.argmax(logits, dim=2)
        
        # Decode first (and only) prediction
        pred_indices = predictions[0].cpu().numpy()
        predicted_text = text_preprocessor.decode_text(pred_indices)
        
        return predicted_text

def test_model_simple():
    """Simple test with validation images"""
    model_path = "models/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please wait for training to complete at least 1 epoch.")
        return
    
    print("🤖 Urdu OCR Model Test")
    print("=" * 50)
    
    # Load model
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test with some validation images
    validation_dir = "data/validation/images"
    test_images = []
    
    # Get first few validation images
    if os.path.exists(validation_dir):
        for i in range(800, 805):  # Test with img_00800 to img_00804
            img_path = os.path.join(validation_dir, f"img_{i:05d}.png")
            if os.path.exists(img_path):
                test_images.append(img_path)
    
    if not test_images:
        print("❌ No validation images found for testing")
        return
    
    print(f"\n🧪 Testing with {len(test_images)} validation images:")
    print("-" * 50)
    
    for i, img_path in enumerate(test_images, 1):
        print(f"\n{i}. Testing: {os.path.basename(img_path)}")
        
        try:
            # Preprocess image
            image_tensor = preprocess_image(img_path)
            
            # Predict
            predicted_text = predict_text(model, image_tensor)
            
            print(f"   Predicted: '{predicted_text}'")
            
            # Try to get ground truth
            img_name = os.path.basename(img_path)
            labels_file = "data/validation/labels.txt"
            if os.path.exists(labels_file):
                with open(labels_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip().startswith(img_name):
                            _, actual_text = line.strip().split('\t', 1)
                            print(f"   Actual:    '{actual_text}'")
                            break
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ Testing complete!")

def test_custom_image(image_path):
    """Test with a custom image"""
    model_path = "models/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"🤖 Testing custom image: {image_path}")
    print("=" * 50)
    
    # Load model
    model = load_model(model_path)
    
    # Predict
    try:
        image_tensor = preprocess_image(image_path)
        predicted_text = predict_text(model, image_tensor)
        
        print(f"\n📝 Predicted Text:")
        print(f"'{predicted_text}'")
        print(f"\n✅ Prediction complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Test Urdu OCR Model')
    parser.add_argument('--image', type=str, help='Path to image file to test')
    parser.add_argument('--simple', action='store_true', help='Run simple test with validation images')
    
    args = parser.parse_args()
    
    if args.image:
        test_custom_image(args.image)
    else:
        test_model_simple()

if __name__ == "__main__":
    main()