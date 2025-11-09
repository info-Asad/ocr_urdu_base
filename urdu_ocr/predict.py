"""
Prediction/Inference Script for Urdu OCR
Use trained model to recognize Urdu text from images
"""
import os
import sys
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import glob

# Try to import matplotlib, make it optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization disabled.")

import config
from model import UrduOCRModel
from data_preprocessing import ImagePreprocessor, TextPreprocessor
import utils


class UrduOCRPredictor:
    """Predictor class for Urdu OCR inference"""
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize preprocessors
        self.image_preprocessor = ImagePreprocessor()
        self.text_preprocessor = TextPreprocessor()
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = UrduOCRModel()
        self.model, self.checkpoint = utils.load_model_checkpoint(
            self.model, model_path, self.device
        )
        self.model.eval()
        
        print("Model loaded successfully!\n")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction
        
        Args:
            image_path: Path to image file
        
        Returns:
            image_tensor: Preprocessed image tensor
            original_image: Original image for visualization
        """
        # Load image
        original_image = self.image_preprocessor.load_image(image_path)
        if original_image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Preprocess
        image = self.image_preprocessor.resize_image(original_image, maintain_aspect=True)
        image = self.image_preprocessor.normalize_image(image)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        return image_tensor, original_image
    
    def predict_single(self, image_path, visualize=False):
        """
        Predict text from a single image
        
        Args:
            image_path: Path to image file
            visualize: Whether to display the image with prediction
        
        Returns:
            predicted_text: Recognized text
        """
        # Preprocess image
        image_tensor, original_image = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model.predict(image_tensor)
        
        # Decode prediction
        pred_indices = predictions[0].cpu().numpy()
        predicted_text = self.text_preprocessor.decode_text(pred_indices)
        
        # Visualize if requested
        if visualize:
            self._visualize_prediction(original_image, predicted_text, image_path)
        
        return predicted_text
    
    def predict_batch(self, image_paths, batch_size=32):
        """
        Predict text from multiple images
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
        
        Returns:
            predictions: List of predicted texts
        """
        predictions = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            
            # Load and preprocess batch
            for path in batch_paths:
                try:
                    tensor, _ = self.preprocess_image(path)
                    batch_tensors.append(tensor)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    predictions.append("")
                    continue
            
            if not batch_tensors:
                continue
            
            # Stack batch
            batch = torch.cat(batch_tensors, dim=0).to(self.device)
            
            # Predict
            with torch.no_grad():
                batch_predictions = self.model.predict(batch)
            
            # Decode predictions
            for j in range(batch_predictions.size(0)):
                pred_indices = batch_predictions[j].cpu().numpy()
                predicted_text = self.text_preprocessor.decode_text(pred_indices)
                predictions.append(predicted_text)
        
        return predictions
    
    def predict_directory(self, directory, output_file=None, visualize=False):
        """
        Predict text from all images in a directory
        
        Args:
            directory: Path to directory containing images
            output_file: Path to save predictions (optional)
            visualize: Whether to display predictions
        
        Returns:
            results: Dictionary mapping image names to predictions
        """
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(directory, ext)))
            image_paths.extend(glob.glob(os.path.join(directory, ext.upper())))
        
        if not image_paths:
            print(f"No images found in {directory}")
            return {}
        
        print(f"Found {len(image_paths)} images")
        print("Processing...\n")
        
        # Predict
        results = {}
        for i, image_path in enumerate(image_paths, 1):
            image_name = os.path.basename(image_path)
            
            try:
                predicted_text = self.predict_single(image_path, visualize=False)
                results[image_name] = predicted_text
                
                print(f"[{i}/{len(image_paths)}] {image_name}")
                print(f"  → {predicted_text}\n")
                
            except Exception as e:
                print(f"Error processing {image_name}: {e}\n")
                results[image_name] = ""
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("Image Name\tPredicted Text\n")
                f.write("="*80 + "\n")
                for image_name, text in results.items():
                    f.write(f"{image_name}\t{text}\n")
            print(f"\nPredictions saved to {output_file}")
        
        return results
    
    def _visualize_prediction(self, image, predicted_text, image_path):
        """Visualize image with prediction"""
        if not MATPLOTLIB_AVAILABLE:
            print(f"Visualization skipped (matplotlib not available)")
            print(f"Image: {os.path.basename(image_path)}")
            print(f"Predicted: {predicted_text}")
            return
            
        plt.figure(figsize=(12, 4))
        plt.imshow(image)
        plt.title(f"Predicted: {predicted_text}\nImage: {os.path.basename(image_path)}", 
                 fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def interactive_mode(self):
        """Interactive mode for predicting single images"""
        print("\n" + "="*70)
        print("INTERACTIVE PREDICTION MODE")
        print("="*70)
        print("Enter image path (or 'quit' to exit)")
        print()
        
        while True:
            image_path = input("Image path: ").strip()
            
            if image_path.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break
            
            if not os.path.exists(image_path):
                print(f"Error: File not found: {image_path}\n")
                continue
            
            try:
                predicted_text = self.predict_single(image_path, visualize=True)
                print(f"\nPredicted text: {predicted_text}")
                print("-" * 70 + "\n")
            
            except Exception as e:
                print(f"Error: {e}\n")


def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description='Urdu OCR Prediction Script')
    
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image for prediction')
    parser.add_argument('--directory', type=str, default=None,
                       help='Path to directory containing images')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save predictions')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize predictions')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please train a model first using train.py")
        sys.exit(1)
    
    # Initialize predictor
    predictor = UrduOCRPredictor(args.model, device=args.device)
    
    # Run prediction based on mode
    if args.interactive:
        predictor.interactive_mode()
    
    elif args.image:
        print(f"Predicting text from: {args.image}\n")
        predicted_text = predictor.predict_single(args.image, visualize=args.visualize)
        print(f"Predicted text: {predicted_text}")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(f"Image: {args.image}\n")
                f.write(f"Predicted: {predicted_text}\n")
            print(f"Prediction saved to {args.output}")
    
    elif args.directory:
        print(f"Predicting text from all images in: {args.directory}\n")
        results = predictor.predict_directory(
            args.directory, 
            output_file=args.output,
            visualize=args.visualize
        )
        print(f"\nProcessed {len(results)} images")
    
    else:
        print("No input specified. Use one of the following:")
        print("  --image <path>       : Predict single image")
        print("  --directory <path>   : Predict all images in directory")
        print("  --interactive        : Run in interactive mode")
        print("\nFor help: python predict.py --help")


if __name__ == "__main__":
    main()
