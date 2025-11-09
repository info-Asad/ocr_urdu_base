#!/usr/bin/env python3
"""
Minimal Urdu OCR Training Script - No checkpoints until end
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add urdu_ocr to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'urdu_ocr'))

import config
from model import UrduOCRModel
from dataset import UrduOCRDataset, create_data_loaders
from data_preprocessing import ImagePreprocessor, TextPreprocessor
from utils import calculate_cer

def train_epoch(model, dataloader, criterion, optimizer, device, text_preprocessor):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        images = batch['images'].to(device)
        texts = batch['texts']
        
        # Encode texts
        encoded_texts = []
        text_lengths = []
        for text in texts:
            encoded = text_preprocessor.encode_text(text)
            encoded_texts.append(encoded)
            text_lengths.append(len(encoded))
        
        # Pad sequences
        max_len = max(text_lengths)
        padded_texts = []
        for encoded in encoded_texts:
            padded = encoded + [0] * (max_len - len(encoded))
            padded_texts.append(padded)
        
        targets = torch.tensor(padded_texts, dtype=torch.int32)
        target_lengths = torch.tensor(text_lengths, dtype=torch.int32)
        
        # Forward pass
        optimizer.zero_grad()
        log_probs, input_lengths = model(images)
        
        # Convert to int32 for CTC loss
        input_lengths = input_lengths.int()
        
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Memory cleanup
        del images, targets, log_probs, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device, text_preprocessor):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for batch in progress_bar:
            images = batch['images'].to(device)
            texts = batch['texts']
            
            # Encode texts
            encoded_texts = []
            text_lengths = []
            for text in texts:
                encoded = text_preprocessor.encode_text(text)
                encoded_texts.append(encoded)
                text_lengths.append(len(encoded))
            
            # Pad sequences
            max_len = max(text_lengths) if text_lengths else 1
            padded_texts = []
            for encoded in encoded_texts:
                padded = encoded + [0] * (max_len - len(encoded))
                padded_texts.append(padded)
            
            targets = torch.tensor(padded_texts, dtype=torch.int32)
            target_lengths = torch.tensor(text_lengths, dtype=torch.int32)
            
            # Forward pass
            log_probs, input_lengths = model(images)
            
            # Convert to int32 for CTC loss
            input_lengths = input_lengths.int()
            
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()
            
            # Decode predictions (convert log_probs back to logits format)
            # log_probs is (seq_len, batch_size, num_classes), need (batch_size, seq_len, num_classes)
            logits_for_pred = log_probs.permute(1, 0, 2)
            predictions = torch.argmax(logits_for_pred, dim=2)
            for i, pred in enumerate(predictions):
                decoded_pred = text_preprocessor.decode_text(pred.cpu().numpy())
                all_predictions.append(decoded_pred)
                all_targets.append(texts[i])
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate CER
    cer = calculate_cer(all_targets, all_predictions)
    return total_loss / len(dataloader), cer, all_predictions[:3], all_targets[:3]

def main():
    print("Minimal Urdu OCR Training")
    print("=" * 50)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    print("\nLoading datasets...")
    train_loader, val_loader = create_data_loaders(
        train_dir=config.TRAIN_DIR,
        val_dir=config.VAL_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=0  # Reduce memory usage
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Model
    model = UrduOCRModel(config.NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Text preprocessor
    text_preprocessor = TextPreprocessor()
    
    # Training
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    print("=" * 50)
    
    best_cer = float('inf')
    best_model_state = None
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, text_preprocessor)
        
        # Validate
        val_loss, cer, sample_preds, sample_targets = validate_epoch(
            model, val_loader, criterion, device, text_preprocessor
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"CER: {cer:.4f}")
        
        # Sample predictions
        print("\nSample predictions:")
        for i in range(min(2, len(sample_preds))):
            print(f"  Target: {sample_targets[i]}")
            print(f"  Pred:   {sample_preds[i]}")
        
        # Save best model
        if cer < best_cer:
            best_cer = cer
            best_model_state = model.state_dict().copy()
            print(f"✓ New best CER: {best_cer:.4f}")
        
        # Early stopping for testing (stop after 5 epochs)
        if epoch >= 4:
            print(f"\nStopping after {epoch+1} epochs for testing...")
            break
    
    # Save final model
    print("\nSaving final model...")
    os.makedirs('models', exist_ok=True)
    
    if best_model_state is not None:
        torch.save({
            'model_state_dict': best_model_state,
            'best_cer': best_cer,
            'config': {
                'num_classes': config.NUM_CLASSES,
                'vocab_size': len(config.CHARACTERS)
            }
        }, 'models/best_model.pth')
        print(f"Best model saved with CER: {best_cer:.4f}")
    
    print("Training complete!")

if __name__ == "__main__":
    main()