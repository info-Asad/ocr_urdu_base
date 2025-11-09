"""
Training Script for Urdu OCR Model
Handles model training, validation, checkpointing, and logging
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

import config
from model import UrduOCRModel, print_model_summary
from dataset import create_data_loaders
from data_preprocessing import TextPreprocessor
import utils


class Trainer:
    """Trainer class for Urdu OCR model"""
    
    def __init__(
        self, 
        model, 
        train_loader, 
        val_loader, 
        device='cuda',
        learning_rate=config.LEARNING_RATE,
        num_epochs=config.NUM_EPOCHS,
        model_dir=config.MODEL_DIR,
        log_dir=config.LOG_DIR
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.model_dir = model_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Loss function (CTC Loss)
        self.criterion = nn.CTCLoss(blank=config.BLANK_INDEX, zero_infinity=True)
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        # Text preprocessor for decoding
        self.text_preprocessor = TextPreprocessor()
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_cer = float('inf')
        self.epochs_without_improvement = 0
        
        print("Trainer initialized")
        print(f"Device: {device}")
        print(f"Learning rate: {learning_rate}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Number of epochs: {num_epochs}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.num_epochs} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['images'].to(self.device)
            encoded_texts = batch['encoded_texts']
            text_lengths = batch['text_lengths']
            
            # Forward pass
            log_probs, input_lengths = self.model(images)
            
            # Calculate CTC loss
            # log_probs: (T, N, C) where T=sequence_length, N=batch_size, C=num_classes
            # targets: (N, S) where S=target_length
            # input_lengths: (N,) length of each input sequence
            # target_lengths: (N,) length of each target sequence
            
            loss = self.criterion(
                log_probs.cpu(),
                encoded_texts,
                input_lengths.cpu(),
                text_lengths
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRADIENT_CLIP)
            
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Log to tensorboard
            global_step = self.current_epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1}/{self.num_epochs} [Val]")
            
            for batch in pbar:
                # Move data to device
                images = batch['images'].to(self.device)
                encoded_texts = batch['encoded_texts']
                text_lengths = batch['text_lengths']
                texts = batch['texts']
                
                # Forward pass
                log_probs, input_lengths = self.model(images)
                
                # Calculate loss
                loss = self.criterion(
                    log_probs.cpu(),
                    encoded_texts,
                    input_lengths.cpu(),
                    text_lengths
                )
                
                total_loss += loss.item()
                
                # Get predictions
                predictions = self.model.predict(images)
                
                # Decode predictions
                for i in range(predictions.size(0)):
                    pred_indices = predictions[i].cpu().numpy()
                    pred_text = self.text_preprocessor.decode_text(pred_indices)
                    target_text = texts[i]
                    
                    all_predictions.append(pred_text)
                    all_targets.append(target_text)
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate Character Error Rate (CER)
        cer = utils.calculate_cer(all_predictions, all_targets)
        
        # Calculate Word Accuracy
        word_accuracy = utils.calculate_word_accuracy(all_predictions, all_targets)
        
        return avg_loss, cer, word_accuracy, all_predictions, all_targets
    
    def save_checkpoint(self, filename, is_best=False, val_loss=None, cer=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_cer': self.best_cer,
        }
        
        # Add current values if provided
        if val_loss is not None:
            checkpoint['val_loss'] = val_loss
        if cer is not None:
            checkpoint['cer'] = cer
        
        try:
            filepath = os.path.join(self.model_dir, filename)
            torch.save(checkpoint, filepath)
            print(f"Checkpoint saved: {filepath}")
            
            if is_best:
                best_path = os.path.join(self.model_dir, 'best_model.pth')
                torch.save(checkpoint, best_path)
                print(f"Best model saved: {best_path}")
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")
            print("Training will continue without saving checkpoints.")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        filepath = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Checkpoint not found: {filepath}")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler states if available
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("Warning: optimizer_state_dict not found, using fresh optimizer")
            
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print("Warning: scheduler_state_dict not found, using fresh scheduler")
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_cer = checkpoint.get('best_cer', float('inf'))
        
        print(f"Checkpoint loaded: {filepath}")
        print(f"Resuming from epoch {self.current_epoch}")
        
        return True
    
    def train(self, resume=False):
        """Main training loop"""
        if resume:
            self.load_checkpoint('last_checkpoint.pth')
        
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, cer, word_acc, predictions, targets = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log to tensorboard
            self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/CER', cer, epoch)
            self.writer.add_scalar('Val/WordAccuracy', word_acc, epoch)
            self.writer.add_scalar('Train/LearningRate', 
                                 self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  CER: {cer:.4f}")
            print(f"  Word Accuracy: {word_acc:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Print sample predictions
            print("\n  Sample Predictions:")
            for i in range(min(3, len(predictions))):
                print(f"    Target: {targets[i]}")
                print(f"    Pred:   {predictions[i]}")
                print()
            
            # Check if this is the best model
            is_best = cer < self.best_cer
            if is_best:
                self.best_cer = cer
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                print(f"  ✓ New best model! CER: {cer:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint after every epoch for resuming
            self.save_checkpoint('last_checkpoint.pth', is_best=False, val_loss=val_loss, cer=cer)
            
            # Save best model
            if is_best:
                self.save_checkpoint('best_model.pth', is_best=True, val_loss=val_loss, cer=cer)
                
            # Save numbered checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', is_best=False, val_loss=val_loss, cer=cer)
            
            # Early stopping
            if self.epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            print("-" * 70)
        
        # Training completed
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best CER: {self.best_cer:.4f}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print("="*70 + "\n")
        
        self.writer.close()


def main():
    """Main training function"""
    print("Urdu OCR Training Script")
    print("="*70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Update config device
    config.DEVICE = device
    
    # Create data loaders
    print("\nLoading datasets...")
    try:
        train_loader, val_loader = create_data_loaders(
            train_dir=config.TRAIN_DIR,
            val_dir=config.VAL_DIR,
            batch_size=config.BATCH_SIZE,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("\nPlease make sure your dataset is prepared correctly.")
        print("See DATASET_GUIDE.md for instructions.")
        return
    
    # Create model
    print("\nInitializing model...")
    model = UrduOCRModel()
    print_model_summary(model)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Check for existing checkpoint
    last_checkpoint = os.path.join('models', 'last_checkpoint.pth')
    resume_training = os.path.exists(last_checkpoint)
    
    if resume_training:
        print(f"\n🔄 Found existing checkpoint: {last_checkpoint}")
        print("Resuming training from last checkpoint...")
    else:
        print("\n🚀 Starting fresh training...")
    
    # Start training
    trainer.train(resume=resume_training)


if __name__ == "__main__":
    main()
