"""
Utility Functions for Urdu OCR
Includes evaluation metrics, visualization, and helper functions
"""
import os
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization functions will be disabled.")

# Try to import editdistance, fallback to custom implementation
try:
    import editdistance
    EDITDISTANCE_AVAILABLE = True
except ImportError:
    EDITDISTANCE_AVAILABLE = False
    print("Warning: editdistance not available. Using custom implementation.")
    
    def editdistance_eval(s1, s2):
        """Simple edit distance (Levenshtein distance) implementation"""
        if len(s1) < len(s2):
            return editdistance_eval(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    # Create a dummy editdistance module
    class editdistance:
        @staticmethod
        def eval(s1, s2):
            return editdistance_eval(s1, s2)


def calculate_cer(predictions, targets):
    """
    Calculate Character Error Rate (CER)
    
    CER = (Substitutions + Deletions + Insertions) / Total Characters
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
    
    Returns:
        cer: Character Error Rate
    """
    total_distance = 0
    total_length = 0
    
    for pred, target in zip(predictions, targets):
        # Calculate edit distance
        distance = editdistance.eval(pred, target)
        total_distance += distance
        total_length += len(target)
    
    if total_length == 0:
        return 1.0
    
    cer = total_distance / total_length
    return cer


def calculate_wer(predictions, targets):
    """
    Calculate Word Error Rate (WER)
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
    
    Returns:
        wer: Word Error Rate
    """
    total_distance = 0
    total_words = 0
    
    for pred, target in zip(predictions, targets):
        # Split into words
        pred_words = pred.split()
        target_words = target.split()
        
        # Calculate edit distance on word level
        distance = editdistance.eval(pred_words, target_words)
        total_distance += distance
        total_words += len(target_words)
    
    if total_words == 0:
        return 1.0
    
    wer = total_distance / total_words
    return wer


def calculate_word_accuracy(predictions, targets):
    """
    Calculate word-level accuracy (exact match)
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
    
    Returns:
        accuracy: Word accuracy (0-1)
    """
    if len(predictions) == 0:
        return 0.0
    
    correct = sum(1 for pred, target in zip(predictions, targets) if pred == target)
    accuracy = correct / len(predictions)
    return accuracy


def calculate_accuracy(predictions, targets, threshold=0.8):
    """
    Calculate sequence accuracy with similarity threshold
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
        threshold: Similarity threshold (0-1)
    
    Returns:
        accuracy: Accuracy score
    """
    if len(predictions) == 0:
        return 0.0
    
    correct = 0
    for pred, target in zip(predictions, targets):
        if len(target) == 0:
            continue
        
        # Calculate similarity
        distance = editdistance.eval(pred, target)
        similarity = 1 - (distance / max(len(pred), len(target)))
        
        if similarity >= threshold:
            correct += 1
    
    accuracy = correct / len(predictions)
    return accuracy


def visualize_predictions(images, predictions, targets, num_samples=5, save_path=None):
    """
    Visualize model predictions
    
    Args:
        images: Tensor of images (B, C, H, W)
        predictions: List of predicted strings
        targets: List of target strings
        num_samples: Number of samples to display
        save_path: Path to save visualization
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping visualization.")
        return
    
    num_samples = min(num_samples, len(predictions))
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        # Convert tensor to numpy image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        
        # Display image
        axes[i].imshow(img)
        
        # Create title with prediction and target
        pred_text = predictions[i]
        target_text = targets[i]
        
        # Calculate similarity
        distance = editdistance.eval(pred_text, target_text)
        similarity = 1 - (distance / max(len(pred_text), len(target_text), 1))
        
        title = f"Target: {target_text}\nPred:   {pred_text}\nSimilarity: {similarity:.2%}"
        axes[i].set_title(title, fontsize=10, loc='left')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def save_predictions_to_file(predictions, targets, image_names, output_path):
    """
    Save predictions to text file
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
        image_names: List of image names
        output_path: Path to output file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Image Name\tTarget\tPrediction\tCER\n")
        f.write("="*80 + "\n")
        
        for img_name, target, pred in zip(image_names, targets, predictions):
            # Calculate CER for this sample
            distance = editdistance.eval(pred, target)
            cer = distance / max(len(target), 1)
            
            f.write(f"{img_name}\t{target}\t{pred}\t{cer:.4f}\n")
    
    print(f"Predictions saved to {output_path}")


def load_model_checkpoint(model, checkpoint_path, device='cpu'):
    """
    Load model from checkpoint
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model: Loaded model
        checkpoint: Checkpoint dictionary
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Best CER: {checkpoint.get('best_cer', 'N/A'):.4f}")
    
    return model, checkpoint


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params
    }


def plot_training_history(log_dir, save_path=None):
    """
    Plot training history from tensorboard logs
    
    Args:
        log_dir: Path to tensorboard log directory
        save_path: Path to save plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping training history plot.")
        return
    
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        
        # Get available tags
        tags = ea.Tags()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training loss
        if 'Train/EpochLoss' in tags['scalars']:
            train_loss = ea.Scalars('Train/EpochLoss')
            steps = [s.step for s in train_loss]
            values = [s.value for s in train_loss]
            axes[0, 0].plot(steps, values)
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
        
        # Plot validation loss
        if 'Val/Loss' in tags['scalars']:
            val_loss = ea.Scalars('Val/Loss')
            steps = [s.step for s in val_loss]
            values = [s.value for s in val_loss]
            axes[0, 1].plot(steps, values)
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)
        
        # Plot CER
        if 'Val/CER' in tags['scalars']:
            cer = ea.Scalars('Val/CER')
            steps = [s.step for s in cer]
            values = [s.value for s in cer]
            axes[1, 0].plot(steps, values)
            axes[1, 0].set_title('Character Error Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('CER')
            axes[1, 0].grid(True)
        
        # Plot word accuracy
        if 'Val/WordAccuracy' in tags['scalars']:
            acc = ea.Scalars('Val/WordAccuracy')
            steps = [s.step for s in acc]
            values = [s.value for s in acc]
            axes[1, 1].plot(steps, values)
            axes[1, 1].set_title('Word Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error plotting training history: {e}")


def create_confusion_matrix(predictions, targets, character_set):
    """
    Create confusion matrix for character predictions
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
        character_set: Set of all possible characters
    
    Returns:
        confusion_matrix: Numpy array (num_chars, num_chars)
    """
    char_to_idx = {char: idx for idx, char in enumerate(sorted(character_set))}
    num_chars = len(char_to_idx)
    
    confusion = np.zeros((num_chars, num_chars), dtype=np.int32)
    
    for pred, target in zip(predictions, targets):
        # Align sequences (simple approach)
        max_len = max(len(pred), len(target))
        pred_padded = pred + ' ' * (max_len - len(pred))
        target_padded = target + ' ' * (max_len - len(target))
        
        for p_char, t_char in zip(pred_padded, target_padded):
            if p_char in char_to_idx and t_char in char_to_idx:
                confusion[char_to_idx[t_char], char_to_idx[p_char]] += 1
    
    return confusion


def print_evaluation_metrics(predictions, targets):
    """
    Print comprehensive evaluation metrics
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
    """
    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)
    
    # Calculate metrics
    cer = calculate_cer(predictions, targets)
    wer = calculate_wer(predictions, targets)
    word_acc = calculate_word_accuracy(predictions, targets)
    acc_80 = calculate_accuracy(predictions, targets, threshold=0.8)
    acc_90 = calculate_accuracy(predictions, targets, threshold=0.9)
    
    print(f"\nCharacter Error Rate (CER): {cer:.4f} ({cer*100:.2f}%)")
    print(f"Word Error Rate (WER): {wer:.4f} ({wer*100:.2f}%)")
    print(f"Word Accuracy (Exact Match): {word_acc:.4f} ({word_acc*100:.2f}%)")
    print(f"Sequence Accuracy (≥80% similarity): {acc_80:.4f} ({acc_80*100:.2f}%)")
    print(f"Sequence Accuracy (≥90% similarity): {acc_90:.4f} ({acc_90*100:.2f}%)")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test CER calculation
    predictions = ["یہ ٹیسٹ ہے", "اردو زبان", "پاکستان"]
    targets = ["یہ ایک ٹیسٹ ہے", "اردو", "پاکستان"]
    
    cer = calculate_cer(predictions, targets)
    wer = calculate_wer(predictions, targets)
    word_acc = calculate_word_accuracy(predictions, targets)
    
    print(f"\nTest Results:")
    print(f"CER: {cer:.4f}")
    print(f"WER: {wer:.4f}")
    print(f"Word Accuracy: {word_acc:.4f}")
    
    print("\nUtility functions test completed!")
