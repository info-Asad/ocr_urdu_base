"""
Neural Network Model for Urdu OCR
Architecture: CNN (Feature Extraction) + Bidirectional LSTM (Sequence Modeling) + CTC Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class CNNFeatureExtractor(nn.Module):
    """
    Convolutional Neural Network for extracting features from images
    Outputs: (batch_size, sequence_length, feature_dim)
    """
    
    def __init__(self, input_channels=config.IMG_CHANNELS, cnn_filters=config.CNN_FILTERS):
        super(CNNFeatureExtractor, self).__init__()
        
        # Conv Block 1: (B, 3, 64, 256) -> (B, 64, 32, 128)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, cnn_filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_filters[0], cnn_filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_filters[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Conv Block 2: (B, 64, 32, 128) -> (B, 128, 16, 64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_filters[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_filters[1], cnn_filters[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_filters[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Conv Block 3: (B, 128, 16, 64) -> (B, 256, 8, 32)
        self.conv3 = nn.Sequential(
            nn.Conv2d(cnn_filters[1], cnn_filters[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_filters[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_filters[2], cnn_filters[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_filters[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # Only height pooling
        )
        
        # Conv Block 4: (B, 256, 8, 32) -> (B, 512, 4, 32)
        self.conv4 = nn.Sequential(
            nn.Conv2d(cnn_filters[2], cnn_filters[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_filters[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_filters[3], cnn_filters[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_filters[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # Only height pooling
        )
        
        # Conv Block 5: (B, 512, 4, 32) -> (B, 512, 1, 32)
        self.conv5 = nn.Sequential(
            nn.Conv2d(cnn_filters[3], cnn_filters[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_filters[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))  # Reduce height to 1
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, channels, height, width)
        Returns:
            features: (batch_size, sequence_length, feature_dim)
        """
        # Pass through conv blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # x shape: (batch_size, 512, 1, sequence_length)
        # Remove height dimension and transpose
        batch_size, channels, height, width = x.size()
        x = x.squeeze(2)  # (batch_size, 512, sequence_length)
        x = x.permute(0, 2, 1)  # (batch_size, sequence_length, 512)
        
        return x


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM for sequence modeling
    """
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=config.DROPOUT):
        super(BidirectionalLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
        Returns:
            output: (batch_size, sequence_length, hidden_size*2)
        """
        output, _ = self.lstm(x)
        return output


class UrduOCRModel(nn.Module):
    """
    Complete Urdu OCR Model
    CNN + Bidirectional LSTM + Linear Layer for CTC
    """
    
    def __init__(
        self, 
        num_classes=config.NUM_CLASSES,
        cnn_filters=config.CNN_FILTERS,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
        dropout=config.DROPOUT
    ):
        super(UrduOCRModel, self).__init__()
        
        self.num_classes = num_classes
        
        # Feature extraction
        self.cnn = CNNFeatureExtractor(
            input_channels=config.IMG_CHANNELS,
            cnn_filters=cnn_filters
        )
        
        # Sequence modeling
        cnn_output_size = cnn_filters[-1]  # 512
        self.lstm = BidirectionalLSTM(
            input_size=cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout
        )
        
        # Fully connected layer for classification
        lstm_output_size = lstm_hidden_size * 2  # Bidirectional
        self.fc = nn.Linear(lstm_output_size, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input images (batch_size, channels, height, width)
        Returns:
            output: Log probabilities (batch_size, sequence_length, num_classes)
            sequence_lengths: Length of each sequence
        """
        # Feature extraction
        features = self.cnn(x)  # (batch_size, sequence_length, feature_dim)
        
        # Sequence modeling
        lstm_out = self.lstm(features)  # (batch_size, sequence_length, hidden_size*2)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Classification
        logits = self.fc(lstm_out)  # (batch_size, sequence_length, num_classes)
        
        # Apply log softmax for CTC loss
        log_probs = F.log_softmax(logits, dim=2)
        
        # For CTC loss, we need (sequence_length, batch_size, num_classes)
        log_probs = log_probs.permute(1, 0, 2)
        
        # Get sequence lengths (all sequences have same length after CNN)
        sequence_length = log_probs.size(0)
        batch_size = log_probs.size(1)
        sequence_lengths = torch.full((batch_size,), sequence_length, dtype=torch.long)
        
        return log_probs, sequence_lengths
    
    def predict(self, x):
        """
        Prediction mode (no gradient computation)
        
        Args:
            x: Input images (batch_size, channels, height, width)
        Returns:
            predictions: Predicted character indices
        """
        self.eval()
        with torch.no_grad():
            log_probs, _ = self.forward(x)
            # Get most likely character at each position
            _, predictions = torch.max(log_probs, dim=2)
            # Transpose back to (batch_size, sequence_length)
            predictions = predictions.permute(1, 0)
        return predictions


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model):
    """Print detailed model summary"""
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*70)
    
    print("\n1. CNN Feature Extractor:")
    print(f"   - Input: (batch_size, {config.IMG_CHANNELS}, {config.IMG_HEIGHT}, {config.IMG_WIDTH})")
    print(f"   - Conv blocks: {len(config.CNN_FILTERS)}")
    print(f"   - Filters: {config.CNN_FILTERS}")
    print(f"   - Output: (batch_size, sequence_length, {config.CNN_FILTERS[-1]})")
    
    print("\n2. Bidirectional LSTM:")
    print(f"   - Input size: {config.CNN_FILTERS[-1]}")
    print(f"   - Hidden size: {config.LSTM_HIDDEN_SIZE}")
    print(f"   - Num layers: {config.LSTM_NUM_LAYERS}")
    print(f"   - Dropout: {config.DROPOUT}")
    print(f"   - Output: (batch_size, sequence_length, {config.LSTM_HIDDEN_SIZE * 2})")
    
    print("\n3. Fully Connected Layer:")
    print(f"   - Input size: {config.LSTM_HIDDEN_SIZE * 2}")
    print(f"   - Output size (num_classes): {config.NUM_CLASSES}")
    
    print("\n4. Model Statistics:")
    print(f"   - Total parameters: {count_parameters(model):,}")
    print(f"   - Vocabulary size: {config.NUM_CLASSES}")
    print(f"   - Blank index for CTC: {config.BLANK_INDEX}")
    
    print("\n" + "="*70 + "\n")


def test_model():
    """Test model with random input"""
    print("Testing model with random input...")
    
    # Create model
    model = UrduOCRModel()
    print_model_summary(model)
    
    # Create random input
    batch_size = 4
    dummy_input = torch.randn(batch_size, config.IMG_CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    log_probs, seq_lengths = model(dummy_input)
    
    print(f"Output shape: {log_probs.shape}")
    print(f"Sequence lengths: {seq_lengths}")
    
    # Test prediction
    predictions = model.predict(dummy_input)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample prediction (first sequence): {predictions[0][:10].tolist()}")
    
    print("\nModel test completed successfully!")


if __name__ == "__main__":
    test_model()
