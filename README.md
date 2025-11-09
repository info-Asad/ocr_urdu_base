# Urdu OCR Neural Network

A complete deep learning system for recognizing Urdu text from images using CNN + Bidirectional LSTM + CTC Loss architecture.

## 🎯 Overview

This project implements an end-to-end Optical Character Recognition (OCR) system specifically designed for Urdu text. The neural network can read Urdu text from images and convert it to machine-readable text.

### Features

- ✅ **Complete OCR Pipeline**: From image input to text output
- ✅ **Production-Ready Architecture**: CNN + BiLSTM + CTC Loss
- ✅ **Data Augmentation**: Automatic augmentation during training
- ✅ **Training & Validation**: Complete training pipeline with checkpointing
- ✅ **Inference Scripts**: Easy-to-use prediction interface
- ✅ **TensorBoard Integration**: Real-time training monitoring
- ✅ **Urdu Character Support**: Full Urdu alphabet, digits, and diacritics
- ✅ **GPU Acceleration**: CUDA support for faster training

## 📁 Project Structure

```
project/
├── urdu_ocr/
│   ├── config.py              # Configuration and hyperparameters
│   ├── model.py               # Neural network architecture
│   ├── dataset.py             # Dataset loading and processing
│   ├── data_preprocessing.py  # Image preprocessing utilities
│   ├── train.py               # Training script
│   ├── predict.py             # Inference script
│   └── utils.py               # Utility functions
├── data/
│   ├── train/                 # Training data
│   │   ├── images/
│   │   └── labels.txt
│   └── validation/            # Validation data
│       ├── images/
│       └── labels.txt
├── models/                    # Saved model checkpoints
├── logs/                      # TensorBoard logs
├── requirements.txt           # Python dependencies
├── DATASET_GUIDE.md          # Dataset preparation guide
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM recommended

### Installation

1. **Clone or download this project**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python urdu_ocr/model.py
```

This will test the model architecture and display model summary.

## 📊 Dataset Preparation

### Quick Start

Your dataset should follow this structure:

```
data/
├── train/
│   ├── images/
│   │   ├── img_0001.jpg
│   │   └── ...
│   └── labels.txt
└── validation/
    ├── images/
    │   ├── img_0001.jpg
    │   └── ...
    └── labels.txt
```

### Labels Format

`labels.txt` should contain tab-separated values:

```
img_0001.jpg	یہ ایک ٹیسٹ ہے
img_0002.jpg	اردو زبان بہت خوبصورت ہے
```

### Detailed Instructions

See **[DATASET_GUIDE.md](DATASET_GUIDE.md)** for:
- Dataset structure requirements
- Where to get Urdu datasets
- How to create synthetic data
- Dataset quality tips
- Example scripts

## 🏋️ Training

### Basic Training

```bash
python urdu_ocr/train.py
```

### Training Options

Edit `urdu_ocr/config.py` to customize:
- Batch size
- Learning rate
- Number of epochs
- Model architecture
- Image dimensions

### Monitor Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir=logs
```

Then open http://localhost:6006 in your browser.

### Training Features

- **Automatic checkpointing**: Best model saved automatically
- **Early stopping**: Stops if no improvement for N epochs
- **Learning rate scheduling**: Reduces LR on plateau
- **Data augmentation**: Applied automatically
- **Resume training**: Load checkpoint and continue

## 🔮 Prediction/Inference

### Predict Single Image

```bash
python urdu_ocr/predict.py --image path/to/image.jpg --model models/best_model.pth
```

### Predict All Images in Directory

```bash
python urdu_ocr/predict.py --directory path/to/images/ --output predictions.txt
```

### Interactive Mode

```bash
python urdu_ocr/predict.py --interactive --model models/best_model.pth
```

### Prediction Options

```bash
python urdu_ocr/predict.py --help
```

Available options:
- `--model`: Path to trained model checkpoint
- `--image`: Single image path
- `--directory`: Directory containing multiple images
- `--output`: Save predictions to file
- `--visualize`: Display images with predictions
- `--interactive`: Interactive mode
- `--device`: Use 'cuda' or 'cpu'

## 🏗️ Model Architecture

### Overview

```
Input Image (256x64x3)
        ↓
CNN Feature Extractor (5 conv blocks)
        ↓
Sequence Features (32x512)
        ↓
Bidirectional LSTM (2 layers, 256 hidden)
        ↓
Fully Connected Layer
        ↓
CTC Loss / Decoding
        ↓
Output Text (Urdu)
```

### Components

1. **CNN Feature Extractor**
   - 5 convolutional blocks
   - Batch normalization
   - MaxPooling
   - Output: Sequence of visual features

2. **Bidirectional LSTM**
   - 2-layer BiLSTM
   - 256 hidden units
   - Captures sequential dependencies

3. **CTC (Connectionist Temporal Classification)**
   - Handles variable-length sequences
   - No explicit segmentation needed
   - Automatic alignment

### Model Statistics

- **Total Parameters**: ~15-20 million (depending on config)
- **Input Size**: 256×64×3 (configurable)
- **Output**: Variable-length text
- **Vocabulary Size**: 100+ characters (Urdu + English + digits)

## 📈 Performance Metrics

The model evaluates performance using:

- **Character Error Rate (CER)**: Primary metric
- **Word Error Rate (WER)**: Word-level accuracy
- **Sequence Accuracy**: Exact match percentage
- **Similarity Score**: Fuzzy matching

## ⚙️ Configuration

All hyperparameters are in `urdu_ocr/config.py`:

```python
# Image Settings
IMG_HEIGHT = 64
IMG_WIDTH = 256

# Model Architecture
CNN_FILTERS = [64, 128, 256, 512]
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 2

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
```

## 🛠️ Advanced Usage

### Resume Training from Checkpoint

```python
trainer.train(resume=True)
```

### Custom Model Configuration

Edit `config.py`:

```python
# Larger model
CNN_FILTERS = [64, 128, 256, 512, 1024]
LSTM_HIDDEN_SIZE = 512
LSTM_NUM_LAYERS = 3
```

### Using Pretrained Model

```python
from urdu_ocr.predict import UrduOCRPredictor

predictor = UrduOCRPredictor('models/best_model.pth')
text = predictor.predict_single('image.jpg')
print(text)
```

### Programmatic Training

```python
from urdu_ocr.train import Trainer
from urdu_ocr.model import UrduOCRModel
from urdu_ocr.dataset import create_data_loaders

# Create data loaders
train_loader, val_loader = create_data_loaders(
    'data/train', 
    'data/validation'
)

# Create model
model = UrduOCRModel()

# Create trainer
trainer = Trainer(model, train_loader, val_loader)

# Train
trainer.train()
```

## 🧪 Testing

### Test Data Preprocessing

```bash
python urdu_ocr/data_preprocessing.py
```

### Test Dataset Loading

```bash
python urdu_ocr/dataset.py
```

### Test Model Architecture

```bash
python urdu_ocr/model.py
```

### Test Utility Functions

```bash
python urdu_ocr/utils.py
```

## 📝 Dataset Requirements

### Minimum Requirements

- **Training samples**: 5,000+
- **Validation samples**: 500+
- **Image quality**: Clear, readable text
- **Format**: JPG, PNG, BMP, TIFF

### Recommended

- **Training samples**: 20,000+
- **Validation samples**: 2,000+
- **Variety**: Different fonts, sizes, backgrounds
- **Quality**: High-resolution images

See **[DATASET_GUIDE.md](DATASET_GUIDE.md)** for detailed instructions.

## 🎓 How It Works

### Training Process

1. **Data Loading**: Images and labels loaded in batches
2. **Preprocessing**: Resize, normalize, augment
3. **Forward Pass**: Image → CNN → LSTM → Predictions
4. **Loss Calculation**: CTC loss between predictions and targets
5. **Backpropagation**: Update model weights
6. **Validation**: Evaluate on validation set
7. **Checkpointing**: Save best model

### Inference Process

1. **Load Image**: Read image from file
2. **Preprocess**: Resize and normalize
3. **Forward Pass**: Extract features and predict
4. **Decode**: Convert predictions to text (CTC decoding)
5. **Output**: Return recognized text

## 🐛 Troubleshooting

### CUDA Out of Memory

Reduce batch size in `config.py`:
```python
BATCH_SIZE = 16  # or smaller
```

### Dataset Not Loading

- Check dataset structure matches requirements
- Verify `labels.txt` format (TAB-separated)
- Ensure UTF-8 encoding
- Check image paths

### Poor Accuracy

- Increase dataset size
- Train for more epochs
- Adjust learning rate
- Check data quality
- Verify labels are correct

### Slow Training

- Use GPU (CUDA)
- Increase batch size
- Reduce image size
- Use fewer augmentations

## 📚 Dependencies

Main dependencies:
- PyTorch >= 2.0.0
- OpenCV >= 4.8.0
- Pillow >= 10.0.0
- NumPy >= 1.24.0
- albumentations >= 1.3.0

See `requirements.txt` for complete list.

## 🔄 Workflow Summary

```
1. Prepare Dataset → See DATASET_GUIDE.md
2. Install Dependencies → pip install -r requirements.txt
3. Configure Settings → Edit urdu_ocr/config.py
4. Start Training → python urdu_ocr/train.py
5. Monitor Progress → tensorboard --logdir=logs
6. Make Predictions → python urdu_ocr/predict.py
```

## 💡 Tips for Best Results

1. **Quality Data**: Use high-quality, clear images
2. **Large Dataset**: More data = better accuracy
3. **Balanced Data**: Variety of fonts, sizes, backgrounds
4. **Clean Labels**: Double-check text transcriptions
5. **Patience**: Training takes time (hours to days)
6. **GPU**: Use CUDA for faster training
7. **Monitoring**: Watch training curves for overfitting
8. **Checkpoints**: Save regularly, test on validation

## 📖 Character Set

The model supports:
- **Urdu Letters**: Complete Urdu alphabet (ا to ی)
- **Urdu Digits**: ۰ to ۹
- **Diacritics**: Zabar, Zer, Pesh, etc.
- **English**: a-z, A-Z, 0-9 (optional)
- **Punctuation**: Space, period, comma, etc.

Total vocabulary: ~100 characters

## 🤝 Contributing

To improve this project:
1. Add more data augmentation techniques
2. Experiment with different architectures
3. Optimize for mobile/edge devices
4. Add more evaluation metrics
5. Create web interface

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- CTC Loss implementation inspired by PyTorch
- Architecture based on state-of-the-art OCR research
- Urdu character set from Unicode standard

## 📞 Support

If you encounter issues:
1. Check this README
2. See DATASET_GUIDE.md
3. Verify dataset structure
4. Check error messages
5. Review configuration settings

## 🎯 Next Steps

After successful training:
1. Test on real-world images
2. Fine-tune hyperparameters
3. Collect more training data
4. Deploy for production use
5. Create API/web interface

---

**Happy Training! 🚀**

For dataset preparation instructions, see **[DATASET_GUIDE.md](DATASET_GUIDE.md)**
