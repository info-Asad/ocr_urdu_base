# Project Structure Overview

```
project/
│
├── 📄 README.md                      # Main documentation
├── 📄 QUICKSTART.md                  # Quick start guide
├── 📄 DATASET_GUIDE.md               # Dataset preparation guide
├── 📄 requirements.txt                # Python dependencies
├── 📄 create_sample_dataset.py       # Script to create test dataset
│
├── 📁 urdu_ocr/                      # Main package
│   ├── __init__.py                   # Package initialization
│   ├── config.py                     # Configuration & hyperparameters
│   ├── model.py                      # Neural network architecture
│   ├── dataset.py                    # Dataset loading & processing
│   ├── data_preprocessing.py         # Image preprocessing utilities
│   ├── train.py                      # Training script
│   ├── predict.py                    # Inference/prediction script
│   └── utils.py                      # Utility functions (metrics, etc.)
│
├── 📁 data/                          # Dataset directory
│   ├── 📁 train/                     # Training data
│   │   ├── 📁 images/                # Training images
│   │   └── 📄 labels.txt             # Training labels
│   └── 📁 validation/                # Validation data
│       ├── 📁 images/                # Validation images
│       └── 📄 labels.txt             # Validation labels
│
├── 📁 models/                        # Saved model checkpoints
│   ├── best_model.pth                # Best model (created during training)
│   ├── last_checkpoint.pth           # Latest checkpoint
│   └── checkpoint_epoch_N.pth        # Epoch checkpoints
│
└── 📁 logs/                          # Training logs
    └── [tensorboard logs]            # TensorBoard log files

```

## File Descriptions

### Core Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `config.py` | Settings & hyperparameters | Edit before training to adjust model |
| `model.py` | Neural network architecture | Run to test model, edit to change architecture |
| `dataset.py` | Data loading | Run to verify dataset, auto-used in training |
| `data_preprocessing.py` | Image preprocessing | Run to test preprocessing, auto-used |
| `train.py` | Training script | Run to train model |
| `predict.py` | Inference script | Run to make predictions on new images |
| `utils.py` | Helper functions | Auto-used by other scripts |

### Documentation Files

| File | Content |
|------|---------|
| `README.md` | Complete project documentation |
| `QUICKSTART.md` | Quick start guide (5-minute setup) |
| `DATASET_GUIDE.md` | Dataset preparation instructions |
| `PROJECT_STRUCTURE.md` | This file |

### Generated During Use

| File/Folder | Created By | Purpose |
|-------------|-----------|---------|
| `models/*.pth` | Training | Saved model checkpoints |
| `logs/*` | Training | TensorBoard training logs |
| `data/train/*` | User | Training dataset (you provide) |
| `data/validation/*` | User | Validation dataset (you provide) |

## Workflow

```
1. Install → pip install -r requirements.txt
2. Dataset → Prepare data/ folder (see DATASET_GUIDE.md)
3. Config → Edit urdu_ocr/config.py if needed
4. Train → python urdu_ocr/train.py
5. Monitor → tensorboard --logdir=logs
6. Predict → python urdu_ocr/predict.py --image test.jpg
```

## Key Components

### 1. Neural Network (`model.py`)
- **CNN**: Extracts visual features from images
- **BiLSTM**: Models sequential dependencies
- **CTC**: Handles variable-length sequences

### 2. Data Pipeline (`dataset.py` + `data_preprocessing.py`)
- **Loading**: Reads images and labels
- **Preprocessing**: Resizes, normalizes images
- **Augmentation**: Adds noise, blur, rotation, etc.

### 3. Training (`train.py`)
- **Forward Pass**: Image → Predictions
- **Loss**: CTC loss calculation
- **Backprop**: Update weights
- **Validation**: Test on validation set
- **Checkpointing**: Save best models

### 4. Inference (`predict.py`)
- **Load Model**: Load trained checkpoint
- **Preprocess**: Prepare image
- **Predict**: Run through model
- **Decode**: Convert to text

## Directory Sizes (Typical)

| Directory | Typical Size |
|-----------|--------------|
| `urdu_ocr/` | < 1 MB (code) |
| `data/train/` | 1-10 GB (depends on dataset) |
| `data/validation/` | 100 MB - 1 GB |
| `models/` | 100-500 MB (model checkpoints) |
| `logs/` | 10-100 MB (training logs) |

## What You Need to Add

1. **Dataset** → `data/train/` and `data/validation/`
   - Thousands of Urdu text images
   - Corresponding labels.txt files
   - See DATASET_GUIDE.md

2. **Nothing else!** All code is complete and ready to use.

## What Gets Created Automatically

During training:
- `models/best_model.pth` - Best model checkpoint
- `models/checkpoint_epoch_*.pth` - Epoch checkpoints
- `logs/*` - TensorBoard logs

## File Dependencies

```
train.py
  ├── config.py
  ├── model.py
  ├── dataset.py
  │   └── data_preprocessing.py
  └── utils.py

predict.py
  ├── config.py
  ├── model.py
  ├── data_preprocessing.py
  └── utils.py
```

## Configuration Options

Edit `config.py` to customize:

```python
# Image size
IMG_HEIGHT = 64        # Image height
IMG_WIDTH = 256        # Image width

# Model architecture
CNN_FILTERS = [64, 128, 256, 512]  # CNN filter sizes
LSTM_HIDDEN_SIZE = 256              # LSTM hidden units
LSTM_NUM_LAYERS = 2                 # Number of LSTM layers

# Training
BATCH_SIZE = 32                     # Batch size
LEARNING_RATE = 0.0001             # Learning rate
NUM_EPOCHS = 100                    # Maximum epochs
```

## Testing Each Component

```bash
# Test model
python urdu_ocr/model.py

# Test data preprocessing
python urdu_ocr/data_preprocessing.py

# Test dataset loading
python urdu_ocr/dataset.py

# Test utilities
python urdu_ocr/utils.py
```

Each file can be run independently for testing!

---

**Everything is ready to use!** Just prepare your dataset and start training.
