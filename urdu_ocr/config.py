"""
Configuration file for Urdu OCR Neural Network
Contains all hyperparameters and settings
"""
import os
import string

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Image Settings
IMG_HEIGHT = 64
IMG_WIDTH = 256
IMG_CHANNELS = 3

# Model Architecture
CNN_FILTERS = [64, 128, 256, 512]
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 2
DROPOUT = 0.3

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
GRADIENT_CLIP = 5.0

# Urdu Character Set
# Urdu alphabet + digits + special characters
URDU_CHARACTERS = [
    # Urdu letters (basic forms)
    'ا', 'ب', 'پ', 'ت', 'ٹ', 'ث', 'ج', 'چ', 'ح', 'خ',
    'د', 'ڈ', 'ذ', 'ر', 'ڑ', 'ز', 'ژ', 'س', 'ش', 'ص',
    'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل',
    'م', 'ن', 'و', 'ہ', 'ھ', 'ء', 'ی', 'ے',
    # Urdu digits
    '۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹',
    # Arabic/Urdu diacritics
    'ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ',
    # Special characters
    ' ', '.', '،', '؛', '؟', '!', '-', ':', ')', '(',
]

# English characters (optional, for mixed text)
ENGLISH_CHARACTERS = list(string.ascii_letters + string.digits)

# Complete character set
ALL_CHARACTERS = URDU_CHARACTERS + ENGLISH_CHARACTERS

# Add blank character for CTC loss (must be index 0)
BLANK_INDEX = 0
CHARACTERS = ['<BLANK>'] + ALL_CHARACTERS

# Character to index mapping
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARACTERS)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARACTERS)}

NUM_CLASSES = len(CHARACTERS)

# Data Augmentation Settings
AUGMENTATION = {
    'random_brightness': 0.2,
    'random_contrast': 0.2,
    'random_gamma': (80, 120),
    'blur_limit': 3,
    'noise_var': (10, 50),
    'rotation_limit': 2,
}

# Inference Settings
CONFIDENCE_THRESHOLD = 0.5

# Device Settings
DEVICE = 'cuda'  # Will be automatically set to 'cpu' if CUDA is not available

print(f"Configuration loaded. Total characters in vocabulary: {NUM_CLASSES}")
