# Quick Start Guide - Urdu OCR

## 1️⃣ Install Dependencies (5 minutes)

```bash
pip install -r requirements.txt
```

## 2️⃣ Test Installation

```bash
# Test model architecture
python urdu_ocr/model.py

# Test preprocessing
python urdu_ocr/data_preprocessing.py

# Test utilities
python urdu_ocr/utils.py
```

## 3️⃣ Prepare Dataset

### Option A: Use Existing Dataset
Place your dataset in this structure:
```
data/
├── train/
│   ├── images/
│   └── labels.txt
└── validation/
    ├── images/
    └── labels.txt
```

See **DATASET_GUIDE.md** for detailed instructions.

### Option B: Create Sample Dataset (for testing only)
```bash
python create_sample_dataset.py
```
⚠️ This creates only 60 samples - NOT sufficient for real training!

## 4️⃣ Train Model

```bash
python urdu_ocr/train.py
```

Monitor with TensorBoard:
```bash
tensorboard --logdir=logs
```

## 5️⃣ Make Predictions

### Single Image
```bash
python urdu_ocr/predict.py --image path/to/image.jpg --model models/best_model.pth
```

### Multiple Images
```bash
python urdu_ocr/predict.py --directory path/to/images/ --output predictions.txt
```

### Interactive Mode
```bash
python urdu_ocr/predict.py --interactive
```

## 📊 What You Need for Real Training

- **Minimum**: 5,000 training images + 500 validation images
- **Recommended**: 20,000+ training images + 2,000+ validation images
- **Image Quality**: Clear, readable Urdu text
- **Labels**: Accurate text transcriptions

## 🎯 Expected Results

With proper dataset (20,000+ images) and training:
- **Character Error Rate**: 5-15% (lower is better)
- **Word Accuracy**: 70-90%
- **Training Time**: 10-24 hours on GPU

## ⚡ Common Commands

```bash
# Test model
python urdu_ocr/model.py

# Verify dataset
python urdu_ocr/dataset.py

# Train model
python urdu_ocr/train.py

# Predict image
python urdu_ocr/predict.py --image test.jpg

# View training progress
tensorboard --logdir=logs
```

## 🐛 Troubleshooting

### No dataset error
→ Prepare dataset according to DATASET_GUIDE.md

### CUDA out of memory
→ Reduce BATCH_SIZE in urdu_ocr/config.py

### Poor accuracy
→ Need more training data (20,000+ images)

### Can't find model
→ Train model first with train.py

## 📚 Next Steps

1. Read **README.md** for complete documentation
2. Read **DATASET_GUIDE.md** for dataset preparation
3. Prepare your Urdu text dataset
4. Start training
5. Evaluate and improve

---

**Need Help?** Check README.md and DATASET_GUIDE.md for detailed instructions.
