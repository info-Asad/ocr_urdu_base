#!/bin/bash
# Codespaces Setup Script for Urdu OCR

echo "🚀 Setting up Urdu OCR in GitHub Codespaces"
echo "============================================"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/train/images data/validation/images models logs

# Set Python path
export PYTHONPATH="/workspaces/ocr_urdu_base:$PYTHONPATH"

# Create sample data structure info
echo "📋 Creating data structure guide..."
cat > data/README.md << 'EOL'
# Dataset Structure for Urdu OCR

Place your dataset in this structure:

```
data/
├── train/
│   ├── images/           # Training images (.png)
│   └── labels.txt        # Format: img_name.png\tUrdu_text
└── validation/
    ├── images/           # Validation images (.png)  
    └── labels.txt        # Format: img_name.png\tUrdu_text
```

## Sample Upload Commands:

1. **Upload via GitHub:**
   ```bash
   # Create a separate data repository (recommended)
   # Upload large files using Git LFS
   ```

2. **Upload via wget/curl:**
   ```bash
   # If you have dataset hosted online
   wget YOUR_DATASET_URL -O dataset.zip
   unzip dataset.zip -d data/
   ```

3. **Small test dataset:**
   ```bash
   # Use the sample dataset creator
   python create_sample_dataset.py
   ```
EOL

# Create quick test script
echo "🧪 Creating quick test script..."
cat > test_setup.py << 'EOL'
#!/usr/bin/env python3
"""
Quick test to verify Codespaces setup
"""
import sys
import torch
import cv2
import numpy as np
from pathlib import Path

def test_environment():
    print("🧪 Testing Codespaces Environment")
    print("=" * 40)
    
    # Python version
    print(f"✅ Python: {sys.version}")
    
    # PyTorch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA Devices: {torch.cuda.device_count()}")
    
    # OpenCV
    print(f"✅ OpenCV: {cv2.__version__}")
    
    # Test model creation
    try:
        sys.path.append('/workspaces/ocr_urdu_base')
        from urdu_ocr.model import UrduOCRModel
        from urdu_ocr import config
        
        model = UrduOCRModel(config.NUM_CLASSES)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model Created: {total_params:,} parameters")
        
    except Exception as e:
        print(f"❌ Model Test Failed: {e}")
    
    # Check directories
    dirs = ['data/train', 'data/validation', 'models', 'logs']
    for dir_path in dirs:
        if Path(dir_path).exists():
            print(f"✅ Directory: {dir_path}")
        else:
            print(f"❌ Missing: {dir_path}")
    
    print("\n🎉 Environment setup complete!")
    print("📋 Next steps:")
    print("1. Upload your dataset to data/ folder")
    print("2. Run: python urdu_ocr/train.py")
    print("3. Monitor with: tensorboard --logdir=logs")

if __name__ == "__main__":
    test_environment()
EOL

chmod +x test_setup.py

# Create Jupyter notebook for interactive training
echo "📓 Creating Jupyter notebook..."
cat > codespaces_training.ipynb << 'EOL'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🔤 Urdu OCR Training in GitHub Codespaces\n",
    "\n",
    "This notebook helps you train your Urdu OCR model in GitHub Codespaces."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test environment setup\n",
    "exec(open('test_setup.py').read())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check if dataset exists\n",
    "import os\n",
    "if os.path.exists('data/train/labels.txt'):\n",
    "    print(\"✅ Training dataset found\")\n",
    "    with open('data/train/labels.txt', 'r', encoding='utf-8') as f:\n",
    "        lines = f.readlines()[:5]\n",
    "        print(f\"📊 Sample training data ({len(lines)} lines shown):\")\n",
    "        for line in lines:\n",
    "            print(f\"  {line.strip()}\")\n",
    "else:\n",
    "    print(\"❌ Dataset not found. Please upload your dataset or run create_sample_dataset.py\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Start training\n",
    "!python urdu_ocr/train.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test predictions\n",
    "!python test_model.py --simple"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOL

echo ""
echo "✅ Codespaces setup complete!"
echo "📋 Files created:"
echo "   - .devcontainer/devcontainer.json"
echo "   - test_setup.py"
echo "   - codespaces_training.ipynb"
echo "   - data/README.md"
echo ""
echo "🚀 Ready for GitHub Codespaces!"