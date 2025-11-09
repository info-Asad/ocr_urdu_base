# 🚀 GitHub Codespaces Guide for Urdu OCR

## Quick Start in Codespaces

### 1. **Launch Codespaces**
1. Go to your repository: https://github.com/info-Asad/ocr_urdu_base
2. Click **"Code"** → **"Codespaces"** → **"Create codespace on main"**
3. Wait for environment setup (2-3 minutes)

### 2. **Verify Setup**
```bash
# Test the environment
python test_setup.py

# Check installed packages
pip list | grep torch
```

### 3. **Upload Your Dataset**

#### Option A: Small Dataset (< 100MB)
```bash
# Drag and drop files to data/ folder in VS Code
# Or use the file upload feature
```

#### Option B: Large Dataset
```bash
# Upload to Google Drive/Dropbox and download
wget "YOUR_DATASET_LINK" -O dataset.zip
unzip dataset.zip -d data/
```

#### Option C: Use Sample Dataset
```bash
# Create a small test dataset
python create_sample_dataset.py
```

### 4. **Start Training**

#### Command Line:
```bash
# Start training
python urdu_ocr/train.py

# Monitor in another terminal
tensorboard --logdir=logs --host=0.0.0.0 --port=6006
```

#### Jupyter Notebook:
```bash
# Open the training notebook
jupyter notebook codespaces_training.ipynb
```

### 5. **Monitor Training**
- **Logs**: Check terminal output for loss/CER metrics
- **TensorBoard**: Access via Ports tab in Codespaces
- **Files**: Models saved in `models/` folder

### 6. **Test Predictions**
```bash
# Test with validation images
python test_model.py --simple

# Test specific image
python urdu_ocr/predict.py --image "path/to/your/image.png"
```

## 💡 **Codespaces Tips**

### **Performance Optimization:**
- **2-core machine**: Good for development and small datasets
- **4-core machine**: Better for training (if available in your plan)
- **Use CPU training**: GPU not available in standard Codespaces

### **Storage Management:**
- Codespaces has limited storage (32GB)
- Large model files are gitignored
- Download trained models before stopping Codespaces

### **Persistent Data:**
```bash
# Download trained model
curl -L -o best_model.pth "$(cat models/best_model.pth | base64)"

# Or commit and push (without large files)
git add . && git commit -m "Training progress" && git push
```

### **Free Tier Limits:**
- **60 hours/month** for free accounts
- **120 core hours/month** for Pro accounts
- Plan your training sessions accordingly

## 🔧 **Troubleshooting**

### **Out of Memory:**
```python
# Reduce batch size in urdu_ocr/config.py
BATCH_SIZE = 16  # or 8
```

### **Slow Training:**
```python
# Use fewer epochs for testing
NUM_EPOCHS = 10
```

### **Package Issues:**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

## 🎯 **Expected Performance**
- **CPU Training**: ~2-3 minutes per epoch
- **Small Dataset**: Good results in 10-20 epochs
- **Large Dataset**: May need 50+ epochs
- **Memory Usage**: ~2-4GB RAM

## 📊 **Monitoring Commands**
```bash
# Check system resources
htop

# Monitor Python processes
ps aux | grep python

# Check disk usage
df -h

# Check memory usage
free -h
```

Your Urdu OCR model will train efficiently in Codespaces! 🎉