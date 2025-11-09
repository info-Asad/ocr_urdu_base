# Urdu OCR Neural Network - Dataset Preparation Guide

## Overview
This guide explains how to prepare your dataset for training the Urdu OCR model.

## Required Dataset Structure

Your dataset should be organized in the following structure:

```
data/
├── train/
│   ├── images/
│   │   ├── img_0001.jpg
│   │   ├── img_0002.jpg
│   │   └── ...
│   └── labels.txt
└── validation/
    ├── images/
    │   ├── img_0001.jpg
    │   ├── img_0002.jpg
    │   └── ...
    └── labels.txt
```

## Labels File Format

The `labels.txt` file should contain tab-separated values with the following format:

```
image_name.jpg<TAB>urdu_text
```

**Example:**
```
img_0001.jpg	یہ ایک ٹیسٹ ہے
img_0002.jpg	اردو زبان بہت خوبصورت ہے
img_0003.jpg	پاکستان زندہ باد
```

### Important Points:
- Use **TAB character** (\t) to separate image name and text
- Image names should match exactly with files in the `images/` folder
- Text should be in Urdu (Arabic script)
- One line per image
- Use UTF-8 encoding for the file

## Image Requirements

### Format
- Supported formats: JPG, JPEG, PNG, BMP, TIFF
- Color images (RGB) or grayscale

### Quality
- **Minimum resolution**: 256x64 pixels
- **Recommended**: Higher resolution (will be resized automatically)
- **Text clarity**: Text should be clear and readable
- **Background**: Uniform background preferred (but not required)

### Orientation
- Text should be horizontal (left-to-right or right-to-left)
- No rotation preferred
- Model will handle slight rotations through augmentation

## Dataset Size Recommendations

| Purpose | Minimum Samples | Recommended |
|---------|----------------|-------------|
| Training | 5,000 | 20,000+ |
| Validation | 500 | 2,000+ |

### Split Ratio
- Training: 80-90%
- Validation: 10-20%

## Where to Get Urdu Text Datasets

### 1. **Public Datasets**

#### UPTI (Urdu Printed Text Images)
- Publicly available Urdu OCR dataset
- Contains printed Urdu text images
- URL: Search for "UPTI dataset" or "Urdu OCR dataset"

#### CLE Urdu Corpus
- Center for Language Engineering (CLE) provides Urdu datasets
- URL: http://cle.org.pk/

#### IIIT-HW-Urdu (Handwritten)
- Urdu handwritten text dataset
- More challenging but useful for robust models

### 2. **Synthetic Data Generation**

You can create synthetic Urdu text images using:

#### Python Script (Example):
```python
from PIL import Image, ImageDraw, ImageFont
import os

def generate_urdu_image(text, output_path, font_path):
    # Create image
    img = Image.new('RGB', (256, 64), color='white')
    draw = ImageDraw.Draw(img)
    
    # Load Urdu font (e.g., Jameel Noori Nastaleeq)
    font = ImageFont.truetype(font_path, 32)
    
    # Draw text
    draw.text((10, 10), text, font=font, fill='black')
    
    # Save
    img.save(output_path)

# Example usage
urdu_texts = [
    "یہ ٹیسٹ ہے",
    "اردو زبان",
    "پاکستان زندہ باد"
]

for i, text in enumerate(urdu_texts):
    generate_urdu_image(text, f"img_{i:04d}.jpg", "JameelNoori.ttf")
```

#### Tools for Synthetic Data:
- **TextRecognitionDataGenerator**: Python library for text image generation
- **Pillow/PIL**: Python imaging library
- **imgaug**: Image augmentation library

### 3. **Web Scraping**
- Urdu news websites
- Urdu PDF documents (extract text and render)
- Urdu books (digitized)

**Note**: Ensure you have proper permissions and respect copyright laws.

### 4. **Manual Annotation**
- Collect Urdu text images
- Manually transcribe the text
- Use annotation tools like:
  - **Label Studio**
  - **CVAT**
  - **VGG Image Annotator (VIA)**

## Creating Your Dataset

### Step 1: Prepare Images
1. Collect or generate Urdu text images
2. Ensure images are clear and readable
3. Save in a folder (e.g., `all_images/`)

### Step 2: Create Labels
1. For each image, transcribe the Urdu text
2. Create a text file with image-text pairs
3. Format: `image_name.jpg<TAB>urdu_text`

### Step 3: Split Dataset
```python
import os
import random
import shutil

def split_dataset(source_images, source_labels, train_ratio=0.8):
    # Read labels
    with open(source_labels, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Shuffle
    random.shuffle(lines)
    
    # Split
    split_idx = int(len(lines) * train_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]
    
    # Create directories
    os.makedirs('data/train/images', exist_ok=True)
    os.makedirs('data/validation/images', exist_ok=True)
    
    # Write train labels
    with open('data/train/labels.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    # Write validation labels
    with open('data/validation/labels.txt', 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    
    # Copy images
    for line in train_lines:
        img_name = line.split('\t')[0]
        src = os.path.join(source_images, img_name)
        dst = os.path.join('data/train/images', img_name)
        shutil.copy(src, dst)
    
    for line in val_lines:
        img_name = line.split('\t')[0]
        src = os.path.join(source_images, img_name)
        dst = os.path.join('data/validation/images', img_name)
        shutil.copy(src, dst)

# Usage
split_dataset('all_images/', 'all_labels.txt', train_ratio=0.8)
```

### Step 4: Verify Dataset
Run the dataset verification script:
```bash
python urdu_ocr/dataset.py
```

This will:
- Load and analyze your dataset
- Show statistics (number of samples, text lengths, etc.)
- Display sample images with labels

## Data Quality Tips

1. **Consistent Quality**: Ensure consistent image quality across dataset
2. **Balanced Data**: Include variety of:
   - Font styles
   - Text lengths
   - Background types
   - Noise levels
3. **Clean Labels**: Double-check text transcriptions
4. **Diverse Vocabulary**: Include wide range of Urdu words
5. **Special Characters**: Include diacritics, punctuation

## Augmentation

The model automatically applies augmentation during training:
- Gaussian noise
- Blur effects
- Brightness/contrast adjustments
- Slight rotations
- Elastic transformations

You don't need to manually augment images.

## Example Dataset Creation Script

```python
# create_dataset.py
import os
from PIL import Image, ImageDraw, ImageFont
import random

# Sample Urdu sentences
urdu_sentences = [
    "یہ ایک ٹیسٹ ہے",
    "اردو زبان بہت خوبصورت ہے",
    "پاکستان زندہ باد",
    # Add more sentences...
]

def create_sample_dataset(num_images=1000):
    os.makedirs('data/train/images', exist_ok=True)
    
    labels = []
    font_path = "path/to/urdu/font.ttf"  # Update this
    
    for i in range(num_images):
        # Random sentence
        text = random.choice(urdu_sentences)
        
        # Create image
        img = Image.new('RGB', (256, 64), color='white')
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, random.randint(24, 36))
        
        # Random position
        x = random.randint(5, 20)
        y = random.randint(5, 15)
        draw.text((x, y), text, font=font, fill='black')
        
        # Save
        img_name = f"img_{i:05d}.jpg"
        img.save(f"data/train/images/{img_name}")
        
        # Add to labels
        labels.append(f"{img_name}\t{text}\n")
    
    # Save labels
    with open('data/train/labels.txt', 'w', encoding='utf-8') as f:
        f.writelines(labels)
    
    print(f"Created {num_images} sample images")

if __name__ == "__main__":
    create_sample_dataset(1000)
```

## Troubleshooting

### Issue: Labels file not loading
- Check encoding (must be UTF-8)
- Verify TAB character separator
- Check for extra spaces

### Issue: Images not found
- Verify image paths match labels
- Check file extensions (case-sensitive)
- Ensure images are in `images/` subfolder

### Issue: Empty dataset
- Verify directory structure
- Check labels.txt format
- Ensure images exist and are readable

## Next Steps

After preparing your dataset:
1. Verify dataset: `python urdu_ocr/dataset.py`
2. Start training: `python urdu_ocr/train.py`
3. Monitor progress with TensorBoard: `tensorboard --logdir=logs`

## Additional Resources

- Urdu Fonts: Jameel Noori Nastaleeq, Nafees Nastaleeq
- Urdu Unicode Range: U+0600 to U+06FF (Arabic)
- Python Urdu libraries: `urduhack`, `urdu`

## Questions?

If you encounter issues, check:
1. File paths are correct
2. UTF-8 encoding is used
3. Image files are readable
4. Labels format is correct (TAB-separated)

Happy training! 🚀
