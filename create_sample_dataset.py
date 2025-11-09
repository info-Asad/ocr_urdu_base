"""
Example script to create a small sample dataset for testing
"""
import os
from PIL import Image, ImageDraw, ImageFont
import random

# Create directories
os.makedirs('data/train/images', exist_ok=True)
os.makedirs('data/validation/images', exist_ok=True)

# Sample Urdu sentences
urdu_sentences = [
    "یہ ایک ٹیسٹ ہے",
    "اردو زبان",
    "پاکستان زندہ باد",
    "سلام",
    "خوش آمدید",
    "شکریہ",
    "اللہ حافظ",
    "میرا نام",
    "آپ کیسے ہیں",
    "بہت اچھا",
]

def create_sample_image(text, output_path, font_size=32):
    """Create a simple text image without requiring Urdu font"""
    # Create white background
    img = Image.new('RGB', (256, 64), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use system font that supports Urdu
        # On Windows, Arial or Tahoma might work
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("tahoma.ttf", font_size)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
    
    # Draw text (right-to-left for Urdu)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (256 - text_width) // 2
    y = (64 - text_height) // 2
    
    draw.text((x, y), text, font=font, fill='black')
    
    # Save
    img.save(output_path)


def create_sample_dataset(num_train=50, num_val=10):
    """Create a small sample dataset"""
    print("Creating sample dataset...")
    print("Note: For actual training, you need a real dataset with thousands of images.")
    print("See DATASET_GUIDE.md for instructions.\n")
    
    # Training data
    train_labels = []
    for i in range(num_train):
        text = random.choice(urdu_sentences)
        img_name = f"train_{i:04d}.jpg"
        img_path = f"data/train/images/{img_name}"
        
        create_sample_image(text, img_path)
        train_labels.append(f"{img_name}\t{text}\n")
    
    with open('data/train/labels.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_labels)
    
    # Validation data
    val_labels = []
    for i in range(num_val):
        text = random.choice(urdu_sentences)
        img_name = f"val_{i:04d}.jpg"
        img_path = f"data/validation/images/{img_name}"
        
        create_sample_image(text, img_path)
        val_labels.append(f"{img_name}\t{text}\n")
    
    with open('data/validation/labels.txt', 'w', encoding='utf-8') as f:
        f.writelines(val_labels)
    
    print(f"Created {num_train} training samples")
    print(f"Created {num_val} validation samples")
    print("\nDataset created in 'data/' directory")
    print("\nWARNING: This is just a tiny sample for testing!")
    print("For real training, prepare a proper dataset with 5,000+ images.")
    print("See DATASET_GUIDE.md for details.")


if __name__ == "__main__":
    create_sample_dataset(num_train=50, num_val=10)
