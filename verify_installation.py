"""
Installation Verification Script
Run this script to verify everything is set up correctly
"""
import sys
import os

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'albumentations': 'Albumentations',
        'tqdm': 'tqdm',
        'matplotlib': 'Matplotlib',
    }
    
    missing = []
    installed = []
    
    for module, name in required.items():
        try:
            __import__(module)
            installed.append(name)
            print(f"✓ {name} - OK")
        except ImportError:
            missing.append(name)
            print(f"✗ {name} - NOT FOUND")
    
    return len(missing) == 0, missing


def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA (GPU support)...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("⚠ CUDA not available - will use CPU (slower)")
            return False
    except:
        print("✗ Could not check CUDA")
        return False


def check_project_structure():
    """Check if project structure is correct"""
    print("\nChecking project structure...")
    
    required_files = [
        'urdu_ocr/config.py',
        'urdu_ocr/model.py',
        'urdu_ocr/dataset.py',
        'urdu_ocr/data_preprocessing.py',
        'urdu_ocr/train.py',
        'urdu_ocr/predict.py',
        'urdu_ocr/utils.py',
        'requirements.txt',
        'README.md',
    ]
    
    required_dirs = [
        'urdu_ocr',
        'data',
        'models',
        'logs',
    ]
    
    all_ok = True
    
    # Check files
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - NOT FOUND")
            all_ok = False
    
    # Check directories
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ - NOT FOUND")
            all_ok = False
    
    return all_ok


def check_dataset():
    """Check if dataset is present"""
    print("\nChecking dataset...")
    
    train_images = os.path.join('data', 'train', 'images')
    train_labels = os.path.join('data', 'train', 'labels.txt')
    val_images = os.path.join('data', 'validation', 'images')
    val_labels = os.path.join('data', 'validation', 'labels.txt')
    
    dataset_ok = True
    
    if os.path.exists(train_images):
        num_train = len([f for f in os.listdir(train_images) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"✓ Training images folder: {num_train} images")
    else:
        print(f"✗ Training images folder - NOT FOUND")
        dataset_ok = False
    
    if os.path.exists(train_labels):
        with open(train_labels, 'r', encoding='utf-8') as f:
            num_labels = len(f.readlines())
        print(f"✓ Training labels file: {num_labels} labels")
    else:
        print(f"✗ Training labels file - NOT FOUND")
        dataset_ok = False
    
    if os.path.exists(val_images):
        num_val = len([f for f in os.listdir(val_images) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"✓ Validation images folder: {num_val} images")
    else:
        print(f"✗ Validation images folder - NOT FOUND")
        dataset_ok = False
    
    if os.path.exists(val_labels):
        with open(val_labels, 'r', encoding='utf-8') as f:
            num_val_labels = len(f.readlines())
        print(f"✓ Validation labels file: {num_val_labels} labels")
    else:
        print(f"✗ Validation labels file - NOT FOUND")
        dataset_ok = False
    
    return dataset_ok


def test_import_modules():
    """Test importing project modules"""
    print("\nTesting project modules...")
    
    modules = ['config', 'model', 'dataset', 'data_preprocessing', 'train', 'predict', 'utils']
    
    sys.path.insert(0, 'urdu_ocr')
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}.py - OK")
        except Exception as e:
            print(f"✗ {module}.py - ERROR: {e}")
            all_ok = False
    
    return all_ok


def main():
    """Main verification function"""
    print("="*70)
    print("URDU OCR - INSTALLATION VERIFICATION")
    print("="*70)
    
    results = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies()[0],
        'CUDA': check_cuda(),
        'Project Structure': check_project_structure(),
        'Dataset': check_dataset(),
        'Module Imports': test_import_modules(),
    }
    
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    for check, status in results.items():
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"{check:.<50} {status_str}")
    
    print("="*70)
    
    # Overall status
    if all(results.values()):
        print("\n✓ ALL CHECKS PASSED - SYSTEM READY!")
        print("\nNext steps:")
        print("1. If dataset not found, prepare dataset (see DATASET_GUIDE.md)")
        print("2. Start training: python urdu_ocr/train.py")
        print("3. Monitor: tensorboard --logdir=logs")
    elif results['Python Version'] and results['Dependencies'] and results['Project Structure']:
        print("\n⚠ SYSTEM PARTIALLY READY")
        
        if not results['Dataset']:
            print("\n⚠ Dataset not found")
            print("   → Prepare dataset according to DATASET_GUIDE.md")
            print("   → Or create sample: python create_sample_dataset.py")
        
        if not results['CUDA']:
            print("\n⚠ CUDA not available")
            print("   → Training will use CPU (much slower)")
            print("   → Consider using GPU for faster training")
        
        print("\nOnce dataset is ready:")
        print("  python urdu_ocr/train.py")
    else:
        print("\n✗ INSTALLATION INCOMPLETE")
        
        if not results['Dependencies']:
            print("\nMissing dependencies detected.")
            print("Run: pip install -r requirements.txt")
        
        if not results['Project Structure']:
            print("\nProject structure incomplete.")
            print("Ensure all files are present.")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
