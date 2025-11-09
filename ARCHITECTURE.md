# Neural Network Architecture Diagram

## Complete System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT IMAGE                              │
│                    (Urdu Text - 256x64x3)                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   IMAGE PREPROCESSING                            │
│  • Resize to 256x64                                             │
│  • Normalize [0-1]                                              │
│  • Data Augmentation (training only)                            │
│    - Gaussian noise, blur                                       │
│    - Brightness/contrast adjustment                             │
│    - Slight rotation                                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CNN FEATURE EXTRACTOR                           │
│                                                                  │
│  Conv Block 1:  64 filters  → MaxPool(2,2)  → 128x32           │
│  Conv Block 2: 128 filters  → MaxPool(2,2)  → 64x16            │
│  Conv Block 3: 256 filters  → MaxPool(2,1)  → 32x16            │
│  Conv Block 4: 512 filters  → MaxPool(2,1)  → 16x16            │
│  Conv Block 5: 512 filters  → MaxPool(4,1)  → 1x32             │
│                                                                  │
│  Output: Sequence of 32 feature vectors (512-dim each)          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              BIDIRECTIONAL LSTM (2 layers)                       │
│                                                                  │
│  Forward LSTM  →→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→               │
│                                                                  │
│  Backward LSTM ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←               │
│                                                                  │
│  • Hidden size: 256                                             │
│  • Bidirectional: 256 × 2 = 512 output                         │
│  • Dropout: 0.3                                                 │
│                                                                  │
│  Output: Sequence of 32 vectors (512-dim each)                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FULLY CONNECTED LAYER                           │
│                                                                  │
│  Input: 512 dimensions                                          │
│  Output: Vocabulary size (~100 classes)                         │
│  Activation: Log Softmax                                        │
│                                                                  │
│  Output: Log probabilities for each character at each position  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CTC DECODING                                │
│                                                                  │
│  • Collapse repeated characters                                 │
│  • Remove blank tokens                                          │
│  • Convert indices to characters                                │
│                                                                  │
│  Example:                                                        │
│    [0,1,1,0,2,2,2,0,3,0] → [1,2,3] → "یہ"                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT TEXT                                 │
│                    (Recognized Urdu Text)                        │
└─────────────────────────────────────────────────────────────────┘
```

## Training Flow

```
┌─────────────┐
│   Dataset   │ (Images + Labels)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Data Loader │ (Batching + Shuffling)
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│         Training Loop               │
│                                     │
│  For each batch:                    │
│    1. Forward Pass                  │
│       Image → CNN → LSTM → Output   │
│                                     │
│    2. Loss Calculation              │
│       CTC Loss(Output, Target)      │
│                                     │
│    3. Backward Pass                 │
│       Compute Gradients             │
│                                     │
│    4. Optimization                  │
│       Update Weights (Adam)         │
│                                     │
│  After each epoch:                  │
│    5. Validation                    │
│       Evaluate on validation set    │
│                                     │
│    6. Metrics                       │
│       Calculate CER, WER, Accuracy  │
│                                     │
│    7. Checkpointing                 │
│       Save best model               │
│                                     │
│    8. Early Stopping Check          │
│       Stop if no improvement        │
└─────────────────────────────────────┘
```

## Data Flow Dimensions

```
Stage                     Shape                    Description
─────────────────────────────────────────────────────────────────
Input Image              (B, 3, 64, 256)          RGB image

After Conv Block 1       (B, 64, 32, 128)         64 channels
After Conv Block 2       (B, 128, 16, 64)         128 channels
After Conv Block 3       (B, 256, 8, 32)          256 channels
After Conv Block 4       (B, 512, 4, 32)          512 channels
After Conv Block 5       (B, 512, 1, 32)          Height collapsed

CNN Output (reshaped)    (B, 32, 512)             Sequence features

After BiLSTM             (B, 32, 512)             Bidirectional output

After FC Layer           (B, 32, ~100)            Class logits

After Log Softmax        (32, B, ~100)            Log probabilities
                                                   (transposed for CTC)

After CTC Decoding       Variable length           Text output

B = Batch size (default: 32)
```

## Architecture Breakdown

### CNN Feature Extractor
```
┌──────────────────────────┐
│  Input: 3×64×256         │
├──────────────────────────┤
│  Conv2d(3→64, 3×3)      │
│  BatchNorm + ReLU        │
│  Conv2d(64→64, 3×3)     │
│  BatchNorm + ReLU        │
│  MaxPool(2×2)            │
├──────────────────────────┤
│  Conv2d(64→128, 3×3)    │
│  BatchNorm + ReLU        │
│  Conv2d(128→128, 3×3)   │
│  BatchNorm + ReLU        │
│  MaxPool(2×2)            │
├──────────────────────────┤
│  Conv2d(128→256, 3×3)   │
│  BatchNorm + ReLU        │
│  Conv2d(256→256, 3×3)   │
│  BatchNorm + ReLU        │
│  MaxPool(2×1)            │
├──────────────────────────┤
│  Conv2d(256→512, 3×3)   │
│  BatchNorm + ReLU        │
│  Conv2d(512→512, 3×3)   │
│  BatchNorm + ReLU        │
│  MaxPool(2×1)            │
├──────────────────────────┤
│  Conv2d(512→512, 3×3)   │
│  BatchNorm + ReLU        │
│  MaxPool(4×1)            │
├──────────────────────────┤
│  Output: 512×1×32        │
│  (Reshape to 32×512)     │
└──────────────────────────┘
```

### Bidirectional LSTM
```
┌──────────────────────────┐
│  Input: 32×512           │
├──────────────────────────┤
│  LSTM Layer 1            │
│  Forward:  256 units     │
│  Backward: 256 units     │
│  Output: 512 (concat)    │
├──────────────────────────┤
│  LSTM Layer 2            │
│  Forward:  256 units     │
│  Backward: 256 units     │
│  Output: 512 (concat)    │
├──────────────────────────┤
│  Dropout: 0.3            │
├──────────────────────────┤
│  Output: 32×512          │
└──────────────────────────┘
```

### Output Layer
```
┌──────────────────────────┐
│  Input: 512              │
├──────────────────────────┤
│  Linear(512 → ~100)      │
│  (Vocabulary size)       │
├──────────────────────────┤
│  Log Softmax             │
├──────────────────────────┤
│  Output: ~100 classes    │
│  (Character probabilities)│
└──────────────────────────┘
```

## CTC Loss Explanation

```
CTC (Connectionist Temporal Classification)

Problem:
- Input sequence length ≠ Output sequence length
- No character-level alignment needed
- Handles variable-length sequences

Solution:
- Add "blank" token (index 0)
- Allow repetitions
- Collapse sequences

Example:
Target: "یہ"  (2 characters)
Input: 32 time steps

Possible alignments:
- "___یہ_______________________"
- "___ییہہ_____________________"
- "_____یییییییہہہ_____________"
All decode to: "یہ"

CTC Loss:
- Computes probability of all valid alignments
- Maximizes likelihood of correct text
```

## Model Parameters

```
Component              Parameters      Percentage
──────────────────────────────────────────────────
Conv Layers            ~3M            15-20%
BatchNorm Layers       ~50K           <1%
LSTM Layers            ~12M           60-70%
FC Layer               ~50K           <1%
──────────────────────────────────────────────────
Total                  ~15M           100%

Memory Usage (approximate):
- Model: ~60 MB (FP32)
- Training batch (32): ~100 MB
- Total GPU memory: ~2-4 GB
```

## Inference Flow

```
┌───────────────┐
│  Load Image   │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Preprocess   │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Load Model   │
└───────┬───────┘
        │
        ▼
┌───────────────────────┐
│  Forward Pass         │
│  (No gradient)        │
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  Get Predictions      │
│  (argmax at each pos) │
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  CTC Decode           │
│  (collapse + remove)  │
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  Convert to Text      │
│  (indices → chars)    │
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  Return Text          │
└───────────────────────┘
```

---

This architecture is specifically designed for:
- ✅ Variable-length text recognition
- ✅ No character segmentation needed
- ✅ Robust to noise and variations
- ✅ Sequential modeling (important for Urdu)
- ✅ End-to-end differentiable training
