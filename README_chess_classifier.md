# Chess Piece Classifier

A deep learning system for classifying chess pieces using DINOv2 feature extractor and synthetic data generation.

## Overview

This system classifies chess pieces into 13 classes:
- **Empty square** (1 class)
- **Chess pieces** (12 classes): White and Black versions of Pawn, Rook, Knight, Bishop, Queen, King

The model uses Facebook's DINOv2 as a feature extractor with a custom 3-layer linear classifier head.

## Key Features

- **Synthetic Data Generation**: Creates training data on-the-fly by combining chess piece images with board backgrounds
- **Data Augmentation**: Applies random transformations (rotation, translation, brightness, contrast, saturation)
- **Multiple Piece Styles**: Supports 38+ different piece visual styles
- **Multiple Board Themes**: Uses various board backgrounds for diversity
- **Transfer Learning**: Leverages pre-trained DINOv2 for robust visual feature extraction

## Installation

1. Install dependencies:
```bash
pip install -r train_requirements.txt
```

2. Ensure you have the following directory structure:
```
chess_pieces/
├── classic/
│   ├── wp.png, wr.png, wn.png, wb.png, wq.png, wk.png
│   └── bp.png, br.png, bn.png, bb.png, bq.png, bk.png
├── wood/
├── metal/
└── ... (other styles)

boards/
├── brown_w.png, brown_b.png
├── wood_w.png, wood_b.png
└── ... (other board themes)
```

## Usage

### 1. Test Data Generation

Before training, verify that the data generation pipeline works:

```bash
python test_data_generation.py
```

This will:
- Check piece availability across styles
- Generate sample images
- Show class distribution
- Save test samples as `test_samples.png`

### 2. Train the Model

Run the training script:

```bash
python chess_piece_classifier.py
```

Training parameters:
- **Epochs**: 15
- **Batch Size**: 32
- **Learning Rate**: 0.001 (with step decay)
- **Training Samples**: 8,000 per epoch
- **Validation Samples**: 2,000 per epoch
- **Image Size**: 224×224 pixels (DINOv2 standard)

The script will:
- Generate training data on-the-fly
- Train the model with validation
- Save the trained model as `chess_piece_classifier.pth`
- Generate training curves plots
- Show sample images during training

### 3. Run Inference

Use the trained model to classify new images:

```bash
# Single image
python inference.py --image path/to/chess_image.png

# Multiple images
python inference.py --images image1.png image2.png image3.png

# Use GPU acceleration (auto-detects MPS/CUDA/CPU)
python inference.py --image path/to/chess_image.png --device auto

# Or specify device explicitly
python inference.py --image path/to/chess_image.png --device mps   # Apple Silicon
python inference.py --image path/to/chess_image.png --device cuda  # NVIDIA GPU
```

## Model Architecture

```
Input (224×224×3) 
    ↓
DINOv2 Feature Extractor (frozen)
    ↓
CLS Token Extraction → 768 features
    ↓
Linear(768 → 512) + ReLU + Dropout(0.2)
    ↓
Linear(512 → 256) + ReLU + Dropout(0.2)
    ↓
Linear(256 → 13) → 13 classes
```

## Classes

| Class ID | Label | Description |
|----------|-------|-------------|
| 0 | empty | Empty square |
| 1 | wp | White Pawn |
| 2 | wr | White Rook |
| 3 | wn | White Knight |
| 4 | wb | White Bishop |
| 5 | wq | White Queen |
| 6 | wk | White King |
| 7 | bp | Black Pawn |
| 8 | br | Black Rook |
| 9 | bn | Black Knight |
| 10 | bb | Black Bishop |
| 11 | bq | Black Queen |
| 12 | bk | Black King |

## Data Generation Process

For each training sample:

1. **Choose class**: 1/13 probability for empty square, 12/13 for chess pieces
2. **Select piece**: Random piece from random style directory
3. **Select background**: Random board background (light/dark square)
4. **Compose image**: Place piece (80% of square size) centered on background
5. **Apply augmentations** (if enabled):
   - Random rotation (±10°)
   - Random translation (±10% of image size)
   - Random brightness (0.8-1.2×)
   - Random contrast (0.8-1.2×)
   - Random saturation (0.8-1.2×)

## Expected Performance

The model should achieve:
- **Training accuracy**: >95% after 10-15 epochs
- **Validation accuracy**: >90% with good generalization
- **Quick test (3 epochs)**: ~88% validation accuracy
- **Inference speed**: ~10-50ms per image (depending on hardware)
- **Trainable parameters**: ~528K parameters

## Customization

### Modify Classes
Edit the `classes` list in `ChessPieceDataset.__init__()`:
```