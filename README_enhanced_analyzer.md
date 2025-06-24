# Enhanced Chess Position Analyzer

A complete chess position analysis system that combines board detection, square extraction, and piece classification with support for both single images and batch processing.

## Features

### 🎯 Core Capabilities
- **Chess Board Detection**: Automatically detects and extracts chess boards from images
- **Square Extraction**: Precisely extracts individual squares from the detected board
- **Piece Classification**: Classifies pieces in each square using a trained DINOv2-based model
- **Position Analysis**: Generates standard chess notation and position analysis

### 🚀 Performance Optimizations
- **Batch Processing**: Process multiple images efficiently with optimized GPU utilization
- **Smart Batching**: Automatically batches square classifications for optimal performance
- **Memory Efficient**: Processes large batches without memory issues
- **Multi-Device Support**: Automatic device detection (CPU, CUDA, MPS)

### 📊 Analysis Features
- **Confidence Scoring**: Provides confidence scores for each piece prediction
- **Statistics**: Detailed statistics about piece counts, confidence levels, and analysis quality
- **Debug Visualizations**: Save intermediate results and visualizations for debugging
- **Error Handling**: Robust error handling with detailed error reporting

## Installation

### Prerequisites
```bash
# Install required packages
pip install -r requirements.txt

# For training (if needed)
pip install -r train_requirements.txt
```

### Model Setup
The analyzer requires a trained chess piece classifier model. If you don't have one:

1. **Use the provided model**: The repository includes a pre-trained model (`chess_piece_classifier.pth`)
2. **Train your own**: Run the training script to create a custom model

```bash
# Train a new model (optional)
python chess_piece_classifier.py
```

## Usage

### Command Line Interface

#### Single Image Analysis
```bash
# Basic analysis
python chess_position_analyzer.py path/to/chess_image.jpg

# With custom confidence threshold
python chess_position_analyzer.py path/to/chess_image.jpg --confidence 0.7

# With debug visualizations
python chess_position_analyzer.py path/to/chess_image.jpg --debug

# Specify device
python chess_position_analyzer.py path/to/chess_image.jpg --device cuda
```

#### Batch Processing
```bash
# Process all images in a directory
python chess_position_analyzer.py path/to/image/directory --batch

# Process specific images
python chess_position_analyzer.py image1.jpg image2.jpg image3.jpg --batch

# With custom output directory and summary
python chess_position_analyzer.py path/to/directory --batch --output-dir results --summary
```

#### Advanced Options
```bash
# Full batch processing with all features
python chess_position_analyzer.py path/to/directory \
    --batch \
    --confidence 0.6 \
    --device cuda \
    --debug \
    --output-dir analysis_results \
    --summary \
    --quiet
```

### Python API

#### Single Image Analysis
```python
from chess_position_analyzer import ChessPositionAnalyzer

# Initialize analyzer
analyzer = ChessPositionAnalyzer(
    model_path="chess_piece_classifier.pth",
    device="auto"  # or "cpu", "cuda", "mps"
)

# Analyze single image
result = analyzer.analyze_position(
    image_path="chess_board.jpg",
    confidence_threshold=0.5,
    save_debug=True
)

# Print results
analyzer.print_analysis(result)

# Access results
print(f"Position: {result['position_string']}")
print(f"Piece predictions: {result['piece_predictions']}")
print(f"Statistics: {result['statistics']}")
```

#### Batch Processing
```python
# Process multiple images
image_paths = ["board1.jpg", "board2.jpg", "board3.jpg"]
results = analyzer.analyze_batch(
    image_paths=image_paths,
    confidence_threshold=0.5,
    save_debug=True,
    debug_dir="batch_output"
)

# Process directory
results = analyzer.analyze_batch(
    image_paths="path/to/chess/images/",
    confidence_threshold=0.5,
    save_debug=True
)

# Process results
for result in results:
    if result.get('board_detected', False):
        print(f"✅ {result['image_path']}: {result['statistics']['total_pieces']} pieces")
    else:
        print(f"❌ {result['image_path']}: {result.get('error', 'Unknown error')}")
```

## Output Format

### Analysis Results
Each analysis returns a dictionary with the following structure:

```python
{
    'position_grid': List[List[str]],      # 8x8 grid representation
    'position_string': str,                # Formatted position string
    'piece_predictions': Dict,             # Detailed predictions for each square
    'statistics': Dict,                    # Analysis statistics
    'board_detected': bool,                # Success flag
    'total_squares': int,                  # Number of squares processed
    'image_path': str                      # Path to analyzed image
}
```

### Piece Predictions
Each square prediction contains:
```python
{
    'predicted_class': str,        # Class name (e.g., 'wp', 'br', 'empty')
    'confidence': float,           # Confidence score (0.0-1.0)
    'all_probabilities': List,     # All class probabilities
    'piece_symbol': str            # Chess notation symbol (e.g., 'p', 'R', '·')
}
```

### Statistics
Analysis statistics include:
```python
{
    'piece_counts': Dict,          # Count of each piece type
    'average_confidence': float,   # Average confidence across all squares
    'min_confidence': float,       # Minimum confidence score
    'max_confidence': float,       # Maximum confidence score
    'low_confidence_squares': List, # Squares with low confidence
    'total_pieces': int            # Total number of pieces detected
}
```

## Performance

### Batch Processing Benefits
- **Speedup**: 2-4x faster than processing images individually
- **GPU Utilization**: Optimized batch sizes for maximum GPU efficiency
- **Memory Management**: Efficient memory usage for large batches
- **Error Isolation**: Individual image failures don't affect the entire batch

### Recommended Batch Sizes
- **GPU (CUDA/MPS)**: 32-64 images per batch
- **CPU**: 8-16 images per batch
- **Memory-constrained**: 4-8 images per batch

## Debug and Visualization

### Debug Outputs
When `--debug` is enabled, the analyzer saves:
- Original image
- Detected board
- Board with grid overlay
- Individual square images
- Prediction visualization
- Confidence heatmap
- Classification debug information

### Debug Directory Structure
```
debug_output/
├── 01_original_image.png
├── 02_detected_board.png
├── 03_board_with_grid.png
├── 04_predictions_visualization.png
├── squares/
│   ├── a1.png
│   ├── a2.png
│   └── ...
├── classification_debug.txt
└── confidence_heatmap.png
```

## Testing

Run the comprehensive test suite:
```bash
python test_batch_analyzer.py
```

This will test:
- Single image analysis
- Batch processing
- Directory processing
- Performance comparison

## Troubleshooting

### Common Issues

1. **Model not found**
   ```
   FileNotFoundError: Model file not found: chess_piece_classifier.pth
   ```
   **Solution**: Ensure the model file exists or train a new one.

2. **No chess board detected**
   ```
   ValueError: No chess board detected in the image
   ```
   **Solution**: Check image quality, lighting, and board visibility.

3. **Low confidence predictions**
   - Adjust confidence threshold: `--confidence 0.3`
   - Check image quality and piece visibility
   - Consider retraining the model with similar images

4. **Memory issues with large batches**
   - Reduce batch size by processing fewer images at once
   - Use CPU instead of GPU: `--device cpu`
   - Process images in smaller groups

### Performance Tips

1. **Use GPU acceleration** when available
2. **Batch process** multiple images for better efficiency
3. **Adjust confidence threshold** based on your needs
4. **Use appropriate image resolution** (224x224 is optimal)
5. **Disable debug output** for faster processing

## Model Information

### Chess Piece Classes
The model recognizes 13 classes:
- `empty`: Empty square
- `wp`, `wr`, `wn`, `wb`, `wq`, `wk`: White pieces (pawn, rook, knight, bishop, queen, king)
- `bp`, `br`, `bn`, `bb`, `bq`, `bk`: Black pieces (pawn, rook, knight, bishop, queen, king)

### Model Architecture
- **Backbone**: DINOv2-base (frozen)
- **Classifier**: 3-layer linear head (768 → 128 → 64 → 13)
- **Input Size**: 224x224 pixels
- **Output**: 13-class probabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DINOv2 model from Facebook Research
- Chess piece detection algorithms
- OpenCV for image processing
- PyTorch for deep learning framework 