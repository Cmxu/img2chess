# Enhanced Chess Position Analyzer with Debug Output

The chess position analyzer now includes comprehensive debug functionality that saves PNG images for each step of the analysis process, making it easier to understand and debug the chess board detection and piece classification pipeline.

## Debug Output Overview

When you run the analyzer with `--debug` flag or `save_debug=True`, it creates a `debug_output/` directory containing:

### 📁 Main Debug Images (PNG files)

1. **`01_original_image.png`** - The original input image
2. **`02_detected_board.png`** - The chess board after detection and cropping
3. **`03_board_with_grid.png`** - Board with grid lines and square labels overlaid
4. **`04_predictions_visualization.png`** - Board with piece predictions and confidence scores
5. **`05_confidence_heatmap.png`** - Confidence heatmap with piece symbols

### 📁 Individual Square Images

- **`squares/`** directory containing 64 individual PNG files:
  - `a1.png`, `a2.png`, ..., `h8.png`
  - Each file shows the extracted square image that gets classified

### 📄 Debug Text Files

- **`classification_debug.txt`** - Detailed classification results for each square
- **`wrong_predictions_summary.txt`** - Summary of any classification errors (if applicable)

## Usage

### Command Line Usage

```bash
# Basic usage with debug output
python chess_position_analyzer.py sample_chess_frame.jpg --debug

# With custom confidence threshold
python chess_position_analyzer.py sample_chess_frame.jpg --debug --confidence 0.7

# With specific device
python chess_position_analyzer.py sample_chess_frame.jpg --debug --device cuda
```

### Python API Usage

```python
from chess_position_analyzer import ChessPositionAnalyzer

# Initialize analyzer
analyzer = ChessPositionAnalyzer(
    model_path="chess_piece_classifier.pth",
    device="auto"
)

# Analyze with debug output
result = analyzer.analyze_position(
    image_path="sample_chess_frame.jpg",
    confidence_threshold=0.5,
    save_debug=True  # Enable debug output
)
```

### Test Script

Use the provided test script for quick testing:

```bash
python test_debug_analyzer.py sample_chess_frame.jpg
```

## Debug Output Details

### 1. Original Image (`01_original_image.png`)
- Shows the exact input image as loaded by OpenCV
- Useful for verifying the input quality and format

### 2. Detected Board (`02_detected_board.png`)
- The chess board after detection and perspective correction
- Shows what the board detector found and extracted
- Helps debug board detection issues

### 3. Board with Grid (`03_board_with_grid.png`)
- The detected board with grid lines and square labels
- Shows how the board is divided into 64 squares
- Square labels follow chess notation (a1, a2, ..., h8)

### 4. Predictions Visualization (`04_predictions_visualization.png`)
- Board with piece predictions overlaid on each square
- Shows:
  - Piece symbols (p, r, n, b, q, k for white; P, R, N, B, Q, K for black)
  - Confidence scores in green text
  - Color-coded borders (white/black/gray)
- Helps visualize classification results at a glance

### 5. Confidence Heatmap (`05_confidence_heatmap.png`)
- Color-coded heatmap showing confidence levels
- Green = high confidence, Red = low confidence
- Includes piece symbols and confidence values
- Helps identify problematic squares

### 6. Individual Squares (`squares/` directory)
- 64 PNG files, one for each square
- Each file is named by chess notation (a1.png, a2.png, etc.)
- Shows exactly what the classifier sees for each square
- Useful for debugging classification errors

### 7. Classification Debug (`classification_debug.txt`)
- Detailed text output for each square
- Shows:
  - Predicted class and confidence
  - Top 3 probability scores for all classes
  - Piece symbol mapping
- Helps understand why certain classifications were made

## Debug Output Example

```
debug_output/
├── 01_original_image.png
├── 02_detected_board.png
├── 03_board_with_grid.png
├── 04_predictions_visualization.png
├── 05_confidence_heatmap.png
├── classification_debug.txt
└── squares/
    ├── a1.png
    ├── a2.png
    ├── ...
    └── h8.png
```

## Analyzing Debug Output

### Board Detection Issues
1. Check `01_original_image.png` - Is the input image clear?
2. Check `02_detected_board.png` - Was the board detected correctly?
3. Check `03_board_with_grid.png` - Are the squares properly aligned?

### Classification Issues
1. Check `04_predictions_visualization.png` - Which squares have wrong predictions?
2. Check `05_confidence_heatmap.png` - Which squares have low confidence?
3. Check individual square images in `squares/` - What does the classifier see?
4. Check `classification_debug.txt` - What were the top probabilities?

### Common Debug Scenarios

#### Low Confidence Predictions
- Look for red areas in the confidence heatmap
- Check the individual square images for poor quality
- Review the classification debug file for alternative predictions

#### Wrong Piece Classifications
- Compare the square image with the prediction
- Check if similar pieces are being confused (e.g., bishops vs knights)
- Look for color confusion (white vs black pieces)

#### Board Detection Problems
- Check if the original image has good lighting and contrast
- Verify the board is clearly visible and not occluded
- Look for perspective issues in the detected board

## Tips for Effective Debugging

1. **Start with the heatmap** - It gives a quick overview of problematic areas
2. **Check individual squares** - Look at the actual images being classified
3. **Review confidence scores** - Low confidence often indicates poor image quality
4. **Compare predictions** - Look for patterns in classification errors
5. **Use the test script** - It provides a structured way to run and review debug output

## Troubleshooting

### No Debug Output Generated
- Ensure `save_debug=True` is passed to `analyze_position()`
- Check that the `debug_output/` directory is writable
- Verify that the analysis completed successfully

### Missing Files
- Some files may not be generated if the analysis fails at certain steps
- Check the console output for error messages
- Ensure all required dependencies are installed

### Poor Quality Debug Images
- Check the input image quality
- Verify that the board detection is working correctly
- Consider adjusting the confidence threshold

## Performance Notes

- Debug output adds some overhead to the analysis
- PNG files can be large, especially for high-resolution images
- Consider disabling debug output for production use
- The debug directory is cleared at the start of each run to avoid confusion 