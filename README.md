# img2chess

A Python library for extracting chess boards from images and splitting them into individual squares using computer vision techniques.

## Features

- 🔍 **Automatic Chess Board Detection**: Uses advanced computer vision techniques to detect chess boards in images
- 📐 **Perspective Correction**: Automatically corrects perspective distortion to extract a perfect square board
- ✂️ **Square Extraction**: Splits the board into 64 individual squares with proper chess notation (a1-h8)
- 🎯 **Multiple Detection Methods**: Combines corner detection and contour analysis for robust board detection
- 📊 **Visualization Tools**: Built-in tools for visualizing results with grid lines and square labels
- 🔧 **Preprocessing Pipeline**: Image enhancement utilities for better detection accuracy

## Installation

### From PyPI (coming soon)
```bash
pip install img2chess
```

### From Source
```bash
git clone https://github.com/yourusername/img2chess.git
cd img2chess
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
import cv2
from img2chess import ChessBoardDetector, SquareExtractor, visualize_board, save_squares

# Load an image containing a chess board
image = cv2.imread('chess_board_image.jpg')

# Initialize the detector and extractor
detector = ChessBoardDetector()
extractor = SquareExtractor(square_size=64)

# Detect and extract the chess board
board_image = detector.detect_board(image)

if board_image is not None:
    # Extract all 64 squares
    squares = extractor.extract_squares(board_image)
    
    # Save individual squares
    save_squares(squares, 'output_squares/')
    
    # Create visualization
    visualization = visualize_board(board_image, show_grid=True, show_labels=True)
    cv2.imwrite('board_visualization.jpg', visualization)
    
    print(f"Successfully extracted {len(squares)} squares!")
else:
    print("No chess board detected in the image")
```

## Core Components

### ChessBoardDetector

The main class for detecting chess boards in images:

```python
from img2chess import ChessBoardDetector

detector = ChessBoardDetector(
    min_board_area=10000,    # Minimum board area in pixels
    max_board_area=500000    # Maximum board area in pixels
)

# Detect board in image
board_image = detector.detect_board(image)
```

**Detection Methods:**
- **Corner Detection**: Uses `cv2.findChessboardCorners()` to detect internal grid corners
- **Contour Analysis**: Finds board boundaries using contour detection and polygon approximation
- **Edge-Based Detection**: Uses edge detection and line filtering to identify chess board grid lines
- **Geometric Validation**: Validates detected regions based on area, aspect ratio, and shape

### SquareExtractor

Extracts individual squares from a detected chess board:

```python
from img2chess import SquareExtractor

extractor = SquareExtractor(square_size=64)

# Extract all squares
squares = extractor.extract_squares(board_image)

# Extract specific square
a1_square = extractor.extract_square(board_image, 'a1')

# Create visualization grid
grid = extractor.get_square_grid(squares)
```

### Utility Functions

**Image Loading and Enhancement:**
```python
from img2chess.utils import load_image, enhance_image, resize_image

# Load image with error handling
image = load_image('chess_board.jpg')

# Enhance image for better detection
enhanced = enhance_image(image)

# Resize while maintaining aspect ratio
resized = resize_image(image, max_size=1024)
```

**Validation and Statistics:**
```python
from img2chess.utils import validate_chess_position, get_square_statistics

# Validate extracted squares
is_valid = validate_chess_position(squares)

# Get statistics about the squares
stats = get_square_statistics(squares)
print(f"Mean brightness: {stats['mean_brightness']:.2f}")
```

## Examples

### Basic Usage

See `examples/basic_usage.py` for a complete example:

```bash
cd examples
python basic_usage.py
```

### Batch Processing

Process multiple images at once:

```python
from img2chess.utils import setup_logging
import glob

setup_logging("INFO")

# Process all images in a directory
image_files = glob.glob("images/*.jpg")
for image_file in image_files:
    # Process each image...
```

See `examples/advanced_usage.py` for batch processing implementation.

## Chess Square Notation

The library uses standard algebraic notation for chess squares:

- **Files (columns)**: a-h (left to right)
- **Ranks (rows)**: 1-8 (bottom to top from White's perspective)
- **Examples**: `a1` (bottom-left), `h8` (top-right), `e4` (center)

## Image Requirements

For best results, input images should:

- ✅ Contain a clearly visible chess board
- ✅ Have good lighting and contrast
- ✅ Show the full board (all 64 squares visible)
- ✅ Be reasonably high resolution (at least 400x400 pixels)
- ✅ Have the board as the main subject

**Supported formats**: JPG, PNG, BMP, TIFF

## Algorithm Details

### Board Detection Pipeline

1. **Preprocessing**: Convert to grayscale, apply noise reduction
2. **Corner Detection**: Attempt to find chessboard pattern using OpenCV
3. **Contour Detection**: Fallback method using edge detection and polygon approximation
4. **Edge-Based Detection**: Alternative method using line filtering techniques
5. **Validation**: Check area, aspect ratio, and geometric properties
6. **Perspective Correction**: Apply homography transformation to get square board

### Square Extraction

1. **Grid Division**: Divide corrected board into 8x8 grid
2. **Square Extraction**: Extract each square region
3. **Normalization**: Resize all squares to consistent size
4. **Labeling**: Apply chess notation (a1-h8)

## Performance Tips

- **Image Size**: Resize large images (>2000px) for faster processing
- **Lighting**: Ensure even lighting across the board
- **Angle**: Minimize perspective distortion when possible
- **Background**: Use contrasting background for better edge detection

## Future Enhancements

This library is designed to work with chess piece recognition models. The extracted squares can be fed into:

- **CNN Models**: For piece classification (pawn, rook, knight, etc.)
- **Position Analysis**: For FEN string generation
- **Game State Recognition**: For move detection and validation

## Dependencies

- `opencv-python >= 4.5.0`: Computer vision operations
- `numpy >= 1.20.0`: Numerical computations
- `pillow >= 8.0.0`: Image processing
- `matplotlib >= 3.3.0`: Visualization utilities

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV community for computer vision tools
- Chess programming community for notation standards
- Contributors and users of this library

## Support

- 📖 **Documentation**: Check the examples and docstrings
- 🐛 **Bug Reports**: Open an issue on GitHub
- 💡 **Feature Requests**: Open an issue with the enhancement label
- 💬 **Discussions**: Use GitHub Discussions for questions

---

Built with ❤️ for the chess and computer vision communities.