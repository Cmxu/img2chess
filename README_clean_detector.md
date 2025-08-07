# Clean Edge-Based Chess Board Detector

A simplified, configuration-driven chess board detector that focuses on clean code and ease of use.

## Features

- **YAML Configuration**: All hyperparameters are loaded from a YAML file
- **Simple Interface**: Returns either a board or None - no complex logging
- **Clean Code**: Focused implementation without debug outputs or extensive logging
- **Configurable**: Easy to tune parameters without code changes

## Usage

```python
import cv2
from img2chess.clean_edge_detector import CleanEdgeBasedDetector

# Initialize detector with config file
detector = CleanEdgeBasedDetector('detector_config.yaml')

# Load your image
image = cv2.imread('chess_board_image.jpg')

# Detect board
board = detector.detect_board(image)

if board is not None:
    # Board detected successfully
    cv2.imwrite('extracted_board.jpg', board)
    print("Board extracted!")
else:
    # No board found
    print("No chess board detected")
```

## Configuration

The detector uses a YAML configuration file (`detector_config.yaml`) that contains all hyperparameters organized into logical sections:

- **edge_detection**: Gaussian blur and Canny edge detection parameters
- **line_detection**: Hough line detection parameters  
- **line_filtering**: Angle tolerance for vertical/horizontal classification
- **distance_analysis**: Parameters for finding frequent line distances
- **line_selection**: Pattern matching and boundary filtering parameters
- **board_validation**: Minimum requirements and geometry validation
- **board_extraction**: Output board size
- **duplicate_removal**: Smart duplicate line removal

## Testing

Run the test script to verify the detector works:

```bash
cd img2chess
python test_clean_detector.py
```

## Configuration Parameters

### Key Parameters to Tune

1. **min_instances** (distance_analysis): Minimum occurrences for a distance to be considered "frequent"
2. **tolerance_pixels** (distance_analysis): Tolerance for grouping similar distances  
3. **min_square_size** (distance_analysis): Minimum reasonable chess square size
4. **threshold** (line_detection): Hough transform threshold - higher = fewer lines
5. **min_line_length** (line_detection): Minimum line length to detect
6. **angle_tolerance** (line_filtering): Tolerance for vertical/horizontal classification

### Example Tuning for Different Scenarios

**For high-quality, clear images:**
- Increase `threshold` to 250-300
- Decrease `tolerance_pixels` to 2.0
- Increase `min_instances` to 15

**For noisy or low-quality images:**
- Decrease `threshold` to 150-200  
- Increase `tolerance_pixels` to 3.0
- Decrease `min_instances` to 10
- Increase `angle_tolerance` to 1.5

**For smaller chess boards:**
- Decrease `min_square_size` to 20
- Decrease `min_line_length` to 100

## Architecture

The detector follows these main steps:

1. **Edge Detection**: Canny edge detection with adaptive thresholds
2. **Line Detection**: HoughLinesP to find straight lines
3. **Line Filtering**: Separate vertical and horizontal lines by angle
4. **Distance Analysis**: Compute all pairwise line distances
5. **Frequent Distances**: Find commonly occurring distances
6. **Pattern Validation**: Verify distances match chess square multiples
7. **Line Selection**: Choose lines that match chess board spacing
8. **Boundary Filtering**: Remove lines outside the board region
9. **Duplicate Removal**: Intelligently remove redundant lines
10. **Corner Calculation**: Find board corners from outermost lines
11. **Geometry Validation**: Verify corners form a proper square
12. **Board Extraction**: Perspective transform to extract board

## Advantages over Original Detector

- **Cleaner Code**: No logging clutter, focused on detection logic
- **Configuration-Driven**: Easy to tune without code changes
- **Simple Interface**: Just returns board or None
- **Maintainable**: Clear structure and well-documented parameters
- **Faster**: No debug outputs or extensive logging overhead