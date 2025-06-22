"""
Utility functions for the img2chess library.
"""

import cv2
import numpy as np
import os
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def visualize_board(board_image: np.ndarray, squares: Optional[Dict[str, np.ndarray]] = None, 
                   show_grid: bool = True, show_labels: bool = True) -> np.ndarray:
    """
    Create a visualization of the chess board with optional grid and labels.
    
    Args:
        board_image: Chess board image
        squares: Optional dictionary of extracted squares
        show_grid: Whether to show grid lines
        show_labels: Whether to show square labels
        
    Returns:
        Visualization image
    """
    from .square_extractor import SquareExtractor
    
    viz_image = board_image.copy()
    extractor = SquareExtractor()
    
    if show_grid:
        viz_image = extractor.add_grid_lines(viz_image)
        
    if show_labels:
        viz_image = extractor.add_square_labels(viz_image)
    
    return viz_image


def save_squares(squares: Dict[str, np.ndarray], output_dir: str, prefix: str = "square_") -> None:
    """
    Save extracted squares as individual image files.
    
    Args:
        squares: Dictionary mapping square names to images
        output_dir: Directory to save images
        prefix: Filename prefix for saved images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for square_name, square_img in squares.items():
        filename = f"{prefix}{square_name}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, square_img)
        
    logger.info(f"Saved {len(squares)} square images to {output_dir}")


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from file with error handling.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Loaded image or None if failed
    """
    if not os.path.exists(image_path):
        logger.error(f"Image file does not exist: {image_path}")
        return None
        
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return None
        
    return image


def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    if max(height, width) <= max_size:
        return image
        
    # Calculate scaling factor
    scale = max_size / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized


def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Apply image enhancement for better chess board detection.
    
    Args:
        image: Input image
        
    Returns:
        Enhanced image
    """
    # Convert to LAB color space for better illumination handling
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced


def validate_chess_position(squares: Dict[str, np.ndarray]) -> bool:
    """
    Perform basic validation on extracted chess squares.
    
    Args:
        squares: Dictionary of extracted squares
        
    Returns:
        True if squares pass basic validation
    """
    # Check we have all 64 squares
    if len(squares) != 64:
        logger.warning(f"Expected 64 squares, got {len(squares)}")
        return False
        
    # Check all squares have same dimensions
    sizes = [square.shape for square in squares.values()]
    if len(set(sizes)) > 1:
        logger.warning("Squares have different dimensions")
        return False
        
    # Check square names are valid
    expected_squares = set()
    for file in 'abcdefgh':
        for rank in '12345678':
            expected_squares.add(file + rank)
            
    if set(squares.keys()) != expected_squares:
        logger.warning("Invalid square names detected")
        return False
        
    return True


def create_comparison_grid(original_squares: Dict[str, np.ndarray], 
                          processed_squares: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Create a side-by-side comparison grid of original and processed squares.
    
    Args:
        original_squares: Original extracted squares
        processed_squares: Processed squares (e.g., after preprocessing)
        
    Returns:
        Comparison grid image
    """
    from .square_extractor import SquareExtractor
    
    extractor = SquareExtractor()
    
    # Create grids for both sets
    original_grid = extractor.get_square_grid(original_squares)
    processed_grid = extractor.get_square_grid(processed_squares)
    
    # Combine side by side
    comparison = np.hstack([original_grid, processed_grid])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Processed", (original_grid.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
    
    return comparison


def get_square_statistics(squares: Dict[str, np.ndarray]) -> Dict:
    """
    Calculate statistics about the extracted squares.
    
    Args:
        squares: Dictionary of extracted squares
        
    Returns:
        Dictionary containing statistics
    """
    if not squares:
        return {}
        
    # Calculate brightness statistics
    brightness_values = []
    for square in squares.values():
        gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY) if len(square.shape) == 3 else square
        brightness_values.append(np.mean(gray))
        
    stats = {
        'num_squares': len(squares),
        'square_size': squares[list(squares.keys())[0]].shape,
        'mean_brightness': np.mean(brightness_values),
        'std_brightness': np.std(brightness_values),
        'min_brightness': np.min(brightness_values),
        'max_brightness': np.max(brightness_values)
    }
    
    return stats


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging for the img2chess library.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )