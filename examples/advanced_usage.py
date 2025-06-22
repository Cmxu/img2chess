#!/usr/bin/env python3
"""
Advanced usage example for the img2chess library.

This demonstrates:
1. Image preprocessing techniques
2. Handling multiple board detection methods
3. Custom validation and error handling
4. Batch processing multiple images
"""

import cv2
import os
import glob
from img2chess import ChessBoardDetector, SquareExtractor
from img2chess.utils import load_image, enhance_image, resize_image, setup_logging
import logging

logger = logging.getLogger(__name__)


def process_single_image(image_path: str, output_base_dir: str) -> bool:
    """Process a single chess board image."""
    logger.info(f"Processing: {image_path}")
    
    # Load and preprocess image
    image = load_image(image_path)
    if image is None:
        return False
    
    # Resize if too large
    image = resize_image(image, max_size=1024)
    
    # Try enhanced version if initial detection fails
    detector = ChessBoardDetector(min_board_area=5000, max_board_area=800000)
    board_image = detector.detect_board(image)
    
    if board_image is None:
        logger.info("Initial detection failed, trying enhanced image")
        enhanced_image = enhance_image(image)
        board_image = detector.detect_board(enhanced_image)
        
    if board_image is None:
        logger.warning(f"No board detected in {image_path}")
        return False
    
    # Extract squares
    extractor = SquareExtractor(square_size=128)  # Larger squares for better quality
    squares = extractor.extract_squares(board_image)
    
    # Create output directory
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(output_base_dir, basename)
    
    # Save results
    from img2chess.utils import save_squares, visualize_board
    save_squares(squares, output_dir)
    
    # Save visualization
    viz = visualize_board(board_image, show_grid=True, show_labels=True)
    cv2.imwrite(os.path.join(output_dir, "visualization.jpg"), viz)
    
    logger.info(f"Successfully processed {image_path}")
    return True


def batch_process(input_dir: str, output_dir: str):
    """Process all images in a directory."""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(input_dir, '**', ext)
        image_files.extend(glob.glob(pattern, recursive=True))
    
    logger.info(f"Found {len(image_files)} images to process")
    
    success_count = 0
    for image_file in image_files:
        if process_single_image(image_file, output_dir):
            success_count += 1
    
    logger.info(f"Successfully processed {success_count}/{len(image_files)} images")


if __name__ == "__main__":
    setup_logging("INFO")
    
    # Example usage
    input_directory = "input_images"  # Directory containing chess board images
    output_directory = "processed_results"
    
    if os.path.exists(input_directory):
        batch_process(input_directory, output_directory)
    else:
        print(f"Input directory '{input_directory}' not found")
        print("Create the directory and add some chess board images to test batch processing")