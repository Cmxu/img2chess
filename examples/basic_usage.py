#!/usr/bin/env python3
"""
Basic usage example for the img2chess library.

This example demonstrates how to:
1. Load an image containing a chess board
2. Detect and extract the chess board
3. Split the board into individual squares
4. Visualize and save the results
"""

import cv2
import os
from img2chess import ChessBoardDetector, SquareExtractor, visualize_board, save_squares
from img2chess.utils import load_image, setup_logging


def main():
    # Setup logging
    setup_logging("INFO")
    
    # Define paths
    input_image_path = "sample_chess_board.jpg"  # Replace with your image path
    output_dir = "extracted_squares"
    
    # Check if input image exists
    if not os.path.exists(input_image_path):
        print(f"Sample image not found at {input_image_path}")
        print("Please provide a chess board image to test with.")
        return
    
    # Load the image
    print(f"Loading image: {input_image_path}")
    image = load_image(input_image_path)
    if image is None:
        print("Failed to load image")
        return
    
    print(f"Image loaded successfully. Size: {image.shape[1]}x{image.shape[0]}")
    
    # Initialize the chess board detector
    detector = ChessBoardDetector(min_board_area=10000, max_board_area=500000)
    
    # Detect and extract the chess board
    print("Detecting chess board...")
    board_image = detector.detect_board(image)
    
    if board_image is None:
        print("No chess board detected in the image")
        print("Try adjusting the detection parameters or using a clearer image")
        return
    
    print(f"Chess board detected and extracted. Size: {board_image.shape[1]}x{board_image.shape[0]}")
    
    # Save the extracted board
    cv2.imwrite("extracted_board.jpg", board_image)
    print("Extracted board saved as 'extracted_board.jpg'")
    
    # Initialize the square extractor
    extractor = SquareExtractor(square_size=64)
    
    # Extract all squares
    print("Extracting individual squares...")
    squares = extractor.extract_squares(board_image)
    
    print(f"Successfully extracted {len(squares)} squares")
    
    # Save individual squares
    save_squares(squares, output_dir)
    print(f"Individual squares saved to directory: {output_dir}")
    
    # Create visualization
    print("Creating visualization...")
    visualization = visualize_board(board_image, squares, show_grid=True, show_labels=True)
    cv2.imwrite("board_visualization.jpg", visualization)
    print("Board visualization saved as 'board_visualization.jpg'")
    
    # Create square grid
    square_grid = extractor.get_square_grid(squares)
    cv2.imwrite("square_grid.jpg", square_grid)
    print("Square grid saved as 'square_grid.jpg'")
    
    # Print some statistics
    from img2chess.utils import get_square_statistics, validate_chess_position
    
    stats = get_square_statistics(squares)
    print("\nSquare Statistics:")
    print(f"  Number of squares: {stats['num_squares']}")
    print(f"  Square size: {stats['square_size']}")
    print(f"  Mean brightness: {stats['mean_brightness']:.2f}")
    print(f"  Brightness std: {stats['std_brightness']:.2f}")
    
    # Validate extraction
    is_valid = validate_chess_position(squares)
    print(f"\nValidation result: {'PASSED' if is_valid else 'FAILED'}")
    
    # Show a few example squares
    print("\nExample squares extracted:")
    example_squares = ['a1', 'a8', 'h1', 'h8', 'd4', 'e4']
    for square_name in example_squares:
        if square_name in squares:
            square_img = squares[square_name]
            filename = f"example_{square_name}.jpg"
            cv2.imwrite(filename, square_img)
            print(f"  {square_name}: saved as {filename}")
    
    print("\nProcessing complete!")
    print("Check the generated files to see the results.")


if __name__ == "__main__":
    main()