"""
Square extraction module for splitting chess boards into individual squares.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class SquareExtractor:
    """
    Extracts individual squares from a chess board image.
    
    Takes a perspective-corrected chess board image and splits it into
    64 individual square images for piece recognition.
    """
    
    def __init__(self, square_size: int = 64):
        """
        Initialize the square extractor.
        
        Args:
            square_size: Size of output square images in pixels
        """
        self.square_size = square_size
        
    def extract_squares(self, board_image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all 64 squares from a chess board image.
        
        Args:
            board_image: Perspective-corrected chess board image
            
        Returns:
            Dictionary mapping square names (e.g., 'a1', 'h8') to square images
        """
        if board_image is None:
            raise ValueError("Board image cannot be None")
            
        # Ensure board is square
        height, width = board_image.shape[:2]
        if height != width:
            logger.warning(f"Board image is not square ({width}x{height}), resizing to square")
            size = min(width, height)
            board_image = board_image[:size, :size]
            
        # Calculate square dimensions
        board_size = board_image.shape[0]
        square_width = board_size // 8
        square_height = board_size // 8
        
        squares = {}
        
        # Extract each square
        for row in range(8):
            for col in range(8):
                # Calculate square boundaries
                y_start = row * square_height
                y_end = (row + 1) * square_height
                x_start = col * square_width
                x_end = (col + 1) * square_width
                
                # Extract square region
                square_region = board_image[y_start:y_end, x_start:x_end]
                
                # Resize to standard size
                square_resized = cv2.resize(square_region, (self.square_size, self.square_size))
                
                # Convert coordinates to chess notation
                square_name = self._coords_to_chess_notation(row, col)
                squares[square_name] = square_resized
                
        logger.info(f"Extracted {len(squares)} squares from chess board")
        return squares
    
    def extract_square(self, board_image: np.ndarray, square_name: str) -> np.ndarray:
        """
        Extract a specific square from the chess board.
        
        Args:
            board_image: Perspective-corrected chess board image
            square_name: Chess notation for the square (e.g., 'a1', 'h8')
            
        Returns:
            Square image
        """
        row, col = self._chess_notation_to_coords(square_name)
        
        # Calculate square dimensions
        board_size = board_image.shape[0]
        square_width = board_size // 8
        square_height = board_size // 8
        
        # Calculate square boundaries
        y_start = row * square_height
        y_end = (row + 1) * square_height
        x_start = col * square_width
        x_end = (col + 1) * square_width
        
        # Extract and resize square
        square_region = board_image[y_start:y_end, x_start:x_end]
        square_resized = cv2.resize(square_region, (self.square_size, self.square_size))
        
        return square_resized
    
    def get_square_grid(self, squares: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Arrange squares back into a grid for visualization.
        
        Args:
            squares: Dictionary of square images
            
        Returns:
            Grid image showing all squares arranged as a chess board
        """
        if len(squares) != 64:
            raise ValueError(f"Expected 64 squares, got {len(squares)}")
            
        # Create grid
        grid_size = 8 * self.square_size
        grid = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
        
        for square_name, square_img in squares.items():
            row, col = self._chess_notation_to_coords(square_name)
            
            y_start = row * self.square_size
            y_end = (row + 1) * self.square_size
            x_start = col * self.square_size
            x_end = (col + 1) * self.square_size
            
            # Ensure square image has 3 channels
            if len(square_img.shape) == 2:
                square_img = cv2.cvtColor(square_img, cv2.COLOR_GRAY2BGR)
            elif square_img.shape[2] == 4:
                square_img = cv2.cvtColor(square_img, cv2.COLOR_BGRA2BGR)
                
            grid[y_start:y_end, x_start:x_end] = square_img
            
        return grid
    
    def _coords_to_chess_notation(self, row: int, col: int) -> str:
        """
        Convert array coordinates to chess notation.
        
        Args:
            row: Row index (0-7, top to bottom)
            col: Column index (0-7, left to right)
            
        Returns:
            Chess square notation (e.g., 'a1', 'h8')
        """
        # Chess notation: columns are a-h, rows are 1-8 (from bottom to top)
        file_letter = chr(ord('a') + col)
        rank_number = str(8 - row)  # Flip row (array index 0 is rank 8)
        return file_letter + rank_number
    
    def _chess_notation_to_coords(self, square_name: str) -> Tuple[int, int]:
        """
        Convert chess notation to array coordinates.
        
        Args:
            square_name: Chess square notation (e.g., 'a1', 'h8')
            
        Returns:
            Tuple of (row, col) coordinates
        """
        if len(square_name) != 2:
            raise ValueError(f"Invalid square name: {square_name}")
            
        file_letter = square_name[0].lower()
        rank_number = square_name[1]
        
        if file_letter < 'a' or file_letter > 'h':
            raise ValueError(f"Invalid file letter: {file_letter}")
        if rank_number < '1' or rank_number > '8':
            raise ValueError(f"Invalid rank number: {rank_number}")
            
        col = ord(file_letter) - ord('a')
        row = 8 - int(rank_number)  # Flip row (rank 8 is array index 0)
        
        return row, col
    
    def add_grid_lines(self, board_image: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Add grid lines to visualize square boundaries.
        
        Args:
            board_image: Chess board image
            color: Grid line color (BGR)
            thickness: Line thickness
            
        Returns:
            Board image with grid lines
        """
        board_with_grid = board_image.copy()
        board_size = board_image.shape[0]
        square_size = board_size // 8
        
        # Draw vertical lines
        for i in range(1, 8):
            x = i * square_size
            cv2.line(board_with_grid, (x, 0), (x, board_size), color, thickness)
            
        # Draw horizontal lines
        for i in range(1, 8):
            y = i * square_size
            cv2.line(board_with_grid, (0, y), (board_size, y), color, thickness)
            
        return board_with_grid
    
    def add_square_labels(self, board_image: np.ndarray, font_scale: float = 0.5, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """
        Add chess notation labels to each square.
        
        Args:
            board_image: Chess board image
            font_scale: Text font scale
            color: Text color (BGR)
            
        Returns:
            Board image with square labels
        """
        labeled_board = board_image.copy()
        board_size = board_image.shape[0]
        square_size = board_size // 8
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        
        for row in range(8):
            for col in range(8):
                square_name = self._coords_to_chess_notation(row, col)
                
                # Calculate text position (center of square)
                x = col * square_size + square_size // 2 - 10
                y = row * square_size + square_size // 2 + 5
                
                cv2.putText(labeled_board, square_name, (x, y), font, font_scale, color, thickness)
                
        return labeled_board