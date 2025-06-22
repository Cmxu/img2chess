"""Tests for the square_extractor module."""

import unittest
import numpy as np
import cv2
from img2chess.square_extractor import SquareExtractor


class TestSquareExtractor(unittest.TestCase):
    
    def setUp(self):
        self.extractor = SquareExtractor(square_size=64)
        # Create a simple test board image (640x640)
        self.test_board = np.zeros((640, 640, 3), dtype=np.uint8)
        # Add checkerboard pattern
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    y_start = i * 80
                    y_end = (i + 1) * 80
                    x_start = j * 80
                    x_end = (j + 1) * 80
                    self.test_board[y_start:y_end, x_start:x_end] = [255, 255, 255]
                    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        extractor = SquareExtractor()
        self.assertEqual(extractor.square_size, 64)
        
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        extractor = SquareExtractor(square_size=128)
        self.assertEqual(extractor.square_size, 128)
        
    def test_extract_squares_count(self):
        """Test that extract_squares returns 64 squares."""
        squares = self.extractor.extract_squares(self.test_board)
        self.assertEqual(len(squares), 64)
        
    def test_extract_squares_names(self):
        """Test that square names are correct."""
        squares = self.extractor.extract_squares(self.test_board)
        
        # Check that all expected square names are present
        expected_squares = set()
        for file in 'abcdefgh':
            for rank in '12345678':
                expected_squares.add(file + rank)
                
        self.assertEqual(set(squares.keys()), expected_squares)
        
    def test_extract_squares_size(self):
        """Test that extracted squares have correct size."""
        squares = self.extractor.extract_squares(self.test_board)
        
        for square_img in squares.values():
            self.assertEqual(square_img.shape[:2], (64, 64))
            
    def test_extract_single_square(self):
        """Test extracting a single square."""
        square_img = self.extractor.extract_square(self.test_board, 'e4')
        self.assertEqual(square_img.shape[:2], (64, 64))
        
    def test_get_square_grid(self):
        """Test creating square grid."""
        squares = self.extractor.extract_squares(self.test_board)
        grid = self.extractor.get_square_grid(squares)
        
        expected_size = 8 * self.extractor.square_size
        self.assertEqual(grid.shape[:2], (expected_size, expected_size))
        
    def test_coords_to_chess_notation(self):
        """Test coordinate to chess notation conversion."""
        self.assertEqual(self.extractor._coords_to_chess_notation(0, 0), 'a8')
        self.assertEqual(self.extractor._coords_to_chess_notation(7, 7), 'h1')
        self.assertEqual(self.extractor._coords_to_chess_notation(3, 4), 'e5')
        
    def test_chess_notation_to_coords(self):
        """Test chess notation to coordinate conversion."""
        self.assertEqual(self.extractor._chess_notation_to_coords('a8'), (0, 0))
        self.assertEqual(self.extractor._chess_notation_to_coords('h1'), (7, 7))
        self.assertEqual(self.extractor._chess_notation_to_coords('e5'), (3, 4))
        
    def test_invalid_square_name(self):
        """Test handling of invalid square names."""
        with self.assertRaises(ValueError):
            self.extractor._chess_notation_to_coords('z9')
            
        with self.assertRaises(ValueError):
            self.extractor._chess_notation_to_coords('a')
            
    def test_add_grid_lines(self):
        """Test adding grid lines to board."""
        board_with_grid = self.extractor.add_grid_lines(self.test_board)
        self.assertEqual(board_with_grid.shape, self.test_board.shape)
        
    def test_add_square_labels(self):
        """Test adding square labels to board."""
        labeled_board = self.extractor.add_square_labels(self.test_board)
        self.assertEqual(labeled_board.shape, self.test_board.shape)
        
    def test_extract_squares_none_input(self):
        """Test handling of None input."""
        with self.assertRaises(ValueError):
            self.extractor.extract_squares(None)


if __name__ == '__main__':
    unittest.main()