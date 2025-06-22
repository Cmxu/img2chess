"""Tests for the board_detector module."""

import unittest
import numpy as np
import cv2
from img2chess.board_detector import ChessBoardDetector


class TestChessBoardDetector(unittest.TestCase):
    
    def setUp(self):
        self.detector = ChessBoardDetector()
        
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        detector = ChessBoardDetector()
        self.assertEqual(detector.min_board_area, 10000)
        self.assertEqual(detector.max_board_area, 500000)
        
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        detector = ChessBoardDetector(min_board_area=5000, max_board_area=100000)
        self.assertEqual(detector.min_board_area, 5000)
        self.assertEqual(detector.max_board_area, 100000)
        
    def test_validate_board_corners_valid_square(self):
        """Test corner validation with valid square corners."""
        corners = np.array([
            [0, 0],
            [100, 0],
            [100, 100],
            [0, 100]
        ], dtype=np.float32)
        
        self.assertTrue(self.detector._validate_board_corners(corners))
        
    def test_validate_board_corners_invalid_aspect_ratio(self):
        """Test corner validation with invalid aspect ratio."""
        corners = np.array([
            [0, 0],
            [200, 0],
            [200, 50],
            [0, 50]
        ], dtype=np.float32)
        
        self.assertFalse(self.detector._validate_board_corners(corners))
        
    def test_validate_board_corners_too_small(self):
        """Test corner validation with too small area."""
        corners = np.array([
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10]
        ], dtype=np.float32)
        
        self.assertFalse(self.detector._validate_board_corners(corners))
        
    def test_coords_to_chess_notation(self):
        """Test coordinate to chess notation conversion."""
        from img2chess.square_extractor import SquareExtractor
        extractor = SquareExtractor()
        
        self.assertEqual(extractor._coords_to_chess_notation(0, 0), 'a8')
        self.assertEqual(extractor._coords_to_chess_notation(7, 7), 'h1')
        self.assertEqual(extractor._coords_to_chess_notation(3, 4), 'e5')
        
    def test_chess_notation_to_coords(self):
        """Test chess notation to coordinate conversion."""
        from img2chess.square_extractor import SquareExtractor
        extractor = SquareExtractor()
        
        self.assertEqual(extractor._chess_notation_to_coords('a8'), (0, 0))
        self.assertEqual(extractor._chess_notation_to_coords('h1'), (7, 7))
        self.assertEqual(extractor._chess_notation_to_coords('e5'), (3, 4))
        
    def test_order_corners(self):
        """Test corner ordering functionality."""
        # Create unordered corners
        corners = np.array([
            [100, 100],  # bottom-right
            [0, 0],      # top-left
            [100, 0],    # top-right
            [0, 100]     # bottom-left
        ], dtype=np.float32)
        
        ordered = self.detector._order_corners(corners)
        
        # Should be ordered: top-left, top-right, bottom-right, bottom-left
        expected = np.array([
            [0, 0],      # top-left
            [100, 0],    # top-right
            [100, 100],  # bottom-right
            [0, 100]     # bottom-left
        ], dtype=np.float32)
        
        np.testing.assert_array_almost_equal(ordered, expected)


if __name__ == '__main__':
    unittest.main()