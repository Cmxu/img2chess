"""
img2chess - A Python library for extracting chess boards from images

This library provides functionality to:
1. Detect chess boards in images using computer vision techniques
2. Extract the board region from the image
3. Split the board into individual squares for piece recognition
"""

from .board_detector import ChessBoardDetector
from .square_extractor import SquareExtractor
from .utils import visualize_board, save_squares

__version__ = "0.1.0"
__author__ = "img2chess"
__email__ = ""

__all__ = [
    "ChessBoardDetector",
    "SquareExtractor", 
    "visualize_board",
    "save_squares"
]