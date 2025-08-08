#!/usr/bin/env python3
"""
Complete chess board analysis pipeline that combines:
1. Board detection
2. Square extraction 
3. Piece classification
4. FEN notation output
"""

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor
import logging
from typing import Dict, List, Optional, Tuple

from img2chess.board_detector import ChessBoardDetector
from img2chess.square_extractor import SquareExtractor
from chess_piece_classifier import ChessPieceClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChessBoardAnalyzer:
    """Complete chess board analysis pipeline."""
    
    def __init__(self, 
                 model_path: str = "chess_piece_classifier.pth",
                 device: str = 'auto'):
        """
        Initialize the chess board analyzer.
        
        Args:
            model_path: Path to trained classifier model
            device: Computing device ('auto', 'cpu', 'cuda', 'mps')
        """
        self.device = self._get_device(device)
        self.board_detector = ChessBoardDetector()
        self.square_extractor = SquareExtractor(square_size=224)  # DINOv2 uses 224x224
        
        # Load classifier model
        self.model, self.class_names = self._load_model(model_path)
        
        # Initialize feature extractor
        self.feature_extractor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-base", use_fast=True
        )
        
        # Mapping from class names to chess notation
        self.piece_notation = {
            'empty': '·',
            'wp': 'p',  # white pawn -> lowercase
            'wr': 'r',  # white rook -> lowercase  
            'wn': 'n',  # white knight -> lowercase
            'wb': 'b',  # white bishop -> lowercase
            'wq': 'q',  # white queen -> lowercase
            'wk': 'k',  # white king -> lowercase
            'bp': 'P',  # black pawn -> uppercase
            'br': 'R',  # black rook -> uppercase
            'bn': 'N',  # black knight -> uppercase
            'bb': 'B',  # black bishop -> uppercase
            'bq': 'Q',  # black queen -> uppercase
            'bk': 'K'   # black king -> uppercase
        }
        
        logger.info(f"Chess board analyzer initialized on device: {self.device}")
        logger.info(f"Loaded {len(self.class_names)} piece classes: {self.class_names}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best available device."""
        if device == 'auto':
            mps_ok = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            if mps_ok:
                return 'mps'
            elif torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device
    
    def _load_model(self, model_path: str) -> Tuple[ChessPieceClassifier, List[str]]:
        """Load the trained chess piece classifier."""
        logger.info(f"Loading model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Auto-detect model architecture from checkpoint
        state_dict = checkpoint['model_state_dict']
        
        # Check the first classifier layer to determine architecture
        first_layer_weight = state_dict['classifier.0.weight']
        hidden_dim1 = first_layer_weight.shape[0]  # Output dimension of first layer
        
        # Determine the full architecture
        if hidden_dim1 == 512:
            # Original architecture: 768 -> 512 -> 256 -> 13
            from chess_piece_classifier_original import ChessPieceClassifierOriginal
            model = ChessPieceClassifierOriginal(num_classes=13, freeze_backbone=True)
        else:
            # New architecture: 768 -> 128 -> 64 -> 13
            model = ChessPieceClassifier(num_classes=13, freeze_backbone=True)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        class_names = checkpoint['class_names']
        
        return model, class_names
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Complete analysis of chess board image.
        
        Args:
            image_path: Path to chess board image
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Analyzing chess board image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Step 1: Detect chess board
        logger.info("Step 1: Detecting chess board...")
        board_image = self.board_detector.detect_board(image)
        
        if board_image is None:
            return {
                'success': False,
                'error': 'No chess board detected in image',
                'board_grid': None,
                'piece_positions': {}
            }
        
        logger.info("✅ Chess board detected successfully")
        
        # Step 2: Extract squares
        logger.info("Step 2: Extracting squares...")
        squares = self.square_extractor.extract_squares(board_image)
        logger.info(f"✅ Extracted {len(squares)} squares")
        
        # Step 3: Classify pieces in each square
        logger.info("Step 3: Classifying pieces...")
        piece_predictions = self._classify_squares(squares)
        
        # Step 4: Generate board representation
        board_grid = self._generate_board_grid(piece_predictions)
        
        return {
            'success': True,
            'board_image': board_image,
            'squares': squares,
            'piece_predictions': piece_predictions,
            'board_grid': board_grid,
            'piece_positions': self._get_piece_positions(piece_predictions)
        }
    
    def _classify_squares(self, squares: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """Classify pieces in all squares."""
        predictions = {}
        
        # Prepare batch of images
        square_names = []
        images = []
        
        for square_name, square_image in squares.items():
            square_names.append(square_name)
            
            # Convert OpenCV image to PIL
            if len(square_image.shape) == 3:
                # BGR to RGB
                pil_image = Image.fromarray(cv2.cvtColor(square_image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(square_image)
            
            images.append(pil_image)
        
        # Process images in batches for efficiency
        batch_size = 32
        for i in range(0, len(images), batch_size):
            batch_names = square_names[i:i+batch_size]
            batch_images = images[i:i+batch_size]
            
            # Preprocess batch
            inputs = self.feature_extractor(images=batch_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(pixel_values)
                probabilities = F.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(outputs, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
            
            # Store results
            for j, square_name in enumerate(batch_names):
                predicted_class = predicted_classes[j].item()
                confidence = confidences[j].item()
                predicted_label = self.class_names[predicted_class]
                
                predictions[square_name] = {
                    'class': predicted_label,
                    'confidence': confidence,
                    'notation': self.piece_notation[predicted_label]
                }
        
        logger.info("✅ Piece classification completed")
        return predictions
    
    def _generate_board_grid(self, predictions: Dict[str, Dict]) -> List[List[str]]:
        """Generate 8x8 grid representation of the board."""
        grid = []
        
        # Generate grid from rank 8 to rank 1 (top to bottom display)
        for rank in range(8, 0, -1):
            row = []
            for file_char in 'abcdefgh':
                square_name = f"{file_char}{rank}"
                if square_name in predictions:
                    row.append(predictions[square_name]['notation'])
                else:
                    row.append('?')  # Unknown square
            grid.append(row)
        
        return grid
    
    def _get_piece_positions(self, predictions: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Get positions of each piece type."""
        positions = {}
        
        for square_name, pred in predictions.items():
            piece_type = pred['class']
            if piece_type != 'empty':
                if piece_type not in positions:
                    positions[piece_type] = []
                positions[piece_type].append(square_name)
        
        return positions
    
    def print_board(self, board_grid: List[List[str]], show_coordinates: bool = True):
        """Print the board in a nice format."""
        print("\n" + "="*50)
        print("CHESS BOARD ANALYSIS")
        print("="*50)
        
        if show_coordinates:
            print("   a b c d e f g h")
            print("  +-+-+-+-+-+-+-+-+")
        
        for i, row in enumerate(board_grid):
            rank = 8 - i
            if show_coordinates:
                print(f"{rank} |{' '.join(row)}| {rank}")
            else:
                print(' '.join(row))
        
        if show_coordinates:
            print("  +-+-+-+-+-+-+-+-+")
            print("   a b c d e f g h")
        
        print("\nLegend:")
        print("· = empty square")
        print("Lowercase (p,r,n,b,q,k) = white pieces")
        print("Uppercase (P,R,N,B,Q,K) = black pieces")
    
    def save_debug_images(self, result: Dict, output_dir: str = "."):
        """Save debug images showing detection results."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if result['success']:
            # Save detected board
            board_image = result['board_image']
            cv2.imwrite(f"{output_dir}/detected_board.jpg", board_image)
            
            # Save board with grid lines
            board_with_grid = self.square_extractor.add_grid_lines(board_image)
            cv2.imwrite(f"{output_dir}/board_with_grid.jpg", board_with_grid)
            
            # Save board with square labels
            board_labeled = self.square_extractor.add_square_labels(board_with_grid)
            cv2.imwrite(f"{output_dir}/board_labeled.jpg", board_labeled)
            
            logger.info(f"Debug images saved to {output_dir}/")


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze chess board from image')
    parser.add_argument('image', help='Path to chess board image')
    parser.add_argument('--model', default='chess_piece_classifier.pth', 
                       help='Path to trained model')
    parser.add_argument('--device', default='auto', 
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use')
    parser.add_argument('--save-debug', action='store_true',
                       help='Save debug images')
    parser.add_argument('--debug-dir', default='debug_output',
                       help='Directory to save debug images')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = ChessBoardAnalyzer(args.model, args.device)
        
        # Analyze image
        result = analyzer.analyze_image(args.image)
        
        if result['success']:
            # Print board
            analyzer.print_board(result['board_grid'])
            
            # Print piece positions
            print(f"\nPiece positions:")
            for piece_type, positions in result['piece_positions'].items():
                print(f"{piece_type}: {', '.join(positions)}")
            
            # Save debug images if requested
            if args.save_debug:
                analyzer.save_debug_images(result, args.debug_dir)
        else:
            print(f"❌ Analysis failed: {result['error']}")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Analysis failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())