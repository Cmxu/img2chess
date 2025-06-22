"""
Complete chess position analyzer that combines board detection, square extraction, and piece classification.

This module provides a complete pipeline to:
1. Detect chess boards in images
2. Extract individual squares
3. Classify pieces in each square
4. Output the position in standard notation
"""

import os
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor
from typing import Dict, List, Tuple, Optional
import logging

from img2chess import ChessBoardDetector, SquareExtractor
from chess_piece_classifier import ChessPieceClassifier

logger = logging.getLogger(__name__)


class ChessPositionAnalyzer:
    """
    Complete chess position analyzer combining board detection, square extraction, and piece classification.
    """
    
    def __init__(self, model_path: str = "chess_piece_classifier.pth", device: str = "auto"):
        """
        Initialize the chess position analyzer.
        
        Args:
            model_path: Path to trained chess piece classifier model
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        """
        self.device = self._get_device(device)
        self.model, self.class_names = self._load_model(model_path)
        self.board_detector = ChessBoardDetector()
        self.square_extractor = SquareExtractor(square_size=224)  # DINOv2 uses 224x224
        self.feature_extractor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-base", use_fast=True
        )
        
        # Mapping from classifier classes to chess notation
        self.piece_mapping = {
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
        
        logger.info(f"Chess Position Analyzer initialized on device: {self.device}")
        logger.info(f"Available piece classes: {self.class_names}")
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the best available device."""
        if device == "auto":
            if torch.backends.mps.is_available():
                return torch.device('mps')
            elif torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)
    
    def _load_model(self, model_path: str) -> Tuple[ChessPieceClassifier, List[str]]:
        """Load the trained chess piece classifier."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create and load model
        model = ChessPieceClassifier(num_classes=13, freeze_backbone=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        class_names = checkpoint['class_names']
        
        logger.info(f"Model loaded from {model_path}")
        return model, class_names
    
    def analyze_position(self, image_path: str, confidence_threshold: float = 0.5, 
                        save_debug: bool = False) -> Dict:
        """
        Analyze a chess position from an image.
        
        Args:
            image_path: Path to the chess board image
            confidence_threshold: Minimum confidence for piece predictions
            save_debug: Whether to save debug visualizations
            
        Returns:
            Dictionary containing position analysis results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        logger.info(f"Analyzing chess position in: {image_path}")
        
        # Step 1: Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Step 2: Detect chess board
        logger.info("Detecting chess board...")
        board_image = self.board_detector.detect_board(image)
        if board_image is None:
            raise ValueError("No chess board detected in the image")
        
        if save_debug:
            debug_dir = "debug_output"
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(f"{debug_dir}/detected_board.jpg", board_image)
            logger.info("Saved detected board to debug_output/detected_board.jpg")
        
        # Step 3: Extract squares
        logger.info("Extracting squares...")
        squares = self.square_extractor.extract_squares(board_image)
        
        if save_debug:
            # Save grid visualization
            grid_with_labels = self.square_extractor.add_grid_lines(board_image)
            grid_with_labels = self.square_extractor.add_square_labels(grid_with_labels)
            cv2.imwrite(f"{debug_dir}/board_with_grid.jpg", grid_with_labels)
            
            # Save individual squares
            squares_dir = f"{debug_dir}/squares"
            os.makedirs(squares_dir, exist_ok=True)
            for square_name, square_img in squares.items():
                cv2.imwrite(f"{squares_dir}/{square_name}.png", square_img)
            logger.info(f"Saved debug visualizations to {debug_dir}/")
        
        # Step 4: Classify pieces in each square
        logger.info("Classifying pieces in each square...")
        piece_predictions = self._classify_squares(squares, confidence_threshold)
        
        # Step 5: Generate position string
        position_grid = self._create_position_grid(piece_predictions)
        position_string = self._format_position_string(position_grid)
        
        # Step 6: Calculate statistics
        stats = self._calculate_statistics(piece_predictions)
        
        result = {
            'position_grid': position_grid,
            'position_string': position_string,
            'piece_predictions': piece_predictions,
            'statistics': stats,
            'board_detected': True,
            'total_squares': len(squares)
        }
        
        logger.info("Chess position analysis completed successfully")
        return result
    
    def _classify_squares(self, squares: Dict[str, np.ndarray], 
                         confidence_threshold: float) -> Dict[str, Dict]:
        """Classify pieces in each square."""
        predictions = {}
        
        for square_name, square_img in squares.items():
            # Convert BGR (OpenCV) to RGB (PIL)
            square_rgb = cv2.cvtColor(square_img, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(square_rgb)
            
            # Preprocess for model
            inputs = self.feature_extractor(images=image_pil, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(pixel_values)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            predicted_label = self.class_names[predicted_class]
            
            # Apply confidence threshold for non-empty squares
            final_prediction = predicted_label
            if predicted_label != 'empty' and confidence < confidence_threshold:
                final_prediction = 'empty'  # Default to empty if low confidence
                logger.debug(f"Square {square_name}: Low confidence {confidence:.3f} for {predicted_label}, defaulting to empty")
            
            predictions[square_name] = {
                'predicted_class': final_prediction,
                'confidence': confidence,
                'all_probabilities': probabilities[0].cpu().numpy().tolist(),
                'piece_symbol': self.piece_mapping.get(final_prediction, '?')
            }
        
        return predictions
    
    def _create_position_grid(self, predictions: Dict[str, Dict]) -> List[List[str]]:
        """Create an 8x8 grid representation of the position."""
        grid = [['·' for _ in range(8)] for _ in range(8)]
        
        for square_name, prediction in predictions.items():
            # Convert square name to coordinates
            file_letter = square_name[0]
            rank_number = square_name[1]
            
            col = ord(file_letter) - ord('a')  # a=0, b=1, ..., h=7
            row = 8 - int(rank_number)  # 8=0, 7=1, ..., 1=7 (flip for display)
            
            grid[row][col] = prediction['piece_symbol']
        
        return grid
    
    def _format_position_string(self, grid: List[List[str]]) -> str:
        """Format the position grid as a readable string."""
        lines = []
        
        # Add rank numbers and board
        for i, row in enumerate(grid):
            rank = 8 - i  # Display rank 8 at top
            line = f"{rank} | {' '.join(row)}"
            lines.append(line)
        
        # Add file letters
        lines.append("  +------------------")
        lines.append("    a b c d e f g h")
        
        return '\n'.join(lines)
    
    def _calculate_statistics(self, predictions: Dict[str, Dict]) -> Dict:
        """Calculate statistics about the position analysis."""
        piece_counts = {}
        confidence_values = []
        low_confidence_squares = []
        
        for square_name, prediction in predictions.items():
            piece = prediction['predicted_class']
            confidence = prediction['confidence']
            
            # Count pieces
            if piece in piece_counts:
                piece_counts[piece] += 1
            else:
                piece_counts[piece] = 1
            
            # Track confidence
            confidence_values.append(confidence)
            if confidence < 0.7:  # Flag low confidence predictions
                low_confidence_squares.append({
                    'square': square_name,
                    'piece': piece,
                    'confidence': confidence
                })
        
        return {
            'piece_counts': piece_counts,
            'average_confidence': np.mean(confidence_values),
            'min_confidence': np.min(confidence_values),
            'max_confidence': np.max(confidence_values),
            'low_confidence_squares': low_confidence_squares,
            'total_pieces': sum(count for piece, count in piece_counts.items() if piece != 'empty')
        }
    
    def print_analysis(self, result: Dict) -> None:
        """Print a formatted analysis of the chess position."""
        print("="*50)
        print("CHESS POSITION ANALYSIS")
        print("="*50)
        
        print("\nDetected Position:")
        print(result['position_string'])
        
        print(f"\nStatistics:")
        stats = result['statistics']
        print(f"  Total pieces detected: {stats['total_pieces']}")
        print(f"  Average confidence: {stats['average_confidence']:.3f}")
        print(f"  Confidence range: {stats['min_confidence']:.3f} - {stats['max_confidence']:.3f}")
        
        print(f"\nPiece counts:")
        for piece, count in sorted(stats['piece_counts'].items()):
            if piece != 'empty':
                symbol = self.piece_mapping.get(piece, piece)
                piece_name = self._get_piece_name(piece)
                print(f"  {symbol} {piece_name}: {count}")
        
        if stats['low_confidence_squares']:
            print(f"\nLow confidence predictions (< 0.7):")
            for square_info in stats['low_confidence_squares']:
                piece_name = self._get_piece_name(square_info['piece'])
                print(f"  {square_info['square']}: {piece_name} ({square_info['confidence']:.3f})")
        
        print("\n" + "="*50)
    
    def _get_piece_name(self, piece_code: str) -> str:
        """Convert piece code to readable name."""
        piece_names = {
            'empty': 'Empty',
            'wp': 'White Pawn',
            'wr': 'White Rook', 
            'wn': 'White Knight',
            'wb': 'White Bishop',
            'wq': 'White Queen',
            'wk': 'White King',
            'bp': 'Black Pawn',
            'br': 'Black Rook',
            'bn': 'Black Knight', 
            'bb': 'Black Bishop',
            'bq': 'Black Queen',
            'bk': 'Black King'
        }
        return piece_names.get(piece_code, piece_code)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Chess Position Analyzer')
    parser.add_argument('image', help='Path to chess board image')
    parser.add_argument('--model', default='chess_piece_classifier.pth', help='Path to trained model')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], 
                       help='Device to use')
    parser.add_argument('--debug', action='store_true', help='Save debug visualizations')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    if not args.quiet:
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING)
    
    try:
        # Initialize analyzer
        analyzer = ChessPositionAnalyzer(model_path=args.model, device=args.device)
        
        # Analyze position
        result = analyzer.analyze_position(
            image_path=args.image,
            confidence_threshold=args.confidence,
            save_debug=args.debug
        )
        
        # Print results
        analyzer.print_analysis(result)
        
        # Save results to file
        output_file = f"{os.path.splitext(args.image)[0]}_analysis.txt"
        with open(output_file, 'w') as f:
            f.write("CHESS POSITION ANALYSIS\n")
            f.write("="*50 + "\n\n")
            f.write("Position:\n")
            f.write(result['position_string'] + "\n\n")
            
            # Write detailed piece predictions
            f.write("Detailed Predictions:\n")
            for square in sorted(result['piece_predictions'].keys()):
                pred = result['piece_predictions'][square]
                f.write(f"{square}: {pred['predicted_class']} "
                       f"(confidence: {pred['confidence']:.3f})\n")
        
        print(f"\nDetailed results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())