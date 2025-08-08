"""
Complete chess position analyzer that combines board detection, square extraction, and piece classification.

This module provides a complete pipeline to:
1. Detect chess boards in images
2. Extract individual squares
3. Classify pieces in each square
4. Output the position in standard notation

Supports both single image and batch processing for efficiency.
"""

import os
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

from img2chess import ChessBoardDetector, SquareExtractor
from chess_piece_classifier import ChessPieceClassifier

logger = logging.getLogger(__name__)


class ChessPositionAnalyzer:
    """
    Complete chess position analyzer combining board detection, square extraction, and piece classification.
    
    Supports both single image and batch processing for optimal performance.
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
            mps_ok = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            if mps_ok:
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
        
        # Create debug directory if needed
        debug_dir = "debug_output"
        if save_debug:
            os.makedirs(debug_dir, exist_ok=True)
            # Clear previous debug files
            for file in os.listdir(debug_dir):
                file_path = os.path.join(debug_dir, file)
                if os.path.isfile(file_path) and file.endswith('.png'):
                    os.remove(file_path)
            logger.info(f"Debug directory prepared: {debug_dir}")
        
        # Step 1: Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        if save_debug:
            # Save original image
            cv2.imwrite(f"{debug_dir}/01_original_image.png", image)
            logger.info("Saved original image to debug_output/01_original_image.png")
        
        # Step 2: Detect chess board
        logger.info("Detecting chess board...")
        board_image = self.board_detector.detect_board(image)
        if board_image is None:
            raise ValueError("No chess board detected in the image")
        
        if save_debug:
            cv2.imwrite(f"{debug_dir}/02_detected_board.png", board_image)
            logger.info("Saved detected board to debug_output/02_detected_board.png")
        
        # Step 3: Extract squares
        logger.info("Extracting squares...")
        squares = self.square_extractor.extract_squares(board_image)
        
        if save_debug:
            # Save grid visualization
            grid_with_labels = self.square_extractor.add_grid_lines(board_image)
            grid_with_labels = self.square_extractor.add_square_labels(grid_with_labels)
            cv2.imwrite(f"{debug_dir}/03_board_with_grid.png", grid_with_labels)
            logger.info("Saved board with grid to debug_output/03_board_with_grid.png")
            
            # Save individual squares
            squares_dir = f"{debug_dir}/squares"
            os.makedirs(squares_dir, exist_ok=True)
            for square_name, square_img in squares.items():
                cv2.imwrite(f"{squares_dir}/{square_name}.png", square_img)
            logger.info(f"Saved {len(squares)} individual squares to {squares_dir}/")
        
        # Step 4: Classify pieces in each square (with batch processing)
        logger.info("Classifying pieces in each square...")
        piece_predictions = self._classify_squares_batch(squares, confidence_threshold)
        
        if save_debug:
            # Create visualization with predictions
            prediction_vis = self._create_prediction_visualization(board_image, piece_predictions)
            cv2.imwrite(f"{debug_dir}/04_predictions_visualization.png", prediction_vis)
            logger.info("Saved predictions visualization to debug_output/04_predictions_visualization.png")
            
            # Save detailed classification results
            self._save_classification_debug(piece_predictions, debug_dir)
        
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
            'total_squares': len(squares),
            'image_path': image_path
        }
        
        logger.info("Chess position analysis completed successfully")
        return result
    
    def analyze_batch(self, image_paths: Union[List[str], str], 
                     confidence_threshold: float = 0.5,
                     save_debug: bool = False,
                     debug_dir: str = "debug_output") -> List[Dict]:
        """
        Analyze multiple chess positions from a list of images or directory.
        
        Args:
            image_paths: List of image paths or directory path containing images
            confidence_threshold: Minimum confidence for piece predictions
            save_debug: Whether to save debug visualizations
            debug_dir: Directory to save debug outputs
            
        Returns:
            List of dictionaries containing position analysis results
        """
        # Handle input types
        if isinstance(image_paths, str):
            # If it's a directory, get all image files
            if os.path.isdir(image_paths):
                image_paths = self._get_image_files(image_paths)
            else:
                # Single file
                image_paths = [image_paths]
        
        if not image_paths:
            raise ValueError("No valid image files found")
        
        logger.info(f"Starting batch analysis of {len(image_paths)} images")
        
        results = []
        successful = 0
        failed = 0
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                
                # Create image-specific debug directory if needed
                if save_debug:
                    image_name = Path(image_path).stem
                    image_debug_dir = os.path.join(debug_dir, image_name)
                    os.makedirs(image_debug_dir, exist_ok=True)
                else:
                    image_debug_dir = None
                
                # Analyze single image
                result = self.analyze_position(
                    image_path, 
                    confidence_threshold=confidence_threshold,
                    save_debug=save_debug
                )
                
                results.append(result)
                successful += 1
                
                logger.info(f"✅ Successfully analyzed: {image_path}")
                
            except Exception as e:
                logger.error(f"❌ Failed to analyze {image_path}: {str(e)}")
                failed += 1
                
                # Add error result
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'board_detected': False,
                    'success': False
                })
        
        logger.info(f"Batch analysis completed: {successful} successful, {failed} failed")
        
        return results
    
    def _get_image_files(self, directory: str) -> List[str]:
        """Get all image files from a directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for file in os.listdir(directory):
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(directory, file))
        
        return sorted(image_files)
    
    def _classify_squares_batch(self, squares: Dict[str, np.ndarray], 
                               confidence_threshold: float) -> Dict[str, Dict]:
        """Classify pieces in each square using batch processing for efficiency."""
        predictions = {}
        
        # Prepare batch of images
        square_names = []
        images = []
        
        for square_name, square_img in squares.items():
            square_names.append(square_name)
            
            # Convert BGR (OpenCV) to RGB (PIL)
            square_rgb = cv2.cvtColor(square_img, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(square_rgb)
            images.append(image_pil)
        
        # Process images in batches for efficiency
        batch_size = 32  # Optimal batch size for most GPUs
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
                
                # Apply confidence threshold for non-empty squares
                final_prediction = predicted_label
                if predicted_label != 'empty' and confidence < confidence_threshold:
                    final_prediction = 'empty'  # Default to empty if low confidence
                    logger.debug(f"Square {square_name}: Low confidence {confidence:.3f} for {predicted_label}, defaulting to empty")
                
                predictions[square_name] = {
                    'predicted_class': final_prediction,
                    'confidence': confidence,
                    'all_probabilities': probabilities[j].cpu().numpy().tolist(),
                    'piece_symbol': self.piece_mapping.get(final_prediction, '?')
                }
        
        return predictions
    
    def _create_prediction_visualization(self, board_image: np.ndarray, 
                                       predictions: Dict[str, Dict]) -> np.ndarray:
        """Create a visualization of the board with piece predictions overlaid."""
        # Create a copy of the board image
        vis_image = board_image.copy()
        
        # Get image dimensions
        height, width = vis_image.shape[:2]
        square_size = min(height, width) // 8
        
        # Define colors for different piece types
        colors = {
            'empty': (128, 128, 128),  # Gray
            'wp': (255, 255, 255),     # White
            'wr': (255, 255, 255),
            'wn': (255, 255, 255),
            'wb': (255, 255, 255),
            'wq': (255, 255, 255),
            'wk': (255, 255, 255),
            'bp': (0, 0, 0),           # Black
            'br': (0, 0, 0),
            'bn': (0, 0, 0),
            'bb': (0, 0, 0),
            'bq': (0, 0, 0),
            'bk': (0, 0, 0)
        }
        
        # Draw predictions on the board
        for square_name, prediction in predictions.items():
            # Convert square name to coordinates
            file_letter = square_name[0]
            rank_number = square_name[1]
            
            col = ord(file_letter) - ord('a')  # a=0, b=1, ..., h=7
            row = 8 - int(rank_number)  # 8=0, 7=1, ..., 1=7
            
            # Calculate pixel coordinates
            x = col * square_size
            y = row * square_size
            
            # Get piece symbol and color
            piece = prediction['predicted_class']
            confidence = prediction['confidence']
            symbol = self.piece_mapping.get(piece, '?')
            color = colors.get(piece, (255, 0, 0))  # Red for unknown
            
            # Draw background rectangle
            cv2.rectangle(vis_image, (x, y), (x + square_size, y + square_size), 
                         color, 2)
            
            # Draw piece symbol
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            
            # Calculate text position (center of square)
            text_size = cv2.getTextSize(symbol, font, font_scale, thickness)[0]
            text_x = x + (square_size - text_size[0]) // 2
            text_y = y + (square_size + text_size[1]) // 2
            
            # Draw text with outline for better visibility
            cv2.putText(vis_image, symbol, (text_x, text_y), font, font_scale, 
                       (0, 0, 0), thickness + 1)  # Black outline
            cv2.putText(vis_image, symbol, (text_x, text_y), font, font_scale, 
                       color, thickness)  # Colored text
            
            # Draw confidence score
            conf_text = f"{confidence:.2f}"
            conf_font_scale = 0.5
            conf_thickness = 1
            conf_size = cv2.getTextSize(conf_text, font, conf_font_scale, conf_thickness)[0]
            conf_x = x + square_size - conf_size[0] - 5
            conf_y = y + conf_size[1] + 5
            
            cv2.putText(vis_image, conf_text, (conf_x, conf_y), font, conf_font_scale, 
                       (0, 255, 0), conf_thickness + 1)  # Green outline
            cv2.putText(vis_image, conf_text, (conf_x, conf_y), font, conf_font_scale, 
                       (0, 255, 0), conf_thickness)  # Green text
        
        return vis_image
    
    def _save_classification_debug(self, predictions: Dict[str, Dict], debug_dir: str):
        """Save detailed classification debug information."""
        # Create classification debug file
        debug_file = os.path.join(debug_dir, "classification_debug.txt")
        
        with open(debug_file, 'w') as f:
            f.write("CHESS PIECE CLASSIFICATION DEBUG\n")
            f.write("=" * 50 + "\n\n")
            
            # Sort squares by rank and file
            sorted_squares = sorted(predictions.keys(), 
                                  key=lambda x: (x[1], x[0]))  # Sort by rank, then file
            
            for square_name in sorted_squares:
                pred = predictions[square_name]
                f.write(f"Square {square_name}:\n")
                f.write(f"  Predicted: {pred['predicted_class']}\n")
                f.write(f"  Symbol: {pred['piece_symbol']}\n")
                f.write(f"  Confidence: {pred['confidence']:.4f}\n")
                
                # Show top 3 probabilities
                all_probs = pred['all_probabilities']
                top_indices = np.argsort(all_probs)[-3:][::-1]  # Top 3
                f.write(f"  Top 3 probabilities:\n")
                for idx in top_indices:
                    class_name = self.class_names[idx]
                    prob = all_probs[idx]
                    f.write(f"    {class_name}: {prob:.4f}\n")
                f.write("\n")
        
        logger.info(f"Saved classification debug info to {debug_file}")
        
        # Create confidence heatmap
        self._create_confidence_heatmap(predictions, debug_dir)
    
    def _create_confidence_heatmap(self, predictions: Dict[str, Dict], debug_dir: str):
        """Create a confidence heatmap visualization."""
        import matplotlib.pyplot as plt
        
        # Create 8x8 grid for confidence values
        confidence_grid = np.zeros((8, 8))
        
        for square_name, prediction in predictions.items():
            # Convert square name to coordinates
            file_letter = square_name[0]
            rank_number = square_name[1]
            
            col = ord(file_letter) - ord('a')  # a=0, b=1, ..., h=7
            row = 8 - int(rank_number)  # 8=0, 7=1, ..., 1=7
            
            confidence_grid[row, col] = prediction['confidence']
        
        # Create heatmap
        plt.figure(figsize=(8, 8))
        plt.imshow(confidence_grid, cmap='RdYlGn', vmin=0, vmax=1)
        plt.colorbar(label='Confidence')
        
        # Add grid lines
        plt.grid(True, which='major', color='black', linewidth=2)
        plt.xticks(np.arange(-0.5, 8, 1), [])
        plt.yticks(np.arange(-0.5, 8, 1), [])
        
        # Add square labels
        for i in range(8):
            for j in range(8):
                file_letter = chr(ord('a') + j)
                rank_number = 8 - i
                square_name = f"{file_letter}{rank_number}"
                
                # Get piece symbol
                if square_name in predictions:
                    piece = predictions[square_name]['predicted_class']
                    symbol = self.piece_mapping.get(piece, '?')
                    confidence = predictions[square_name]['confidence']
                    
                    # Choose text color based on background
                    text_color = 'black' if confidence > 0.5 else 'white'
                    
                    plt.text(j, i, f"{symbol}\n{confidence:.2f}", 
                            ha='center', va='center', fontsize=10, 
                            fontweight='bold', color=text_color)
        
        plt.title('Confidence Heatmap with Piece Predictions')
        plt.tight_layout()
        
        heatmap_path = os.path.join(debug_dir, "05_confidence_heatmap.png")
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confidence heatmap to {heatmap_path}")
    
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
    
    parser = argparse.ArgumentParser(description='Chess Position Analyzer - Single Image or Batch Processing')
    parser.add_argument('input', help='Path to chess board image, directory of images, or list of image paths')
    parser.add_argument('--model', default='chess_piece_classifier.pth', help='Path to trained model')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], 
                       help='Device to use')
    parser.add_argument('--debug', action='store_true', help='Save debug visualizations')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    parser.add_argument('--batch', action='store_true', help='Force batch processing mode')
    parser.add_argument('--output-dir', default='analysis_results', help='Output directory for batch results')
    parser.add_argument('--summary', action='store_true', help='Generate summary report for batch processing')
    
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
        
        # Determine if this is batch processing
        is_batch = args.batch or os.path.isdir(args.input)
        
        if is_batch:
            # Batch processing
            print(f"Starting batch analysis of: {args.input}")
            
            # Create output directory
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Analyze batch
            results = analyzer.analyze_batch(
                image_paths=args.input,
                confidence_threshold=args.confidence,
                save_debug=args.debug,
                debug_dir=args.output_dir
            )
            
            # Process results
            successful_results = [r for r in results if r.get('board_detected', False)]
            failed_results = [r for r in results if not r.get('board_detected', False)]
            
            print(f"\nBatch Analysis Summary:")
            print(f"  Total images processed: {len(results)}")
            print(f"  Successful analyses: {len(successful_results)}")
            print(f"  Failed analyses: {len(failed_results)}")
            
            # Save individual results
            for i, result in enumerate(results):
                if result.get('board_detected', False):
                    image_path = result.get('image_path', f'image_{i}')
                    image_name = Path(image_path).stem
                    
                    # Save analysis result
                    output_file = os.path.join(args.output_dir, f"{image_name}_analysis.txt")
                    with open(output_file, 'w') as f:
                        f.write("CHESS POSITION ANALYSIS\n")
                        f.write("="*50 + "\n\n")
                        f.write(f"Image: {image_path}\n\n")
                        f.write("Position:\n")
                        f.write(result['position_string'] + "\n\n")
                        
                        # Write detailed piece predictions
                        f.write("Detailed Predictions:\n")
                        for square in sorted(result['piece_predictions'].keys()):
                            pred = result['piece_predictions'][square]
                            f.write(f"{square}: {pred['predicted_class']} "
                                   f"(confidence: {pred['confidence']:.3f})\n")
                    
                    print(f"  ✅ {image_name}: Analysis saved to {output_file}")
                else:
                    image_path = result.get('image_path', f'image_{i}')
                    print(f"  ❌ {Path(image_path).name}: {result.get('error', 'Unknown error')}")
            
            # Generate summary report if requested
            if args.summary and successful_results:
                summary_file = os.path.join(args.output_dir, "batch_summary.txt")
                with open(summary_file, 'w') as f:
                    f.write("BATCH ANALYSIS SUMMARY\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"Total images: {len(results)}\n")
                    f.write(f"Successful: {len(successful_results)}\n")
                    f.write(f"Failed: {len(failed_results)}\n\n")
                    
                    f.write("SUCCESSFUL ANALYSES:\n")
                    f.write("-" * 30 + "\n")
                    for result in successful_results:
                        image_path = result.get('image_path', 'Unknown')
                        stats = result.get('statistics', {})
                        f.write(f"{Path(image_path).name}:\n")
                        f.write(f"  Pieces detected: {stats.get('total_pieces', 0)}\n")
                        f.write(f"  Avg confidence: {stats.get('average_confidence', 0):.3f}\n")
                        f.write(f"  Position: {result.get('position_string', 'N/A')}\n\n")
                    
                    if failed_results:
                        f.write("FAILED ANALYSES:\n")
                        f.write("-" * 20 + "\n")
                        for result in failed_results:
                            image_path = result.get('image_path', 'Unknown')
                            error = result.get('error', 'Unknown error')
                            f.write(f"{Path(image_path).name}: {error}\n")
                
                print(f"\nSummary report saved to: {summary_file}")
            
            print(f"\nAll results saved to: {args.output_dir}/")
            
        else:
            # Single image processing
            result = analyzer.analyze_position(
                image_path=args.input,
                confidence_threshold=args.confidence,
                save_debug=args.debug
            )
            
            # Print results
            analyzer.print_analysis(result)
            
            # Save results to file
            output_file = f"{os.path.splitext(args.input)[0]}_analysis.txt"
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