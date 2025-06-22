#!/usr/bin/env python3
"""
Simple example of using the complete chess board analyzer.
"""

from chess_board_analyzer import ChessBoardAnalyzer


def analyze_chess_board(image_path: str):
    """
    Simple function to analyze a chess board and print the results.
    
    Args:
        image_path: Path to the chess board image
    """
    # Initialize the analyzer
    analyzer = ChessBoardAnalyzer()
    
    # Analyze the image
    result = analyzer.analyze_image(image_path)
    
    if result['success']:
        # Print the board in a nice format
        analyzer.print_board(result['board_grid'])
        
        # Also print just the raw grid (8x8 matrix)
        print("\nRaw 8x8 grid:")
        for row in result['board_grid']:
            print(''.join(row))
    else:
        print(f"❌ Analysis failed: {result['error']}")


if __name__ == "__main__":
    # Example usage
    analyze_chess_board("sample_chess_frame.jpg")