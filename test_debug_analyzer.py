#!/usr/bin/env python3
"""
Test script for the enhanced chess position analyzer with debug functionality.
"""

import os
import sys
from chess_position_analyzer import ChessPositionAnalyzer

def test_analyzer_with_debug(image_path):
    """Test the chess position analyzer with debug output enabled."""
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False
    
    print("="*60)
    print("TESTING CHESS POSITION ANALYZER WITH DEBUG")
    print("="*60)
    print(f"Image: {image_path}")
    print()
    
    try:
        # Initialize analyzer
        print("Initializing analyzer...")
        analyzer = ChessPositionAnalyzer(
            model_path="chess_piece_classifier.pth",
            device="auto"
        )
        
        # Analyze position with debug enabled
        print("Analyzing position with debug output...")
        result = analyzer.analyze_position(
            image_path=image_path,
            confidence_threshold=0.5,
            save_debug=True
        )
        
        # Print results
        print("\nAnalysis Results:")
        analyzer.print_analysis(result)
        
        # List debug files created
        debug_dir = "debug_output"
        if os.path.exists(debug_dir):
            print(f"\nDebug files created in '{debug_dir}':")
            files = sorted([f for f in os.listdir(debug_dir) if f.endswith('.png')])
            for file in files:
                print(f"  - {file}")
            
            # Check for squares directory
            squares_dir = os.path.join(debug_dir, "squares")
            if os.path.exists(squares_dir):
                square_files = [f for f in os.listdir(squares_dir) if f.endswith('.png')]
                print(f"  - squares/ ({len(square_files)} individual square images)")
            
            # Check for text files
            text_files = [f for f in os.listdir(debug_dir) if f.endswith('.txt')]
            for file in text_files:
                print(f"  - {file}")
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Debug output saved to: {debug_dir}/")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the test."""
    
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("Usage: python test_debug_analyzer.py <image_path>")
        print("\nExample:")
        print("  python test_debug_analyzer.py sample_chess_frame.jpg")
        print("  python test_debug_analyzer.py boards/board1.png")
        return 1
    
    image_path = sys.argv[1]
    
    # Run the test
    success = test_analyzer_with_debug(image_path)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 