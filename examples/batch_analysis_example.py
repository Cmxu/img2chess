#!/usr/bin/env python3
"""
Example script demonstrating the enhanced Chess Position Analyzer with batch processing.

This example shows:
1. How to analyze a single chess board image
2. How to process multiple images in batch
3. How to process all images in a directory
4. How to handle results and errors
5. How to generate summary reports
"""

import os
import sys
import time
import json
from pathlib import Path

# Add parent directory to path to import chess_position_analyzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chess_position_analyzer import ChessPositionAnalyzer


def example_single_image():
    """Example: Analyze a single chess board image."""
    print("="*60)
    print("EXAMPLE: Single Image Analysis")
    print("="*60)
    
    # Initialize analyzer
    analyzer = ChessPositionAnalyzer(
        model_path="chess_piece_classifier.pth",
        device="auto"  # Automatically detect best device
    )
    
    # Example image path (modify as needed)
    image_path = "sample_chess_frame.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Example image not found: {image_path}")
        print("Please ensure you have a chess board image to analyze.")
        return
    
    try:
        print(f"🔍 Analyzing: {image_path}")
        start_time = time.time()
        
        # Analyze the image
        result = analyzer.analyze_position(
            image_path=image_path,
            confidence_threshold=0.5,
            save_debug=True
        )
        
        end_time = time.time()
        
        # Print results
        analyzer.print_analysis(result)
        print(f"\n⏱️  Analysis completed in {end_time - start_time:.2f} seconds")
        
        # Access specific results
        print(f"\n📊 Quick Stats:")
        print(f"  Total pieces: {result['statistics']['total_pieces']}")
        print(f"  Average confidence: {result['statistics']['average_confidence']:.3f}")
        print(f"  Position string length: {len(result['position_string'])} characters")
        
        return result
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None


def example_batch_images():
    """Example: Process multiple specific images."""
    print("\n" + "="*60)
    print("EXAMPLE: Batch Image Processing")
    print("="*60)
    
    # Initialize analyzer
    analyzer = ChessPositionAnalyzer()
    
    # List of images to process (modify as needed)
    image_paths = [
        "sample_chess_frame.jpg",
        # Add more images here:
        # "path/to/chess_board_1.jpg",
        # "path/to/chess_board_2.jpg",
    ]
    
    # Filter to only existing images
    existing_images = [path for path in image_paths if os.path.exists(path)]
    
    if not existing_images:
        print("❌ No example images found for batch processing")
        print("Please add some chess board images to analyze.")
        return
    
    print(f"🔍 Processing {len(existing_images)} images in batch...")
    start_time = time.time()
    
    try:
        # Process batch
        results = analyzer.analyze_batch(
            image_paths=existing_images,
            confidence_threshold=0.5,
            save_debug=True,
            debug_dir="batch_example_output"
        )
        
        end_time = time.time()
        
        # Process results
        successful = [r for r in results if r.get('board_detected', False)]
        failed = [r for r in results if not r.get('board_detected', False)]
        
        print(f"\n📊 Batch Results:")
        print(f"  Total processed: {len(results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        print(f"  ⏱️  Total time: {end_time - start_time:.2f} seconds")
        print(f"  ⏱️  Average per image: {(end_time - start_time) / len(results):.2f} seconds")
        
        # Show successful results
        if successful:
            print(f"\n✅ Successful Analyses:")
            for result in successful:
                image_name = Path(result['image_path']).name
                stats = result['statistics']
                print(f"  {image_name}: {stats['total_pieces']} pieces, "
                      f"confidence: {stats['average_confidence']:.3f}")
        
        # Show failed results
        if failed:
            print(f"\n❌ Failed Analyses:")
            for result in failed:
                image_name = Path(result['image_path']).name
                error = result.get('error', 'Unknown error')
                print(f"  {image_name}: {error}")
        
        return results
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return None


def example_directory_processing():
    """Example: Process all images in a directory."""
    print("\n" + "="*60)
    print("EXAMPLE: Directory Processing")
    print("="*60)
    
    # Initialize analyzer
    analyzer = ChessPositionAnalyzer()
    
    # Directory to process (modify as needed)
    directory = "test_evaluation"  # or any directory with chess images
    
    if not os.path.exists(directory) or not os.path.isdir(directory):
        print(f"❌ Directory not found: {directory}")
        print("Please specify a directory containing chess board images.")
        return
    
    # Check if directory contains images
    image_files = analyzer._get_image_files(directory)
    if not image_files:
        print(f"❌ No images found in directory: {directory}")
        return
    
    print(f"🔍 Processing {len(image_files)} images from directory: {directory}")
    start_time = time.time()
    
    try:
        # Process directory
        results = analyzer.analyze_batch(
            image_paths=directory,
            confidence_threshold=0.5,
            save_debug=True,
            debug_dir="directory_example_output"
        )
        
        end_time = time.time()
        
        # Process results
        successful = [r for r in results if r.get('board_detected', False)]
        failed = [r for r in results if not r.get('board_detected', False)]
        
        print(f"\n📊 Directory Processing Results:")
        print(f"  Directory: {directory}")
        print(f"  Images found: {len(image_files)}")
        print(f"  Successfully processed: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        print(f"  ⏱️  Total time: {end_time - start_time:.2f} seconds")
        
        # Generate summary report
        if successful:
            summary = generate_summary_report(successful, directory)
            print(f"\n📋 Summary Report:")
            print(summary)
            
            # Save summary to file
            summary_file = "directory_analysis_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(summary)
            print(f"📄 Summary saved to: {summary_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Directory processing failed: {e}")
        return None


def example_advanced_analysis():
    """Example: Advanced analysis with custom settings and result processing."""
    print("\n" + "="*60)
    print("EXAMPLE: Advanced Analysis")
    print("="*60)
    
    # Initialize analyzer with custom settings
    analyzer = ChessPositionAnalyzer(
        model_path="chess_piece_classifier.pth",
        device="auto"
    )
    
    # Example image
    image_path = "sample_chess_frame.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Example image not found: {image_path}")
        return
    
    try:
        print(f"🔍 Advanced analysis of: {image_path}")
        
        # Analyze with different confidence thresholds
        confidence_levels = [0.3, 0.5, 0.7, 0.9]
        results = {}
        
        for confidence in confidence_levels:
            print(f"\n  Testing confidence threshold: {confidence}")
            
            result = analyzer.analyze_position(
                image_path=image_path,
                confidence_threshold=confidence,
                save_debug=False  # Disable debug for faster processing
            )
            
            results[confidence] = result
            
            # Show quick stats
            stats = result['statistics']
            print(f"    Pieces detected: {stats['total_pieces']}")
            print(f"    Average confidence: {stats['average_confidence']:.3f}")
            print(f"    Low confidence squares: {len(stats['low_confidence_squares'])}")
        
        # Compare results
        print(f"\n📊 Confidence Threshold Comparison:")
        print(f"{'Threshold':<12} {'Pieces':<8} {'Avg Conf':<10} {'Low Conf':<10}")
        print("-" * 40)
        
        for confidence, result in results.items():
            stats = result['statistics']
            print(f"{confidence:<12} {stats['total_pieces']:<8} "
                  f"{stats['average_confidence']:<10.3f} {len(stats['low_confidence_squares']):<10}")
        
        return results
        
    except Exception as e:
        print(f"❌ Advanced analysis failed: {e}")
        return None


def generate_summary_report(results, directory_name):
    """Generate a summary report for batch results."""
    if not results:
        return "No successful analyses to report."
    
    # Calculate overall statistics
    total_pieces = sum(r['statistics']['total_pieces'] for r in results)
    avg_confidence = sum(r['statistics']['average_confidence'] for r in results) / len(results)
    
    # Count piece types across all results
    piece_counts = {}
    for result in results:
        for piece, count in result['statistics']['piece_counts'].items():
            if piece != 'empty':
                piece_counts[piece] = piece_counts.get(piece, 0) + count
    
    # Generate report
    report = f"""
DIRECTORY ANALYSIS SUMMARY
==========================
Directory: {directory_name}
Total successful analyses: {len(results)}
Total pieces detected: {total_pieces}
Average confidence: {avg_confidence:.3f}

PIECE DISTRIBUTION:
"""
    
    # Sort pieces by count
    sorted_pieces = sorted(piece_counts.items(), key=lambda x: x[1], reverse=True)
    for piece, count in sorted_pieces:
        report += f"  {piece}: {count}\n"
    
    report += f"""
INDIVIDUAL RESULTS:
"""
    
    for result in results:
        image_name = Path(result['image_path']).name
        stats = result['statistics']
        report += f"  {image_name}: {stats['total_pieces']} pieces, "
        report += f"confidence: {stats['average_confidence']:.3f}\n"
    
    return report


def main():
    """Run all examples."""
    print("🎯 CHESS POSITION ANALYZER - BATCH PROCESSING EXAMPLES")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists("chess_piece_classifier.pth"):
        print("❌ Model file not found: chess_piece_classifier.pth")
        print("Please ensure you have trained the chess piece classifier model.")
        return 1
    
    # Run examples
    examples = [
        ("Single Image Analysis", example_single_image),
        ("Batch Image Processing", example_batch_images),
        ("Directory Processing", example_directory_processing),
        ("Advanced Analysis", example_advanced_analysis)
    ]
    
    results = {}
    
    for example_name, example_func in examples:
        try:
            print(f"\n🧪 Running: {example_name}")
            results[example_name] = example_func()
        except Exception as e:
            print(f"❌ Example '{example_name}' failed: {e}")
            results[example_name] = None
    
    # Summary
    print("\n" + "="*60)
    print("EXAMPLE SUMMARY")
    print("="*60)
    
    successful = 0
    total = len(examples)
    
    for example_name, result in results.items():
        status = "✅ COMPLETED" if result is not None else "❌ FAILED"
        print(f"  {example_name}: {status}")
        if result is not None:
            successful += 1
    
    print(f"\nOverall: {successful}/{total} examples completed successfully")
    
    if successful > 0:
        print("\n🎉 Examples completed! Check the output directories for results:")
        print("  - debug_output/ (single image debug)")
        print("  - batch_example_output/ (batch processing debug)")
        print("  - directory_example_output/ (directory processing debug)")
        print("  - directory_analysis_summary.txt (summary report)")
    
    return 0 if successful > 0 else 1


if __name__ == "__main__":
    exit(main()) 