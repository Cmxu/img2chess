#!/usr/bin/env python3
"""
Test script for the enhanced Chess Position Analyzer with batch processing capabilities.

This script demonstrates:
1. Single image analysis
2. Batch processing of multiple images
3. Directory-based batch processing
4. Performance comparison between single and batch modes
"""

import os
import time
import logging
from pathlib import Path
from chess_position_analyzer import ChessPositionAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_single_image():
    """Test single image analysis."""
    print("\n" + "="*60)
    print("TESTING SINGLE IMAGE ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = ChessPositionAnalyzer()
    
    # Test with sample image if it exists
    test_image = "sample_chess_frame.jpg"
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        print("Please ensure you have a chess board image to test with.")
        return False
    
    try:
        start_time = time.time()
        
        # Analyze single image
        result = analyzer.analyze_position(
            image_path=test_image,
            confidence_threshold=0.5,
            save_debug=True
        )
        
        end_time = time.time()
        
        # Print results
        analyzer.print_analysis(result)
        print(f"\n⏱️  Analysis completed in {end_time - start_time:.2f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"Single image analysis failed: {e}")
        return False


def test_batch_processing():
    """Test batch processing with multiple images."""
    print("\n" + "="*60)
    print("TESTING BATCH PROCESSING")
    print("="*60)
    
    # Initialize analyzer
    analyzer = ChessPositionAnalyzer()
    
    # Create test images list (you can modify this to test with your own images)
    test_images = []
    
    # Check for sample image
    if os.path.exists("sample_chess_frame.jpg"):
        test_images.append("sample_chess_frame.jpg")
    
    # Check for other potential test images
    potential_images = [
        "test_evaluation/test_board_1.jpg",
        "test_evaluation/test_board_2.jpg", 
        "examples/chess_board_1.jpg",
        "examples/chess_board_2.jpg"
    ]
    
    for img_path in potential_images:
        if os.path.exists(img_path):
            test_images.append(img_path)
    
    if not test_images:
        print("❌ No test images found for batch processing")
        print("Please add some chess board images to test with.")
        return False
    
    print(f"Found {len(test_images)} test images for batch processing")
    
    try:
        start_time = time.time()
        
        # Analyze batch
        results = analyzer.analyze_batch(
            image_paths=test_images,
            confidence_threshold=0.5,
            save_debug=True,
            debug_dir="batch_test_output"
        )
        
        end_time = time.time()
        
        # Process results
        successful = [r for r in results if r.get('board_detected', False)]
        failed = [r for r in results if not r.get('board_detected', False)]
        
        print(f"\n📊 Batch Processing Results:")
        print(f"  Total images: {len(results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        print(f"  ⏱️  Total time: {end_time - start_time:.2f} seconds")
        print(f"  ⏱️  Average time per image: {(end_time - start_time) / len(results):.2f} seconds")
        
        # Show successful results
        if successful:
            print(f"\n✅ Successful Analyses:")
            for result in successful:
                image_path = result.get('image_path', 'Unknown')
                stats = result.get('statistics', {})
                print(f"  {Path(image_path).name}: {stats.get('total_pieces', 0)} pieces, "
                      f"avg confidence: {stats.get('average_confidence', 0):.3f}")
        
        # Show failed results
        if failed:
            print(f"\n❌ Failed Analyses:")
            for result in failed:
                image_path = result.get('image_path', 'Unknown')
                error = result.get('error', 'Unknown error')
                print(f"  {Path(image_path).name}: {error}")
        
        return len(successful) > 0
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return False


def test_directory_processing():
    """Test processing all images in a directory."""
    print("\n" + "="*60)
    print("TESTING DIRECTORY PROCESSING")
    print("="*60)
    
    # Initialize analyzer
    analyzer = ChessPositionAnalyzer()
    
    # Test directories
    test_dirs = [
        "test_evaluation",
        "examples", 
        "debug_output"
    ]
    
    test_dir = None
    for dir_path in test_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            # Check if directory contains images
            image_files = analyzer._get_image_files(dir_path)
            if image_files:
                test_dir = dir_path
                break
    
    if not test_dir:
        print("❌ No suitable test directory found with images")
        print("Please ensure you have a directory with chess board images.")
        return False
    
    print(f"Testing with directory: {test_dir}")
    
    try:
        start_time = time.time()
        
        # Analyze directory
        results = analyzer.analyze_batch(
            image_paths=test_dir,
            confidence_threshold=0.5,
            save_debug=True,
            debug_dir="directory_test_output"
        )
        
        end_time = time.time()
        
        # Process results
        successful = [r for r in results if r.get('board_detected', False)]
        failed = [r for r in results if not r.get('board_detected', False)]
        
        print(f"\n📊 Directory Processing Results:")
        print(f"  Directory: {test_dir}")
        print(f"  Total images: {len(results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        print(f"  ⏱️  Total time: {end_time - start_time:.2f} seconds")
        
        return len(successful) > 0
        
    except Exception as e:
        logger.error(f"Directory processing failed: {e}")
        return False


def performance_comparison():
    """Compare performance between single and batch processing."""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    # Initialize analyzer
    analyzer = ChessPositionAnalyzer()
    
    # Test with a few images
    test_images = []
    if os.path.exists("sample_chess_frame.jpg"):
        test_images.append("sample_chess_frame.jpg")
    
    # Add more test images if available
    for i in range(1, 4):
        test_path = f"test_evaluation/test_board_{i}.jpg"
        if os.path.exists(test_path):
            test_images.append(test_path)
    
    if len(test_images) < 2:
        print("❌ Need at least 2 test images for performance comparison")
        return False
    
    print(f"Comparing performance with {len(test_images)} images")
    
    # Test single image processing (one by one)
    print("\n🔄 Testing single image processing...")
    single_start = time.time()
    
    for image_path in test_images:
        try:
            result = analyzer.analyze_position(
                image_path=image_path,
                confidence_threshold=0.5,
                save_debug=False
            )
        except Exception as e:
            logger.warning(f"Failed to process {image_path}: {e}")
    
    single_end = time.time()
    single_time = single_end - single_start
    
    # Test batch processing
    print("🔄 Testing batch processing...")
    batch_start = time.time()
    
    try:
        results = analyzer.analyze_batch(
            image_paths=test_images,
            confidence_threshold=0.5,
            save_debug=False,
            debug_dir="performance_test_output"
        )
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return False
    
    batch_end = time.time()
    batch_time = batch_end - batch_start
    
    # Compare results
    print(f"\n📈 Performance Comparison:")
    print(f"  Single image processing: {single_time:.2f} seconds")
    print(f"  Batch processing: {batch_time:.2f} seconds")
    print(f"  Speedup: {single_time / batch_time:.2f}x")
    print(f"  Time saved: {single_time - batch_time:.2f} seconds")
    
    return True


def main():
    """Run all tests."""
    print("🧪 CHESS POSITION ANALYZER - BATCH PROCESSING TESTS")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists("chess_piece_classifier.pth"):
        print("❌ Model file not found: chess_piece_classifier.pth")
        print("Please ensure you have trained the chess piece classifier model.")
        return 1
    
    # Run tests
    tests = [
        ("Single Image Analysis", test_single_image),
        ("Batch Processing", test_batch_processing),
        ("Directory Processing", test_directory_processing),
        ("Performance Comparison", performance_comparison)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}")
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The batch processing is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit(main()) 