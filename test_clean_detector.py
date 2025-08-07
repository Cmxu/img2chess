#!/usr/bin/env python3
"""
Test script for the clean edge-based detector.
"""

import cv2
import os
import sys
from img2chess.clean_edge_detector import CleanEdgeBasedDetector


def test_detector():
    """Test the clean detector on sample images."""
    
    # Initialize detector with config
    config_path = "detector_config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return
    
    try:
        detector = CleanEdgeBasedDetector(config_path)
        print(f"✅ Detector initialized with config: {config_path}")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return
    
    # Test on a sample image if available
    test_images = [
        "test_frames/large_frame_data",  # Look for sample frames
        "test_image.jpg",
        "sample.png"
    ]
    
    found_test_image = None
    
    # Look for test frames directory
    if os.path.exists("test_frames/large_frame_data"):
        for video_dir in os.listdir("test_frames/large_frame_data"):
            video_path = os.path.join("test_frames/large_frame_data", video_dir)
            if os.path.isdir(video_path):
                frames = [f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))]
                if frames:
                    found_test_image = os.path.join(video_path, frames[0])
                    break
    
    if found_test_image is None:
        print("⚠️  No test images found. Please provide a test image to verify the detector.")
        print("\nExample usage:")
        print("```python")
        print("import cv2")
        print("from img2chess.clean_edge_detector import CleanEdgeBasedDetector")
        print("")
        print("# Initialize detector")
        print("detector = CleanEdgeBasedDetector('detector_config.yaml')")
        print("")
        print("# Load image")
        print("image = cv2.imread('your_chess_image.jpg')")
        print("")
        print("# Detect board")
        print("board = detector.detect_board(image)")
        print("if board is not None:")
        print("    cv2.imwrite('detected_board.jpg', board)")
        print("    print('✅ Board detected and saved!')")
        print("else:")
        print("    print('❌ No board detected')")
        print("```")
        return
    
    print(f"🧪 Testing with image: {found_test_image}")
    
    # Load test image
    image = cv2.imread(found_test_image)
    if image is None:
        print(f"❌ Could not load image: {found_test_image}")
        return
    
    print(f"📷 Image loaded: {image.shape}")
    
    # Detect board
    try:
        board = detector.detect_board(image)
        
        if board is not None:
            output_path = "test_detected_board.jpg"
            cv2.imwrite(output_path, board)
            print(f"✅ Board detected! Shape: {board.shape}")
            print(f"💾 Saved to: {output_path}")
        else:
            print("❌ No board detected")
            
    except Exception as e:
        print(f"❌ Detection failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_detector()