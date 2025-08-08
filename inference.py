import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor
from chess_piece_classifier import ChessPieceClassifier
import matplotlib.pyplot as plt


def load_model(model_path="chess_piece_classifier.pth", device='cpu'):
    """Load the trained chess piece classifier."""
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    model = ChessPieceClassifier(num_classes=13, freeze_backbone=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    class_names = checkpoint['class_names']
    
    return model, class_names


def predict_image(model, image_path, class_names, device='cpu'):
    """Predict the chess piece in an image."""
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    
    # Resize to 224x224 (same as training - DINOv2 uses 224x224)
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    
    # Use the same feature extractor as training
    feature_extractor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-base", use_fast=True
    )
    
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(pixel_values)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    predicted_label = class_names[predicted_class]
    
    return predicted_label, confidence, probabilities[0].cpu().numpy()


def visualize_prediction(image_path, predicted_label, confidence, probabilities, class_names):
    """Visualize the prediction with confidence scores."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show image
    image = Image.open(image_path)
    ax1.imshow(image)
    ax1.set_title(f'Predicted: {predicted_label}\nConfidence: {confidence:.3f}')
    ax1.axis('off')
    
    # Show probability distribution
    y_pos = np.arange(len(class_names))
    ax2.barh(y_pos, probabilities)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel('Probability')
    ax2.set_title('Class Probabilities')
    ax2.grid(True, alpha=0.3)
    
    # Highlight predicted class
    ax2.barh(predicted_class, probabilities[predicted_class], color='red', alpha=0.7)
    
    plt.tight_layout()
    plt.show()


def batch_predict(model, image_paths, class_names, device='cpu'):
    """Predict multiple images at once."""
    
    feature_extractor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-base", use_fast=True
    )
    
    results = []
    
    for image_path in image_paths:
        try:
            predicted_label, confidence, probabilities = predict_image(
                model, image_path, class_names, device
            )
            results.append({
                'image_path': image_path,
                'predicted_label': predicted_label,
                'confidence': confidence
            })
            print(f"{image_path}: {predicted_label} (confidence: {confidence:.3f})")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({
                'image_path': image_path,
                'predicted_label': 'ERROR',
                'confidence': 0.0
            })
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Chess Piece Classifier Inference')
    parser.add_argument('--model', default='chess_piece_classifier.pth', help='Path to trained model')
    parser.add_argument('--image', help='Path to image to classify')
    parser.add_argument('--images', nargs='+', help='Paths to multiple images to classify')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], help='Device to use (auto detects best available)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        mps_ok = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        if mps_ok:
            device = 'mps'
            print("Auto-detected MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            device = 'cuda'
            print("Auto-detected CUDA (NVIDIA GPU)")
        else:
            device = 'cpu'
            print("Auto-detected CPU")
    else:
        device = args.device
    
    # Load model
    print("Loading model...")
    model, class_names = load_model(args.model, device)
    print(f"Model loaded. Classes: {class_names}")
    
    if args.image:
        # Single image prediction
        predicted_label, confidence, probabilities = predict_image(
            model, args.image, class_names, device
        )
        print(f"Prediction: {predicted_label}")
        print(f"Confidence: {confidence:.3f}")
        
        # Visualize
        predicted_class = class_names.index(predicted_label)
        visualize_prediction(args.image, predicted_label, confidence, probabilities, class_names)
        
    elif args.images:
        # Batch prediction
        results = batch_predict(model, args.images, class_names, device)
        
        # Print summary
        print("\nSummary:")
        for result in results:
            print(f"{result['image_path']}: {result['predicted_label']} ({result['confidence']:.3f})")
    
    else:
        print("Please provide either --image or --images argument")


if __name__ == "__main__":
    main() 