import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import argparse
from PIL import Image

from chess_piece_classifier import ChessPieceClassifier, ChessPieceDataset
from torch.utils.data import DataLoader


def load_model(model_path="chess_piece_classifier.pth", device='cpu'):
    """Load the trained chess piece classifier."""
    
    print(f"Loading model from {model_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    model = ChessPieceClassifier(num_classes=13, freeze_backbone=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    class_names = checkpoint['class_names']
    
    print(f"Model loaded successfully!")
    print(f"Classes: {class_names}")
    
    return model, class_names


def evaluate_model(model, dataloader, class_names, device='cpu', save_wrong_images=False, wrong_images_dir=None):
    """Evaluate the model on the given dataloader."""
    
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_confidences = []
    wrong_samples = []  # Store wrong predictions for visualization
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(outputs, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]
            
            # Store results
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            
            # Store wrong predictions for visualization
            if save_wrong_images:
                for i in range(images.size(0)):
                    if predicted_classes[i] != labels[i]:
                        # Convert tensor back to image
                        img_tensor = images[i].cpu()
                        # Denormalize (DINOv2 uses ImageNet normalization)
                        mean = torch.tensor([0.485, 0.456, 0.406])
                        std = torch.tensor([0.229, 0.224, 0.225])
                        img_denorm = img_tensor * std.unsqueeze(-1).unsqueeze(-1) + mean.unsqueeze(-1).unsqueeze(-1)
                        img_denorm = torch.clamp(img_denorm, 0, 1)
                        img_denorm = img_denorm.permute(1, 2, 0).numpy()
                        
                        wrong_samples.append({
                            'image': img_denorm,
                            'true_label': labels[i].item(),
                            'predicted_label': predicted_classes[i].item(),
                            'confidence': confidences[i].item(),
                            'sample_idx': batch_idx * dataloader.batch_size + i
                        })
    
    return {
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probabilities),
        'confidences': np.array(all_confidences),
        'wrong_samples': wrong_samples
    }


def calculate_metrics(results, class_names):
    """Calculate comprehensive evaluation metrics."""
    
    predictions = results['predictions']
    labels = results['labels']
    confidences = results['confidences']
    
    # Overall accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Per-class accuracy
    class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_mask = labels == i
        if class_mask.sum() > 0:
            class_acc = accuracy_score(labels[class_mask], predictions[class_mask])
            class_accuracy[class_name] = class_acc
        else:
            class_accuracy[class_name] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Classification report
    report = classification_report(labels, predictions, target_names=class_names, output_dict=True)
    
    # Confidence analysis
    correct_predictions = predictions == labels
    avg_confidence_correct = confidences[correct_predictions].mean() if correct_predictions.sum() > 0 else 0
    avg_confidence_incorrect = confidences[~correct_predictions].mean() if (~correct_predictions).sum() > 0 else 0
    
    return {
        'accuracy': accuracy,
        'class_accuracy': class_accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'avg_confidence_correct': avg_confidence_correct,
        'avg_confidence_incorrect': avg_confidence_incorrect,
        'total_samples': len(labels)
    }


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """Plot confusion matrix."""
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_class_accuracy(class_accuracy, save_path='class_accuracy.png'):
    """Plot per-class accuracy."""
    
    classes = list(class_accuracy.keys())
    accuracies = list(class_accuracy.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Per-Class Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Class accuracy plot saved to {save_path}")


def plot_confidence_distribution(results, save_path='confidence_distribution.png'):
    """Plot confidence distribution for correct vs incorrect predictions."""
    
    predictions = results['predictions']
    labels = results['labels']
    confidences = results['confidences']
    
    correct_predictions = predictions == labels
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(confidences[correct_predictions], bins=20, alpha=0.7, 
             label='Correct Predictions', color='green', density=True)
    plt.hist(confidences[~correct_predictions], bins=20, alpha=0.7, 
             label='Incorrect Predictions', color='red', density=True)
    
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.title('Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confidence distribution plot saved to {save_path}")


def print_detailed_report(metrics, class_names):
    """Print detailed evaluation report."""
    
    print("\n" + "="*60)
    print("MODEL EVALUATION REPORT")
    print("="*60)
    
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Total Samples: {metrics['total_samples']:,}")
    
    print(f"\nAverage Confidence:")
    print(f"  Correct predictions: {metrics['avg_confidence_correct']:.4f}")
    print(f"  Incorrect predictions: {metrics['avg_confidence_incorrect']:.4f}")
    
    print(f"\nPer-Class Accuracy:")
    for class_name, acc in metrics['class_accuracy'].items():
        print(f"  {class_name:>4}: {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(
        [class_names[i] for i in range(len(class_names))], 
        [class_names[i] for i in range(len(class_names))], 
        target_names=class_names,
        output_dict=False
    ))
    
    # Find best and worst performing classes
    sorted_classes = sorted(metrics['class_accuracy'].items(), key=lambda x: x[1], reverse=True)
    print(f"\nBest performing classes:")
    for i, (class_name, acc) in enumerate(sorted_classes[:3]):
        print(f"  {i+1}. {class_name}: {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"\nWorst performing classes:")
    for i, (class_name, acc) in enumerate(sorted_classes[-3:]):
        print(f"  {i+1}. {class_name}: {acc:.4f} ({acc*100:.2f}%)")


def save_results_to_csv(results, metrics, class_names, save_path='evaluation_results.csv'):
    """Save evaluation results to CSV."""
    
    # Create detailed results DataFrame
    df_results = pd.DataFrame({
        'true_label': [class_names[label] for label in results['labels']],
        'predicted_label': [class_names[pred] for pred in results['predictions']],
        'confidence': results['confidences'],
        'correct': results['predictions'] == results['labels']
    })
    
    # Create metrics DataFrame
    df_metrics = pd.DataFrame([
        {'metric': 'overall_accuracy', 'value': metrics['accuracy']},
        {'metric': 'avg_confidence_correct', 'value': metrics['avg_confidence_correct']},
        {'metric': 'avg_confidence_incorrect', 'value': metrics['avg_confidence_incorrect']},
        {'metric': 'total_samples', 'value': metrics['total_samples']}
    ])
    
    # Create class accuracy DataFrame
    df_class_acc = pd.DataFrame([
        {'class': class_name, 'accuracy': acc} 
        for class_name, acc in metrics['class_accuracy'].items()
    ])
    
    # Save to CSV
    with pd.ExcelWriter(save_path.replace('.csv', '.xlsx')) as writer:
        df_results.to_excel(writer, sheet_name='Detailed_Results', index=False)
        df_metrics.to_excel(writer, sheet_name='Overall_Metrics', index=False)
        df_class_acc.to_excel(writer, sheet_name='Class_Accuracy', index=False)
    
    print(f"Results saved to {save_path.replace('.csv', '.xlsx')}")


def save_wrong_images(wrong_samples, class_names, output_dir):
    """Save images of incorrectly classified samples."""
    
    if not wrong_samples:
        print("No wrong predictions to save!")
        return
    
    wrong_images_dir = os.path.join(output_dir, 'wrong_predictions')
    
    # Empty the directory if it exists
    if os.path.exists(wrong_images_dir):
        print(f"Clearing existing wrong_predictions directory: {wrong_images_dir}")
        for file in os.listdir(wrong_images_dir):
            file_path = os.path.join(wrong_images_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(wrong_images_dir, exist_ok=True)
    
    print(f"Saving {len(wrong_samples)} wrong prediction images to {wrong_images_dir}...")
    
    # Create a summary file
    summary_file = os.path.join(wrong_images_dir, 'wrong_predictions_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("WRONG PREDICTIONS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        for i, sample in enumerate(wrong_samples):
            true_class = class_names[sample['true_label']]
            pred_class = class_names[sample['predicted_label']]
            confidence = sample['confidence']
            sample_idx = sample['sample_idx']
            
            # Save image
            img = Image.fromarray((sample['image'] * 255).astype(np.uint8))
            img_filename = f"sample_{sample_idx:05d}_true_{true_class}_pred_{pred_class}_conf_{confidence:.3f}.png"
            img_path = os.path.join(wrong_images_dir, img_filename)
            img.save(img_path)
            
            # Write to summary
            f.write(f"Sample {sample_idx:5d}: True={true_class:>4}, Pred={pred_class:>4}, Conf={confidence:.3f}\n")
    
    print(f"Wrong prediction images saved to {wrong_images_dir}")
    print(f"Summary file: {summary_file}")


def create_wrong_predictions_grid(wrong_samples, class_names, output_dir, max_images=64):
    """Create a grid visualization of wrong predictions."""
    
    if not wrong_samples:
        return
    
    # Limit number of images for grid
    samples_to_show = wrong_samples[:max_images]
    
    # Calculate grid dimensions
    n_samples = len(samples_to_show)
    cols = min(8, n_samples)
    rows = (n_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    
    # Handle different array shapes properly
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    else:
        axes = axes.reshape(rows, cols)
    
    for i, sample in enumerate(samples_to_show):
        row = i // cols
        col = i % cols
        
        # Show image
        axes[row, col].imshow(sample['image'])
        
        # Add title with prediction info
        true_class = class_names[sample['true_label']]
        pred_class = class_names[sample['predicted_label']]
        confidence = sample['confidence']
        
        title = f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}"
        axes[row, col].set_title(title, fontsize=8)
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(n_samples, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Wrong Predictions (showing {n_samples} of {len(wrong_samples)} total)', fontsize=14)
    plt.tight_layout()
    
    grid_path = os.path.join(output_dir, 'wrong_predictions_grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Wrong predictions grid saved to {grid_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Chess Piece Classifier')
    parser.add_argument('--model', default='chess_piece_classifier.pth', help='Path to trained model')
    parser.add_argument('--samples', type=int, default=5000, help='Number of evaluation samples')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], 
                       help='Device to use (auto detects best available)')
    parser.add_argument('--output_dir', default='evaluation_results', help='Output directory for results')
    parser.add_argument('--no_plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--save_wrong_images', action='store_true', help='Save images of wrong predictions')
    parser.add_argument('--max_wrong_images', type=int, default=64, help='Maximum number of wrong images to show in grid')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    model, class_names = load_model(args.model, device)
    
    # Create evaluation dataset and dataloader
    print(f"Creating evaluation dataset with {args.samples} samples...")
    eval_dataset = ChessPieceDataset(
        chess_pieces_dir="chess_pieces",
        boards_dir="boards",
        image_size=224,
        samples_per_epoch=args.samples,
        augment=False  # No augmentation for evaluation
    )
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0 if device == 'mps' else 2
    )
    
    # Evaluate model
    results = evaluate_model(
        model, 
        eval_loader, 
        class_names, 
        device, 
        save_wrong_images=args.save_wrong_images,
        wrong_images_dir=args.output_dir
    )
    
    # Calculate metrics
    metrics = calculate_metrics(results, class_names)
    
    # Print report
    print_detailed_report(metrics, class_names)
    
    # Save wrong prediction images if requested
    if args.save_wrong_images and results['wrong_samples']:
        save_wrong_images(results['wrong_samples'], class_names, args.output_dir)
        create_wrong_predictions_grid(results['wrong_samples'], class_names, args.output_dir, args.max_wrong_images)
    
    # Generate plots
    if not args.no_plots:
        plot_confusion_matrix(
            metrics['confusion_matrix'], 
            class_names, 
            os.path.join(args.output_dir, 'confusion_matrix.png')
        )
        
        plot_class_accuracy(
            metrics['class_accuracy'], 
            os.path.join(args.output_dir, 'class_accuracy.png')
        )
        
        plot_confidence_distribution(
            results, 
            os.path.join(args.output_dir, 'confidence_distribution.png')
        )
    
    # Save results
    save_results_to_csv(
        results, 
        metrics, 
        class_names, 
        os.path.join(args.output_dir, 'evaluation_results.xlsx')
    )
    
    print(f"\nEvaluation completed! Results saved to '{args.output_dir}' directory.")


if __name__ == "__main__":
    main() 