#!/usr/bin/env python3
"""
Quick training script for testing the chess piece classifier.
Uses fewer epochs and smaller dataset for faster testing.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from chess_piece_classifier import ChessPieceDataset, ChessPieceClassifier


def quick_train():
    """Quick training with minimal epochs for testing."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Check for best available device (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)
    mps_ok = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if mps_ok:
        device = torch.device('mps')
        print(f'Using device: {device} (Apple Silicon GPU)')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using device: {device} (NVIDIA GPU)')
    else:
        device = torch.device('cpu')
        print(f'Using device: {device} (CPU)')
        print('⚠️  Consider using MPS (Apple Silicon) or CUDA (NVIDIA) for faster training')
    
    # Create smaller datasets for quick testing
    print("Creating datasets...")
    train_dataset = ChessPieceDataset(
        chess_pieces_dir="chess_pieces",
        boards_dir="boards",
        image_size=224,  # DINOv2 uses 224x224
        samples_per_epoch=1000,  # Smaller for quick testing
        augment=True
    )
    
    val_dataset = ChessPieceDataset(
        chess_pieces_dir="chess_pieces",
        boards_dir="boards",
        image_size=224,  # DINOv2 uses 224x224
        samples_per_epoch=200,   # Smaller for quick testing
        augment=False
    )
    
    # Create data loaders (reduce workers for MPS compatibility)
    num_workers = 0 if device.type == 'mps' else 2  
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=num_workers)
    
    # Create model
    print("Creating model...")
    model = ChessPieceClassifier(num_classes=13, freeze_backbone=True)
    model = model.to(device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    # Quick training (just 3 epochs)
    num_epochs = 3
    train_losses = []
    val_accuracies = []
    
    print("Starting quick training...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = running_loss / num_batches
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': train_dataset.classes,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies
    }, 'chess_piece_classifier_quick.pth')
    
    print(f"Quick training completed!")
    print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")
    
    # Test with a few samples
    print("\nTesting with sample predictions...")
    model.eval()
    
    # Get a few test samples
    test_samples = []
    for i in range(5):
        pixel_values, label = val_dataset[i]
        test_samples.append((pixel_values.unsqueeze(0).to(device), label))
    
    with torch.no_grad():
        for i, (pixel_values, true_label) in enumerate(test_samples):
            outputs = model(pixel_values)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.softmax(outputs, dim=1).max().item()
            
            true_class = train_dataset.classes[true_label]
            pred_class = train_dataset.classes[predicted.item()]
            
            print(f"Sample {i+1}: True={true_class}, Predicted={pred_class}, Confidence={confidence:.3f}")
    
    return model, train_dataset.classes


if __name__ == "__main__":
    model, classes = quick_train() 