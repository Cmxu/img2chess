import os
import random
import glob
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt


class ChessPieceDataset(Dataset):
    """
    PyTorch Dataset for generating chess piece classification data on-the-fly.
    
    13 classes:
    - 12 chess pieces: w/b + p/r/n/b/q/k (white/black + pawn/rook/knight/bishop/queen/king)
    - 1 empty square
    """
    
    def __init__(self, 
                 chess_pieces_dir: str = "chess_pieces",
                 boards_dir: str = "boards", 
                 image_size: int = 224,  # DINOv2 uses 224x224
                 samples_per_epoch: int = 10000,
                 augment: bool = True):
        
        self.chess_pieces_dir = chess_pieces_dir
        self.boards_dir = boards_dir
        self.image_size = image_size
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        
        # Define the 13 classes
        self.classes = [
            'empty',  # 0
            'wp', 'wr', 'wn', 'wb', 'wq', 'wk',  # 1-6: white pieces
            'bp', 'br', 'bn', 'bb', 'bq', 'bk'   # 7-12: black pieces
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Get all piece style directories
        self.piece_styles = [d for d in os.listdir(chess_pieces_dir) 
                           if os.path.isdir(os.path.join(chess_pieces_dir, d))]
        
        # Get all board backgrounds
        self.board_files = glob.glob(os.path.join(boards_dir, "*.png"))
        
        print(f"Found {len(self.piece_styles)} piece styles")
        print(f"Found {len(self.board_files)} board backgrounds")
        
        # Initialize the feature extractor
        self.feature_extractor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-base", use_fast=True
        )
        
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        # 1/13 chance for empty square, 12/13 chance for a chess piece
        if random.random() < (1/13):
            # Empty square
            label = 0
            image = self._generate_empty_square()
        else:
            # Chess piece
            piece_name, image = self._generate_piece_image()
            label = self.class_to_idx[piece_name]
        
        # Apply augmentations if enabled
        if self.augment:
            image = self._apply_augmentations(image)
        
        # Convert to tensor and normalize for DINOv2
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # Remove batch dimension
        
        return pixel_values, label
    
    def _generate_empty_square(self) -> Image.Image:
        """Generate an empty square with random board background."""
        # Pick random board background
        board_file = random.choice(self.board_files)
        background = Image.open(board_file).convert("RGBA")
        
        # Resize to target size
        background = background.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        return background.convert("RGB")
    
    def _generate_piece_image(self) -> Tuple[str, Image.Image]:
        """Generate a chess piece image with random background."""
        # Pick random piece style and piece
        style = random.choice(self.piece_styles)
        piece_name = random.choice(self.classes[1:])  # Skip 'empty'
        
        # Load piece image
        piece_path = os.path.join(self.chess_pieces_dir, style, f"{piece_name}.png")
        if not os.path.exists(piece_path):
            # Fallback to classic style if piece doesn't exist
            piece_path = os.path.join(self.chess_pieces_dir, "classic", f"{piece_name}.png")
        
        piece = Image.open(piece_path).convert("RGBA")
        
        # Pick random board background
        board_file = random.choice(self.board_files)
        background = Image.open(board_file).convert("RGBA")
        
        # Resize background to target size
        background = background.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        # Resize piece to fit on the square (with some padding)
        piece_size = int(self.image_size * 0.8)  # 80% of square size
        piece = piece.resize((piece_size, piece_size), Image.Resampling.LANCZOS)
        
        # Center the piece on the background
        x_offset = (self.image_size - piece_size) // 2
        y_offset = (self.image_size - piece_size) // 2
        
        # Composite piece onto background
        background.paste(piece, (x_offset, y_offset), piece)
        
        return piece_name, background.convert("RGB")
    
    def _apply_augmentations(self, image: Image.Image) -> Image.Image:
        """Apply random augmentations to the image."""
        # Random rotation (-10 to +10 degrees)
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)
        
        # Random translation (up to 10% of image size)
        if random.random() < 0.5:
            max_translate = int(0.1 * self.image_size)
            dx = random.randint(-max_translate, max_translate)
            dy = random.randint(-max_translate, max_translate)
            image = ImageOps.expand(image, border=(abs(dx), abs(dy), abs(dx), abs(dy)), fill='white')
            image = image.crop((abs(dx), abs(dy), image.width - abs(dx), image.height - abs(dy)))
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        # Random brightness (0.8 to 1.2)
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Random contrast (0.8 to 1.2)
        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Random saturation (0.8 to 1.2)
        if random.random() < 0.5:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Random erasing to help with numbers/letters on squares
        if random.random() < 0.3:  # 30% chance of applying random erasing
            image = self._apply_random_erasing(image)
        
        return image
    
    def _apply_random_erasing(self, image: Image.Image) -> Image.Image:
        """Apply random erasing by masking out rectangular regions."""
        # Convert to numpy for easier manipulation
        img_array = np.array(image)
        
        # Apply 1-3 random erasing patches
        num_patches = random.randint(1, 3)
        
        for _ in range(num_patches):
            # Random patch size (10-30% of image size)
            patch_size_w = random.randint(int(self.image_size * 0.1), int(self.image_size * 0.3))
            patch_size_h = random.randint(int(self.image_size * 0.1), int(self.image_size * 0.3))
            
            # Random position
            x = random.randint(0, self.image_size - patch_size_w)
            y = random.randint(0, self.image_size - patch_size_h)
            
            # Random fill value (gray to white range)
            fill_value = random.randint(200, 255)
            
            # Apply the patch
            img_array[y:y+patch_size_h, x:x+patch_size_w] = fill_value
        
        return Image.fromarray(img_array)


class ChessPieceClassifier(nn.Module):
    """
    Chess piece classifier using DINOv2 feature extractor + 3-layer linear head.
    """
    
    def __init__(self, num_classes: int = 13, freeze_backbone: bool = True):
        super(ChessPieceClassifier, self).__init__()
        
        # Load DINOv2 backbone
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get feature dimension from backbone
        # DINOv2-base outputs features with dimension 768
        feature_dim = 768  # Fixed dimension for DINOv2-base
        
        # 3-layer linear head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, pixel_values):
        # Extract features using DINOv2
        outputs = self.backbone(pixel_values)
        
        # Get CLS token features (first token in sequence)
        features = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, feature_dim)
        
        # Apply classifier
        logits = self.classifier(features)
        
        return logits
    
    def classify_piece(self, square_image: np.ndarray) -> str:
        """
        Classify a single chess piece from a square image.
        
        Args:
            square_image: numpy array representing the square image
            
        Returns:
            String representation of the piece (e.g., 'wp', 'bk', 'empty')
        """
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert numpy array to PIL Image
        from PIL import Image
        if square_image.dtype != np.uint8:
            square_image = (square_image * 255).astype(np.uint8)
        
        image = Image.fromarray(square_image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to expected input size (224x224 for DINOv2)
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Get feature extractor
        from transformers import AutoImageProcessor
        feature_extractor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
        
        # Preprocess
        inputs = feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        
        # Set model to evaluation mode
        self.eval()
        self = self.to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self(pixel_values)
            _, predicted = torch.max(outputs, 1)
            predicted_idx = predicted.item()
        
        # Map back to class name
        classes = [
            'empty',  # 0
            'wp', 'wr', 'wn', 'wb', 'wq', 'wk',  # 1-6: white pieces
            'bp', 'br', 'bn', 'bb', 'bq', 'bk'   # 7-12: black pieces
        ]
        
        return classes[predicted_idx]


def load_trained_model(model_path: str = "chess_piece_classifier.pth") -> ChessPieceClassifier:
    """
    Load a trained chess piece classifier model.
    
    Args:
        model_path: Path to the saved model checkpoint
        
    Returns:
        Loaded ChessPieceClassifier model
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    model = ChessPieceClassifier(num_classes=13, freeze_backbone=True)  # Usually freeze during inference
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"✅ Loaded trained chess piece classifier from {model_path}")
    
    return model


def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    """Train the chess piece classifier."""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    train_losses = []
    val_accuracies = []
    
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
            
            if batch_idx % 100 == 0:
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
        
        scheduler.step()
    
    return train_losses, val_accuracies


def visualize_samples(dataset, num_samples=8):
    """Visualize some samples from the dataset."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        pixel_values, label = dataset[i]
        
        # Convert tensor back to image for visualization
        # Denormalize (DINOv2 uses ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        image = pixel_values * std.unsqueeze(-1).unsqueeze(-1) + mean.unsqueeze(-1).unsqueeze(-1)
        image = torch.clamp(image, 0, 1)
        image = image.permute(1, 2, 0).numpy()
        
        axes[i].imshow(image)
        axes[i].set_title(f'Class: {dataset.classes[label]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('chess_piece_samples.png', dpi=150, bbox_inches='tight')
    # plt.show()  # Commented out to prevent freezing


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Check for best available device (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f'Using device: {device} (Apple Silicon GPU)')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using device: {device} (NVIDIA GPU)')
    else:
        device = torch.device('cpu')
        print(f'Using device: {device} (CPU)')
        print('⚠️  Consider using MPS (Apple Silicon) or CUDA (NVIDIA) for faster training')
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = ChessPieceDataset(
        chess_pieces_dir="chess_pieces",
        boards_dir="boards",
        image_size=224,  # DINOv2 uses 224x224
        samples_per_epoch=8000,
        augment=True
    )
    
    val_dataset = ChessPieceDataset(
        chess_pieces_dir="chess_pieces",
        boards_dir="boards",
        image_size=224,  # DINOv2 uses 224x224
        samples_per_epoch=2000,
        augment=False
    )
    
    # Create data loaders (reduce workers for MPS compatibility)
    num_workers = 0 if device.type == 'mps' else 2
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=num_workers)
    
    # Create model
    print("Creating model...")
    model = ChessPieceClassifier(num_classes=13, freeze_backbone=False)
    
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Train model
    print("Starting training...")
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, 
        num_epochs=15, device=device
    )
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': train_dataset.classes,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies
    }, 'chess_piece_classifier.pth')
    
    print("Training completed and model saved!")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    # plt.show()  # Commented out to prevent freezing


if __name__ == "__main__":
    main() 