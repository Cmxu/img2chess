"""
Original chess piece classifier architecture to match saved model.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class ChessPieceClassifierOriginal(nn.Module):
    """
    Original chess piece classifier architecture: 768 -> 512 -> 256 -> 13
    """
    
    def __init__(self, num_classes: int = 13, freeze_backbone: bool = True):
        super(ChessPieceClassifierOriginal, self).__init__()
        
        # Load DINOv2 backbone
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get feature dimension from backbone
        # DINOv2-base outputs features with dimension 768
        feature_dim = 768  # Fixed dimension for DINOv2-base
        
        # Original 3-layer linear head with larger dimensions
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, pixel_values):
        # Extract features using DINOv2
        outputs = self.backbone(pixel_values)
        
        # Get CLS token features (first token in sequence)
        features = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, feature_dim)
        
        # Apply classifier
        logits = self.classifier(features)
        
        return logits