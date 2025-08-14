#!/usr/bin/env python3
import os
import random
import glob
from typing import Tuple, List, Optional
import math
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageDraw
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt


class ChessPieceClassifier(nn.Module):
    """
    Chess piece classifier using DINOv2 feature extractor + 3-layer linear head.
    """
    
    def __init__(self, num_classes: int = 13, freeze_backbone: bool = True, model_type: str = "dinov2-small"):
        super(ChessPieceClassifier, self).__init__()
        
        # Load DINOv2 backbone
        self.backbone = AutoModel.from_pretrained(f"facebook/{model_type}")
        self.model_type = model_type
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get feature dimension from backbone
        feature_dim = self.backbone.config.hidden_size
        
        # 3-layer linear head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 96),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(96, num_classes)
        )
        
    def forward(self, pixel_values):
        # Extract features using DINOv2
        outputs = self.backbone(pixel_values)
        
        # Get CLS token features (first token in sequence)
        features = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, feature_dim)
        
        # Apply classifier
        logits = self.classifier(features)
        
        return logits
