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
                 augment: bool = True,
                 model_type: str = "dinov2-small"):
        
        self.chess_pieces_dir = chess_pieces_dir
        self.boards_dir = boards_dir
        self.image_size = image_size
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        self.model_type = model_type
        
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
        
        # Initialize the feature extractor consistent with backbone
        self.feature_extractor = AutoImageProcessor.from_pretrained(
            f"facebook/{model_type}", use_fast=True
        )
        
        # Load cursor sprites (pointer and hand) from local folder
        self.cursors_dir = os.path.join(os.path.dirname(__file__), "cursors")
        self.cursor_images = self._load_cursor_images(self.cursors_dir)
        
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        # 1/13 chance for empty square, 12/13 chance for a chess piece
        if random.random() < (4/13):
            # Empty square
            label = 0
            image = self._generate_empty_square()
        else:
            # Chess piece
            piece_name, image = self._generate_piece_image()
            label = self.class_to_idx[piece_name]
        
        # Apply augmentations if enabled
        if self.augment:
            image = self._apply_augmentations(image, is_piece=(label != 0))
        
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
        
        # Optional: apply a semi-transparent full red overlay over the background
        # when augmentation is enabled (25% probability)
        if self.augment and random.random() < 0.25:
            background = self._apply_full_red_overlay(background)
        
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
        
        # Optional: apply a semi-transparent full red overlay over the background
        # BEFORE placing the piece, when augmentation is enabled (25% probability)
        if self.augment and random.random() < 0.25:
            background = self._apply_full_red_overlay(background)
        
        # Resize piece to fit on the square (with some padding)
        piece_size = int(self.image_size * 0.8)  # 80% of square size
        piece = piece.resize((piece_size, piece_size), Image.Resampling.LANCZOS)
        
        # Center the piece on the background
        x_offset = (self.image_size - piece_size) // 2
        y_offset = (self.image_size - piece_size) // 2
        
        # Composite piece onto background
        background.paste(piece, (x_offset, y_offset), piece)
        
        return piece_name, background.convert("RGB")
    
    def _apply_augmentations(self, image: Image.Image, is_piece: bool = False) -> Image.Image:
        """Apply random augmentations to the image."""
        # Random rotation (-10 to +10 degrees)

        # If no piece is drawn, 10% chance to draw a small brown dot at the center
        if not is_piece and random.random() < 0.10:
            image = self._apply_center_brown_dot(image)

        if random.random() < 0.5:
            angle = random.uniform(-5, 5)
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)
        
        # Random translation (up to 10% of image size)
        if random.random() < 0.5:
            dx = int(random.uniform(-0.10, 0.10) * self.image_size)
            dy = int(random.uniform(-0.10, 0.10) * self.image_size)
            image = F.affine(image, angle=0, translate=[dx, dy], scale=1.0, shear=[0.0,0.0], interpolation=transforms.InterpolationMode.BILINEAR, fill=None)
        
        if random.random() < 0.25:  # gaussian blur
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
        if random.random() < 0.25:  # motion blur (cheap approximation)
            image = image.filter(ImageFilter.BoxBlur(radius=random.randint(1,2)))
        if random.random() < 0.25:  # compression
            from io import BytesIO
            buf = BytesIO(); image.save(buf, format="JPEG", quality=random.randint(35, 85))
            image = Image.open(BytesIO(buf.getvalue())).convert("RGB")
        if random.random() < 0.25:  # downscale-upscale
            s = random.uniform(0.5, 0.9)
            small = image.resize((int(self.image_size*s), int(self.image_size*s)), Image.Resampling.BILINEAR)
            image = small.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        
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
        
        # overlay three semi-transparent orange arrows consistent with a random 8x8 board context
        if random.random() <= 1.0:
            endpoint_prob = 0.05 if not is_piece else 0.10
            image = self._apply_random_board_arrows_overlay(image, num_arrows=4, endpoint_in_square_prob=endpoint_prob)

        # 10% chance to overlay a mouse cursor on top of the piece (never on empty squares)
        if random.random() < 0.10:
            image = self._apply_random_mouse_cursor_overlay(image)
        
        # Uniform random noise ±12 applied with 80% probability
        if random.random() < 0.8:
            image = self._apply_uniform_noise(image, max_delta=20)
        
        
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

    def _apply_random_orange_overlay(self, image: Image.Image) -> Image.Image:
        """Overlay a rotated, semi-transparent orange square with area <= 10% of the image."""
        # Work in RGBA to respect transparency during paste
        base_rgba = image.convert("RGBA")

        # Choose side length so that square area is <= 10% of image area
        # side_fraction in [~0.08, ~0.316] of the image dimension ensures area <= ~10%
        w_min_fraction = 0.25
        w_max_fraction = 0.35
        h_min_fraction = 0.25
        h_max_fraction = 1
        h_side_length = int(random.uniform(h_min_fraction, h_max_fraction) * self.image_size)
        h_side_length = max(1, h_side_length)
        w_side_length = int(random.uniform(w_min_fraction, w_max_fraction) * self.image_size)
        w_side_length = max(1, w_side_length)

        # Random orange color (vary within orange spectrum) and alpha for semi-transparency
        red = random.randint(200, 255)
        green = random.randint(80, 165)
        blue = random.randint(0, 60)
        alpha = random.randint(64, 180)  # 25% - 70% opaque

        square = Image.new("RGBA", (h_side_length, w_side_length), (red, green, blue, alpha))

        # Random rotation of the square
        angle = random.uniform(-45, 45)
        rotated_square = square.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)

        # Random position; allow clipping at edges by constraining top-left
        max_x = max(0, self.image_size - rotated_square.width)
        max_y = max(0, self.image_size - rotated_square.height)
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        # Paste with alpha channel as mask
        base_rgba.paste(rotated_square, (x, y), rotated_square)

        return base_rgba.convert("RGB")

    def _apply_random_board_arrows_overlay(self, image: Image.Image, num_arrows: int = 3, endpoint_in_square_prob: float = 0.10) -> Image.Image:
        """Overlay orange semi-transparent arrows as if drawn on a full 8x8 board, then crop to this square.
        
        The current square is placed at a random (col,row) in an 8x8 grid. For each arrow:
        - With `endpoint_in_square_prob`, exactly one endpoint is the center of the current square.
        - Otherwise, both endpoints are the centers of two randomly chosen squares on the virtual board (may be any squares).
        Arrows may or may not overlap this square; only the overlapping portion is composited.
        """
        s = self.image_size
        board_w = 8 * s
        board_h = 8 * s

        overlay_board = Image.new("RGBA", (board_w, board_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_board, "RGBA")

        # Random location for this square on the virtual board
        sq_col = random.randint(1, 6)
        sq_row = random.randint(1, 6)

        def center_of_square(col: int, row: int) -> Tuple[float, float]:
            x = (col + 0.5) * s
            y = (row + 0.5) * s
            return x, y

        def random_square() -> Tuple[int, int]:
            return random.randint(0, 7), random.randint(0, 7)

        # Prefer end squares aligned horizontally/vertically/diagonally with the start
        def choose_biased_end_square(start_sq: Tuple[int, int], aligned_prob: float = 0.90) -> Tuple[int, int]:
            start_c, start_r = start_sq
            if random.random() < aligned_prob:
                candidates_set = set()
                # same row
                for c in range(8):
                    if c != start_c:
                        candidates_set.add((c, start_r))
                        candidates_set.add((c, start_r))
                # same column
                for r in range(8):
                    if r != start_r:
                        candidates_set.add((start_c, r))
                        candidates_set.add((start_c, r))
                # diagonals
                for delta in range(-7, 8):
                    c1, r1 = start_c + delta, start_r + delta
                    c2, r2 = start_c + delta, start_r - delta
                    if 0 <= c1 < 8 and 0 <= r1 < 8 and (c1, r1) != (start_c, start_r):
                        candidates_set.add((c1, r1))
                    if 0 <= c2 < 8 and 0 <= r2 < 8 and (c2, r2) != (start_c, start_r):
                        candidates_set.add((c2, r2))
                candidates = list(candidates_set)
                if candidates:
                    return random.choice(candidates)
            # fallback: any other square (not the same as start)
            end_sq = random_square()
            tries = 0
            while end_sq == (start_c, start_r) and tries < 8:
                end_sq = random_square()
                tries += 1
            return end_sq

        def draw_arrow(start_xy: Tuple[float, float], end_xy: Tuple[float, float], color: Tuple[int, int, int, int], shaft_width: int, head_length: float, head_width: float) -> None:
            x1, y1 = start_xy
            x2, y2 = end_xy
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length < 1.0:
                return
            ux, uy = dx / length, dy / length
            bx, by = x2 - ux * head_length, y2 - uy * head_length
            px, py = -uy, ux
            left = (bx + px * (head_width / 2.0), by + py * (head_width / 2.0))
            right = (bx - px * (head_width / 2.0), by - py * (head_width / 2.0))
            draw.line([(x1, y1), (bx, by)], fill=color, width=shaft_width)
            draw.polygon([(x2, y2), left, right], fill=color)

        for _ in range(num_arrows):
            # Choose arrow color with probabilities: orange 50%, others share remaining 50%
            if random.random() < 0.8:
                # Randomly modulate orange shade
                r = min(255, max(200, 255 + random.randint(-30, 30)))  # Keep red high for orange
                g = min(200, max(100, 140 + random.randint(-40, 40)))  # Vary green component more
                b = min(50, max(0, 0 + random.randint(-10, 30)))  # Keep blue low for orange
                base_rgb = (r, g, b)  # Modulated orange
            else:
                base_rgb = random.choice([(30, 144, 255), (255, 0, 0), (0, 200, 0)])  # blue, red, green
            alpha = random.randint(200, 250)
            color = (*base_rgb, alpha)

            # Thicker arrows
            shaft_width = max(3, int(random.uniform(0.15, 0.20) * s))
            head_length = random.uniform(0.26, 0.36) * s
            head_width = random.uniform(2.0, 2.5) * shaft_width

            if random.random() < endpoint_in_square_prob:
                in_square_first = random.random() < 0.5
                in_sq = (sq_col, sq_row)
                out_sq = choose_biased_end_square(in_sq)
                start_sq, end_sq = (in_sq, out_sq) if in_square_first else (out_sq, in_sq)
            else:
                start_sq = random_square()
                end_sq = choose_biased_end_square(start_sq)

            start_xy = center_of_square(*start_sq)
            end_xy = center_of_square(*end_sq)

            draw_arrow(start_xy, end_xy, color, shaft_width, head_length, head_width)

        left = sq_col * s
        top = sq_row * s
        tile_overlay = overlay_board.crop((left, top, left + s, top + s))

        base = image.convert("RGBA")
        base.paste(tile_overlay, (0, 0), tile_overlay)
        return base.convert("RGB")

    def _apply_full_red_overlay(self, background: Image.Image) -> Image.Image:
        """Overlay a semi-transparent full-frame colored rectangle over the background.
        
        Color distribution for square highlight: red 50%; orange, blue, and green share remaining 50%.
        Applied prior to rendering the piece so that the piece remains unaffected
        by the overlay (i.e., appears on top). Transparency is randomized.
        """
        if background.mode != "RGBA":
            base = background.convert("RGBA")
        else:
            base = background.copy()
        # Vary alpha for transparency (roughly 10% - 50% opaque)
        alpha = random.randint(25, 128)
        # Choose square highlight color with probabilities: red 50%, others 50%/3 each
        if random.random() < 0.75:
            base_rgb = (255, 0, 0)  # red
        else:
            base_rgb = random.choice([(255, 140, 0), (30, 144, 255), (0, 200, 0)])  # orange, blue, green
        overlay = Image.new("RGBA", base.size, (*base_rgb, alpha))
        # Composite overlay onto the background
        base.paste(overlay, (0, 0), overlay)
        return base
    
    def _apply_uniform_noise(self, image: Image.Image, max_delta: int = 8) -> Image.Image:
        """Add per-pixel uniform noise in [-max_delta, max_delta] to RGB channels."""
        rgb_image = image.convert("RGB")
        img_array = np.array(rgb_image).astype(np.int16)
        noise = np.random.randint(-max_delta, max_delta + 1, size=img_array.shape, dtype=np.int16)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy, mode="RGB")

    def _load_cursor_images(self, cursors_dir: str) -> List[Image.Image]:
        """Load pointer and hand cursor sprites if present.

        Expected filenames: 'pointer.png', 'hand.png' in `cursors_dir`.
        Returns a list of RGBA images; empty if none found.
        """
        loaded: List[Image.Image] = []
        for filename in ("pointer.png", "hand.png"):
            path = os.path.join(cursors_dir, filename)
            if os.path.exists(path):
                try:
                    img = Image.open(path).convert("RGBA")
                    loaded.append(img)
                except Exception as ex:
                    print(f"⚠️  Failed to load cursor sprite '{path}': {ex}")
        if loaded:
            print(f"Found {len(loaded)} cursor sprites in '{cursors_dir}'")
        else:
            print(f"No cursor sprites found in '{cursors_dir}' (optional)")
        return loaded

    def _apply_random_mouse_cursor_overlay(self, image: Image.Image) -> Image.Image:
        """Overlay a mouse cursor sprite (only pointer or hand) onto the image with slight randomness.

        Sprites are loaded from `img2chess/cursors` (pointer.png, hand.png). If none are available,
        this is a no-op.
        """
        s = self.image_size
        if not getattr(self, "cursor_images", None):
            return image

        base = image.convert("RGBA")
        sprite = random.choice(self.cursor_images)

        # Scale sprite so its displayed width is ~18% - 32% of the tile width
        target_w = max(8, int(random.uniform(0.15, 0.25) * s))
        scale = target_w / max(1, sprite.width)
        new_w = max(1, int(sprite.width * scale))
        new_h = max(1, int(sprite.height * scale))
        cursor_img = sprite.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Small random rotation
        angle = random.uniform(-20, 20)
        cursor_img = cursor_img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)

        # Place near image center (so it overlays the piece), with clamping to bounds
        center_x = random.randint(int(0.35 * s), int(0.65 * s))
        center_y = random.randint(int(0.35 * s), int(0.65 * s))
        x = max(0, min(s - cursor_img.width, center_x - cursor_img.width // 2))
        y = max(0, min(s - cursor_img.height, center_y - cursor_img.height // 2))

        base.paste(cursor_img, (x, y), cursor_img)
        return base.convert("RGB")

    def _apply_center_brown_dot(self, image: Image.Image, radius_ratio: float = 0.10) -> Image.Image:
        """Draw a small brown dot at the center of the square.

        The dot radius is radius_ratio * image_size (minimum radius of 2 pixels).
        """
        size = self.image_size
        cx, cy = size // 2, size // 2
        radius = max(2, int(radius_ratio * size))

        base = image.convert("RGBA")
        draw = ImageDraw.Draw(base, "RGBA")
        brown = (165, 105, 45, 255)  # lighter brown color
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=brown)
        return base.convert("RGB")


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
            #nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 96),
            #nn.BatchNorm1d(96),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(96, num_classes)
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(feature_dim, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.SiLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.SiLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(1024, 256),
        #     nn.BatchNorm1d(256),
        #     nn.SiLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(256, num_classes, bias = False)
        # )
        
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
        feature_extractor = AutoImageProcessor.from_pretrained(f"facebook/{self.model_type}", use_fast=True)
        
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
    mps_ok = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    device = torch.device('mps' if mps_ok else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Determine model type from checkpoint (fallback to default)
    model_type = checkpoint.get('model_type', 'dinov2-small')
    
    # Create model
    model = ChessPieceClassifier(num_classes=13, freeze_backbone=True, model_type=model_type)  # Usually freeze during inference
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"✅ Loaded trained chess piece classifier from {model_path}")
    
    return model


def _build_warmup_cosine_scheduler(optimizer: optim.Optimizer, total_epochs: int, warmup_epochs: int):
    """Create a per-epoch LambdaLR that does linear warmup then cosine decay to 0."""
    warmup_epochs = max(0, warmup_epochs)
    total_epochs = max(1, total_epochs)

    def lr_lambda(current_epoch: int):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        # cosine phase
        progress = (current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _set_backbone_requires_grad(model: 'ChessPieceClassifier', requires_grad: bool) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = requires_grad


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs: int = 10,
    device: str = 'cpu',
    two_stage: bool = True,
    stage1_epochs: int = 5,
    head_lr: float = 3e-4,
    backbone_lr: float = 3e-5,
    weight_decay: float = 5e-2,
    warmup_epochs: int = 2,
    label_smoothing: float = 0.05,
):
    """Train the chess piece classifier with AdamW + cosine warmup and optional two-stage fine-tuning."""

    model = model.to(device)
    total_epochs = num_epochs

    # Overall training progress bar across all epochs (both stages)
    overall_pbar = tqdm(total=total_epochs, desc="Overall training", position=0, leave=True, dynamic_ncols=True)

    # Criterion with optional label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing if label_smoothing > 0 else 0.0)

    train_losses: List[float] = []
    train_accuracies: List[float] = []
    val_accuracies: List[float] = []

    current_epoch = 0

    # Stage 1: train head only (if requested)
    stage1_actual_epochs = min(stage1_epochs, total_epochs) if two_stage else total_epochs
    if stage1_actual_epochs > 0:
        _set_backbone_requires_grad(model, False)
        for param in model.classifier.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(model.classifier.parameters(), lr=head_lr, weight_decay=weight_decay)
        scheduler = _build_warmup_cosine_scheduler(optimizer, total_epochs=stage1_actual_epochs, warmup_epochs=min(warmup_epochs, stage1_actual_epochs))

        for local_epoch in range(stage1_actual_epochs):
            model.train()
            running_loss = 0.0
            num_batches = 0
            train_correct = 0
            train_total = 0

            progress_bar = tqdm(train_loader, desc=f"[Stage 1] Epoch {current_epoch+1}/{total_epochs}", leave=False)
            for batch_idx, (images, labels) in enumerate(progress_bar):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_batches += 1

                with torch.no_grad():
                    _, predicted = torch.max(outputs.detach(), 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            avg_loss = running_loss / max(1, num_batches)
            train_losses.append(avg_loss)
            train_accuracy = 100.0 * train_correct / max(1, train_total)
            train_accuracies.append(train_accuracy)

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f"[Val S1] Epoch {current_epoch+1}/{total_epochs}", leave=False)
                for images, labels in val_bar:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = 100.0 * correct / max(1, total)
            val_accuracies.append(val_accuracy)

            print(f'[Stage 1] Epoch {current_epoch+1}/{total_epochs}, Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')

            # Update overall progress bar
            display_epoch = current_epoch + 1
            overall_pbar.set_postfix({"epoch": f"{display_epoch}/{total_epochs}", "loss": f"{avg_loss:.4f}", "train": f"{train_accuracy:.2f}%", "val": f"{val_accuracy:.2f}%"})
            overall_pbar.update(1)

            scheduler.step()
            current_epoch += 1

    # Stage 2: unfreeze backbone and train end-to-end
    remaining_epochs = total_epochs - current_epoch
    if remaining_epochs > 0:
        _set_backbone_requires_grad(model, True)
        for param in model.classifier.parameters():
            param.requires_grad = True

        # Parameter groups with different LRs
        optimizer = optim.AdamW([
            { 'params': model.backbone.parameters(), 'lr': backbone_lr },
            { 'params': model.classifier.parameters(), 'lr': head_lr },
        ], weight_decay=weight_decay)

        scheduler = _build_warmup_cosine_scheduler(optimizer, total_epochs=remaining_epochs, warmup_epochs=min(warmup_epochs, remaining_epochs))

        for local_epoch in range(remaining_epochs):
            model.train()
            running_loss = 0.0
            num_batches = 0
            train_correct = 0
            train_total = 0

            progress_bar = tqdm(train_loader, desc=f"[Stage 2] Epoch {current_epoch+1}/{total_epochs}", leave=False)
            for batch_idx, (images, labels) in enumerate(progress_bar):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                # Optional: gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                running_loss += loss.item()
                num_batches += 1

                with torch.no_grad():
                    _, predicted = torch.max(outputs.detach(), 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                # Update progress bar with both group LRs
                group_lrs = [pg['lr'] for pg in optimizer.param_groups]
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr_head=f"{group_lrs[1]:.2e}", lr_bb=f"{group_lrs[0]:.2e}")

            avg_loss = running_loss / max(1, num_batches)
            train_losses.append(avg_loss)
            train_accuracy = 100.0 * train_correct / max(1, train_total)
            train_accuracies.append(train_accuracy)

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f"[Val S2] Epoch {current_epoch+1}/{total_epochs}", leave=False)
                for images, labels in val_bar:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = 100.0 * correct / max(1, total)
            val_accuracies.append(val_accuracy)

            print(f'[Stage 2] Epoch {current_epoch+1}/{total_epochs}, Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')

            # Update overall progress bar
            display_epoch = current_epoch + 1
            overall_pbar.set_postfix({"epoch": f"{display_epoch}/{total_epochs}", "loss": f"{avg_loss:.4f}", "train": f"{train_accuracy:.2f}%", "val": f"{val_accuracy:.2f}%"})
            overall_pbar.update(1)

            scheduler.step()
            current_epoch += 1

    # Close overall progress bar before returning
    overall_pbar.close()

    return train_losses, train_accuracies, val_accuracies


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
    mps_ok = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    device = torch.device('mps' if mps_ok else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    if mps_ok:
        print(f'Using device: {device} (Apple Silicon GPU)')
    elif torch.cuda.is_available():
        print(f'Using device: {device} (NVIDIA GPU)')
    else:
        print(f'Using device: {device} (CPU)')
        print('⚠️  Consider using MPS (Apple Silicon) or CUDA (NVIDIA) for faster training')
    
    # Create datasets
    print("Creating datasets...")
    model_type = "dinov2-small"

    train_dataset = ChessPieceDataset(
        chess_pieces_dir="chess_pieces",
        boards_dir="boards",
        image_size=224,  # DINOv2 uses 224x224
        samples_per_epoch=8000,
        augment=True,
        model_type=model_type
    )
    
    val_dataset = ChessPieceDataset(
        chess_pieces_dir="chess_pieces",
        boards_dir="boards",
        image_size=224,  # DINOv2 uses 224x224
        samples_per_epoch=500,
        augment=False,
        model_type=model_type
    )
    
    # Create data loaders (reduce workers for MPS compatibility)
    num_workers = 0 if device.type == 'mps' else 4
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=num_workers)
    
    # Create model
    print("Creating model...")
    model = ChessPieceClassifier(num_classes=13, freeze_backbone=False, model_type=model_type)
    
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Train model
    print("Starting training...")
    train_losses, train_accuracies, val_accuracies = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=300,
        device=device,
        two_stage=True,
        stage1_epochs=5,
        head_lr=3e-4,
        backbone_lr=3e-5,
        weight_decay=5e-2,
        warmup_epochs=2,
        label_smoothing=0.05,
    )
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': train_dataset.classes,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'model_type': model_type
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
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    # plt.show()  # Commented out to prevent freezing


if __name__ == "__main__":
    main() 