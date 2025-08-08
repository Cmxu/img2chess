import os
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure we can import from this directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from chess_piece_classifier import ChessPieceDataset  # noqa: E402


def _tensor_to_image(pixel_values: torch.Tensor) -> np.ndarray:
    """Convert DINOv2-normalized CHW tensor to HWC numpy image in [0, 1]."""
    if pixel_values.ndim != 3 or pixel_values.shape[0] != 3:
        raise ValueError("Expected pixel_values shape (3, H, W)")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = pixel_values * std + mean
    img = img.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return img


def generate_grid(
    output_path: str,
    rows: int = 8,
    cols: int = 8,
    image_size: int = 224,
    augment: bool = True,
    chess_pieces_dir: str = "chess_pieces",
    boards_dir: str = "boards",
) -> None:
    """Generate an rows x cols grid of random images from ChessPieceDataset."""
    num_images = rows * cols

    # Set seeds for reproducibility of the grid content (optional)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    dataset = ChessPieceDataset(
        chess_pieces_dir=chess_pieces_dir,
        boards_dir=boards_dir,
        image_size=image_size,
        samples_per_epoch=num_images,
        augment=augment,
    )

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(num_images):
        r = i // cols
        c = i % cols
        pixel_values, _label = dataset[i]
        img = _tensor_to_image(pixel_values)
        axes[r, c].imshow(img)
        axes[r, c].axis("off")

    plt.tight_layout(pad=0.1)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid to {output_path}")


if __name__ == "__main__":
    # Default output to project root for convenience
    default_output = os.path.join(os.path.dirname(SCRIPT_DIR), "training_grid_8x8.png")
    generate_grid(output_path=default_output) 