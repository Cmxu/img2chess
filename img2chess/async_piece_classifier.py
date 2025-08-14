#!/usr/bin/env python3
import os
import threading
import time
from queue import Queue, Empty
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor

# Local imports (model defined alongside this module)
from .chess_piece_classifier import ChessPieceClassifier


class AsyncPieceClassifierService:
    """
    Asynchronous batching service for chess piece classification.

    - Loads model once on the best available device (CUDA > MPS > CPU)
    - Runs a background worker thread that aggregates incoming images into batches
    - Returns (label, confidence) for each submitted image
    """

    DEFAULT_CLASSES = [
        'empty',
        'wp', 'wr', 'wn', 'wb', 'wq', 'wk',
        'bp', 'br', 'bn', 'bb', 'bq', 'bk'
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        max_batch_size: int = 256,
        batch_timeout_s: float = 0.010,
    ) -> None:
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), 'chess_piece_classifier.pth')
        self.device = self._select_device(device)
        self.max_batch_size = max_batch_size
        self.batch_timeout_s = batch_timeout_s

        # Load model once
        self.model, self.class_names = self._load_model(self.model_path, self.device)
        self.feature_extractor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', use_fast=True)

        # Worker infra
        self._queue: "Queue[Dict[str, Any]]" = Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, name='AsyncPieceClassifierWorker', daemon=True)
        self._worker.start()

    def _select_device(self, device: Optional[str]) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def _load_model(self, model_path: str, device: torch.device):
        checkpoint = torch.load(model_path, map_location=device)
        model = ChessPieceClassifier(num_classes=13, freeze_backbone=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        class_names = checkpoint.get('class_names', self.DEFAULT_CLASSES)
        return model, class_names

    def submit(self, square_image_bgr: np.ndarray) -> "torch.futures.Future[Tuple[str, float]]":
        """
        Submit a single square image (BGR, numpy array) for classification.
        Returns a torch Future resolving to (label, confidence).
        """
        fut: torch.futures.Future = torch.futures.Future()
        self._queue.put({'image': square_image_bgr, 'future': fut})
        return fut

    def submit_many(self, square_images_bgr: List[np.ndarray]) -> List["torch.futures.Future[Tuple[str, float]]"]:
        futures: List[torch.futures.Future] = []
        for img in square_images_bgr:
            futures.append(self.submit(img))
        return futures

    def classify_batch_sync(self, square_images_bgr: List[np.ndarray]) -> List[Tuple[str, float]]:
        futures = self.submit_many(square_images_bgr)
        return [fut.wait() for fut in futures]

    def shutdown(self, wait: bool = True) -> None:
        self._stop_event.set()
        if wait and self._worker.is_alive():
            self._worker.join(timeout=2.0)

    # ---------------- Internal -----------------

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            batch_items: List[Dict[str, Any]] = []

            try:
                first = self._queue.get(timeout=0.05)
            except Empty:
                continue

            batch_items.append(first)
            start_time = time.time()

            # Drain additional items until timeout or max batch size
            while len(batch_items) < self.max_batch_size:
                remaining = self.batch_timeout_s - (time.time() - start_time)
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=max(0.0, remaining))
                    batch_items.append(item)
                except Empty:
                    break

            try:
                self._process_batch(batch_items)
            except Exception as e:
                # Fail all futures on unexpected error
                for item in batch_items:
                    fut = item['future']
                    if not fut.done():
                        fut.set_exception(e)

    def _process_batch(self, batch_items: List[Dict[str, Any]]) -> None:
        # Convert BGR numpy arrays to RGB PIL Images
        pil_images: List[Image.Image] = []
        for item in batch_items:
            img_bgr = item['image']
            if img_bgr is None:
                pil_images.append(Image.new('RGB', (224, 224), (0, 0, 0)))
                continue
            if img_bgr.dtype != np.uint8:
                img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)
            # BGR -> RGB
            img_rgb = img_bgr[:, :, ::-1] if img_bgr.ndim == 3 else np.stack([img_bgr]*3, axis=-1)
            pil = Image.fromarray(img_rgb).convert('RGB').resize((224, 224), Image.Resampling.LANCZOS)
            pil_images.append(pil)

        # Preprocess as batch
        inputs = self.feature_extractor(images=pil_images, return_tensors='pt')
        pixel_values = inputs['pixel_values'].to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values)
            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

        for i, item in enumerate(batch_items):
            label_idx = preds[i].item()
            conf = confs[i].item()
            label = self.class_names[label_idx]
            fut: torch.futures.Future = item['future']
            if not fut.done():
                fut.set_result((label, conf))


# Simple singleton helper
_singleton_lock = threading.Lock()
_singleton_instance: Optional[AsyncPieceClassifierService] = None


def get_async_classifier(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    max_batch_size: int = 256,
    batch_timeout_s: float = 0.010,
) -> AsyncPieceClassifierService:
    global _singleton_instance
    if _singleton_instance is not None:
        return _singleton_instance
    with _singleton_lock:
        if _singleton_instance is None:
            _singleton_instance = AsyncPieceClassifierService(
                model_path=model_path,
                device=device,
                max_batch_size=max_batch_size,
                batch_timeout_s=batch_timeout_s,
            )
    return _singleton_instance
