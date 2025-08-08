import os
import cv2
import json
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import time
import concurrent.futures

from img2chess.img2chess.clean_edge_detector import CleanEdgeBasedDetector
from img2chess.img2chess.square_extractor import SquareExtractor
from img2chess.async_piece_classifier import get_async_classifier, AsyncPieceClassifierService


def boards_similar(prev_board: np.ndarray, curr_board: np.ndarray, similarity_threshold: float = 0.75) -> bool:
    if prev_board is None or curr_board is None:
        return False
    if prev_board.shape != curr_board.shape:
        return False
    # Compute absolute difference and fraction of nearly-identical pixels
    diff = cv2.absdiff(prev_board, curr_board)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Consider a pixel unchanged if below small epsilon
    unchanged_mask = (gray <= 5)
    ratio = float(np.count_nonzero(unchanged_mask)) / unchanged_mask.size
    return ratio >= similarity_threshold


def compute_board_similarity(prev_board: Optional[np.ndarray], curr_board: Optional[np.ndarray]) -> Optional[float]:
    """
    Return similarity ratio in [0,1] between two boards (higher = more similar).
    Returns None if either is missing or shapes mismatch.
    """
    if prev_board is None or curr_board is None:
        return None
    if prev_board.shape != curr_board.shape:
        return None
    diff = cv2.absdiff(prev_board, curr_board)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    unchanged_mask = (gray <= 5)
    ratio = float(np.count_nonzero(unchanged_mask)) / unchanged_mask.size
    return ratio


def compute_square_change_mask(prev_board: Optional[np.ndarray], curr_board: Optional[np.ndarray], epsilon: int = 15, changed_threshold: int = 0.08) -> Tuple[Optional[List[List[int]]], Optional[float]]:
    """
    Compute an 8x8 mask of changed (1) vs unchanged (0) squares and return the unchanged ratio.
    A square is considered unchanged only if ALL its pixels differ by <= epsilon.
    Returns (None, None) if inputs are invalid.
    """
    if prev_board is None or curr_board is None:
        return None, None
    if prev_board.shape != curr_board.shape:
        return None, None

    diff = cv2.absdiff(prev_board, curr_board)
    #gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff = np.mean(diff, axis=2)
    changed_pixel_mask = (diff > epsilon)

    size = prev_board.shape[0]
    square = size // 8
    mask: List[List[int]] = []
    unchanged_squares = 0
    for r in range(8):
        row_vals: List[int] = []
        for c in range(8):
            y0 = r * square
            y1 = (r + 1) * square
            x0 = c * square
            x1 = (c + 1) * square
            region = changed_pixel_mask[y0:y1, x0:x1]
            is_unchanged = (region.sum() <= changed_threshold * region.size)#bool(np.all(region))
            row_vals.append(0 if is_unchanged else 1)
            if is_unchanged:
                unchanged_squares += 1
        mask.append(row_vals)

    unchanged_ratio = unchanged_squares / 64.0
    return mask, unchanged_ratio


def classify_board_async(
    board_img: np.ndarray,
    min_confidence: float = 0.95,
    classifier: Optional[AsyncPieceClassifierService] = None,
    squares_to_classify: Optional[List[str]] = None,
    previous_results: Optional[Dict[str, Tuple[str, float]]] = None,
) -> Tuple[Dict[str, Tuple[str, float]], bool]:
    extractor = SquareExtractor(square_size=224)
    squares = extractor.extract_squares(board_img)

    # Determine keys to submit
    classifier = classifier or get_async_classifier()
    square_keys_all = sorted(squares.keys())

    keys_to_submit: List[str]
    if squares_to_classify is None or previous_results is None:
        # Full classification
        keys_to_submit = square_keys_all
        base_results: Dict[str, Tuple[str, float]] = {}
    else:
        # Partial classification: changed squares + any missing from previous_results
        missing = [k for k in square_keys_all if k not in previous_results]
        keys_to_submit = sorted(set([k for k in squares_to_classify if k in squares]) | set(missing))
        # Start from previous results for all squares we have
        base_results = {k: v for k, v in (previous_results or {}).items() if k in squares}

    # Submit only required squares
    futures: List[Tuple[str, object]] = []
    for k in keys_to_submit:
        futures.append((k, classifier.submit(squares[k])))

    # Gather results
    for k, fut in futures:
        label, conf = fut.wait()
        base_results[k] = (label, conf)

    # Ensure we have all squares in final results; if any still missing, classify them now
    missing_final = [k for k in square_keys_all if k not in base_results]
    for k in missing_final:
        label, conf = classifier.submit(squares[k]).wait()
        base_results[k] = (label, conf)

    # Compute all_high over all 64 squares
    all_high = True
    for k in square_keys_all:
        _, conf = base_results[k]
        if conf < min_confidence:
            all_high = False
            break

    return base_results, all_high


def _label_to_piece_char(label: str) -> Optional[str]:
    mapping = {
        'wp': 'P', 'wr': 'R', 'wn': 'N', 'wb': 'B', 'wq': 'Q', 'wk': 'K',
        'bp': 'p', 'br': 'r', 'bn': 'n', 'bb': 'b', 'bq': 'q', 'bk': 'k',
    }
    return mapping.get(label)


_PIECE_IMG_CACHE: Dict[Tuple[str, int], np.ndarray] = {}


def _load_piece_image(label: str, target_size: int) -> Optional[np.ndarray]:
    """
    Load a piece image (BGRA) from img2chess/chess_pieces/modern and resize to target_size.
    Returns BGRA image or None if not found.
    """
    global _PIECE_IMG_CACHE
    key = (label, target_size)
    if key in _PIECE_IMG_CACHE:
        return _PIECE_IMG_CACHE[key]

    base_dir = os.path.join(os.path.dirname(__file__), 'chess_pieces', 'modern')
    piece_path = os.path.join(base_dir, f'{label}.png')
    if not os.path.exists(piece_path):
        return None
    img = cv2.imread(piece_path, cv2.IMREAD_UNCHANGED)  # BGRA
    if img is None:
        return None
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    _PIECE_IMG_CACHE[key] = img
    return img


def _overlay_bgra(dst_bgr: np.ndarray, src_bgra: np.ndarray, x: int, y: int) -> None:
    """
    Alpha blend src_bgra onto dst_bgr at top-left (x, y). Modifies dst_bgr in place.
    Clips if partially outside bounds.
    """
    h, w = dst_bgr.shape[:2]
    sh, sw = src_bgra.shape[:2]

    if x >= w or y >= h:
        return

    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + sw, w)
    y1 = min(y + sh, h)

    if x1 <= x0 or y1 <= y0:
        return

    roi = dst_bgr[y0:y1, x0:x1]
    sx0 = x0 - x
    sy0 = y0 - y
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)
    src_roi = src_bgra[sy0:sy1, sx0:sx1]

    if src_roi.shape[2] == 4:
        src_rgb = src_roi[:, :, :3].astype(np.float32)
        alpha = (src_roi[:, :, 3:4].astype(np.float32) / 255.0)
        dst_rgb = roi.astype(np.float32)
        out = alpha * src_rgb + (1.0 - alpha) * dst_rgb
        roi[:] = out.astype(np.uint8)
    else:
        roi[:] = src_roi


def _render_predicted_board_image(results: Dict[str, Tuple[str, float]], size: int = 512, show_confidences: bool = False) -> np.ndarray:
    """
    Render the predicted board using piece assets from img2chess/chess_pieces/modern.
    Falls back to letter rendering if an asset is missing.
    Optionally overlays per-square confidence scores.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Colors similar to chess.com green theme
    light = (210, 238, 238)  # BGR
    dark = (86, 150, 118)
    square = size // 8

    # Draw squares
    for r in range(8):
        for c in range(8):
            y0 = r * square
            x0 = c * square
            color = light if (r + c) % 2 == 0 else dark
            cv2.rectangle(img, (x0, y0), (x0 + square, y0 + square), color, thickness=-1)

    # Draw pieces
    font = cv2.FONT_HERSHEY_SIMPLEX
    for rank in range(8, 0, -1):  # 8..1 (top to bottom)
        r = 8 - rank  # row index 0..7
        for file_idx, file_char in enumerate('abcdefgh'):
            key = f"{file_char}{rank}"
            label_conf = results.get(key)
            if not label_conf:
                continue
            label, conf = label_conf
            if label == 'empty':
                # Still may overlay confidence if requested
                pass

            # Try asset-based rendering first
            piece_size = int(square * 0.9)
            piece_img = _load_piece_image(label, piece_size)
            x0 = file_idx * square + (square - piece_size) // 2
            y0 = r * square + (square - piece_size) // 2
            if piece_img is not None and label != 'empty':
                _overlay_bgra(img, piece_img, x0, y0)
                
            # Fallback to letter rendering
            if label != 'empty':
                piece_char = _label_to_piece_char(label)
                if piece_char:
                    is_white = piece_char.isupper()
                    text_color = (255, 255, 255) if is_white else (0, 0, 0)
                    text = piece_char
                    text_scale = max(0.6, square / 90.0)
                    thickness = max(1, square // 32)
                    (text_w, text_h), baseline = cv2.getTextSize(text, font, text_scale, thickness)
                    x = file_idx * square + (square - text_w) // 2
                    y = r * square + (square + text_h) // 2
                    border_color = (0, 0, 0) if is_white else (255, 255, 255)
                    for dx in (-1, 1):
                        for dy in (-1, 1):
                            cv2.putText(img, text, (x + dx, y + dy), font, text_scale, border_color, thickness, lineType=cv2.LINE_AA)
                    cv2.putText(img, text, (x, y), font, text_scale, text_color, thickness, lineType=cv2.LINE_AA)

            # Optional confidence overlay
            if show_confidences:
                conf_text = f"{conf:.2f}"
                conf_scale = max(0.4, square / 140.0)
                conf_thickness = max(1, square // 64)
                (cw, ch), _ = cv2.getTextSize(conf_text, font, conf_scale, conf_thickness)
                cx = file_idx * square + square - cw - 4
                cy = r * square + square - 4
                # Draw border for legibility
                for dx in (-1, 1):
                    for dy in (-1, 1):
                        cv2.putText(img, conf_text, (cx + dx, cy + dy), font, conf_scale, (0, 0, 0), conf_thickness, lineType=cv2.LINE_AA)
                cv2.putText(img, conf_text, (cx, cy), font, conf_scale, (255, 255, 255), conf_thickness, lineType=cv2.LINE_AA)

    return img


def process_video_folder(
    frames_dir: str,
    detector_config_path: str,
    output_dir: str,
    similarity_threshold: float = 0.75,
    min_square_confidence: float = 0.95,
    classifier: Optional[AsyncPieceClassifierService] = None,
) -> Dict[str, Dict]:
    """
    Process a single video folder of frames.

    - Detect board once; reuse corners when board remains similar across frames
    - For each extracted board, split into 64 squares and classify via async service
    - Save boards with all squares above confidence

    Returns per-frame results dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    detector = CleanEdgeBasedDetector(detector_config_path)

    # Prepare frame list (support .jpg and .png)
    frame_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.png'))]
    frame_files.sort()

    last_corners: Optional[np.ndarray] = None
    last_board: Optional[np.ndarray] = None
    last_results: Optional[Dict[str, Tuple[str, float]]] = None
    per_frame: Dict[str, Dict] = {}

    for idx, fname in enumerate(frame_files):
        start_t = time.perf_counter()
        fpath = os.path.join(frames_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            per_frame[fname] = {
                "success": False,
                "reason": "read_error",
                "process_time_s": 0.0,
                "inherited_corners": False,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "similarity_score": None,
                "square_changed_mask": None,
            }
            continue

        board_img = None
        corners_used = None
        inherited_corners = False
        similarity_score: Optional[float] = None
        square_changed_mask: Optional[List[List[int]]] = None

        # If we have previous corners, try reuse by comparing boards
        if last_corners is not None and last_board is not None:
            board_try = detector.extract_board_with_corners(img, last_corners)
            if board_try is not None:
                square_changed_mask, similarity_score = compute_square_change_mask(last_board, board_try)
                if similarity_score is not None and similarity_score >= similarity_threshold:
                    board_img = board_try
                    corners_used = last_corners
                    inherited_corners = True

        # Fallback to detection
        if board_img is None:
            board_img, corners = detector.detect_board_and_corners(img)
            corners_used = corners
            inherited_corners = False

        if board_img is None or corners_used is None:
            per_frame[fname] = {
                "success": False,
                "reason": "detect_failed",
                "process_time_s": float(time.perf_counter() - start_t),
                "inherited_corners": False,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "similarity_score": similarity_score,
                "square_changed_mask": square_changed_mask,
            }
            continue

        # Classify 64 squares via async service (partial if inherited)
        if inherited_corners and last_results is not None and square_changed_mask is not None:
            # Derive square names to classify: changed OR previously low confidence
            changed_set: Set[str] = set()
            for r in range(8):
                for c in range(8):
                    if square_changed_mask[r][c] == 1:
                        file_letter = chr(ord('a') + c)
                        rank_num = 8 - r
                        changed_set.add(f"{file_letter}{rank_num}")
            low_conf_set: Set[str] = {name for name, (_, conf) in last_results.items() if conf < min_square_confidence}
            reclassify_names: List[str] = sorted(changed_set | low_conf_set)
            results, all_high = classify_board_async(
                board_img,
                min_square_confidence,
                classifier=classifier,
                squares_to_classify=reclassify_names,
                previous_results=last_results,
            )
        else:
            # Full classification
            results, all_high = classify_board_async(board_img, min_square_confidence, classifier=classifier)

        # Compute confidence stats
        conf_values = [conf for (_, conf) in results.values()] if results else []
        min_conf = float(min(conf_values)) if conf_values else 0.0
        max_conf = float(max(conf_values)) if conf_values else 0.0

        # Save side-by-side image and classification if all squares high confidence
        save_ok = False
        # Render predicted board on the right, extracted board on the left
        h, w = board_img.shape[:2]
        if all_high:
            pred_img = _render_predicted_board_image(results, size=max(h, w), show_confidences=False)
            pred_img = cv2.resize(pred_img, (w, h), interpolation=cv2.INTER_AREA)
            combined = np.concatenate([board_img, pred_img], axis=1)
            out_path = os.path.join(output_dir, f"board_{fname}")
            cv2.imwrite(out_path, combined)
            save_ok = True
        else:
            # Low confidence: still save, with confidences overlayed
            pred_img = _render_predicted_board_image(results, size=max(h, w), show_confidences=True)
            pred_img = cv2.resize(pred_img, (w, h), interpolation=cv2.INTER_AREA)
            combined = np.concatenate([board_img, pred_img], axis=1)
            out_path = os.path.join(output_dir, f"board_lowconf_{fname}")
            cv2.imwrite(out_path, combined)

        elapsed = float(time.perf_counter() - start_t)
        per_frame[fname] = {
            "success": True,
            "all_high": all_high,
            "saved": save_ok,
            "corners": corners_used.tolist() if corners_used is not None else None,
            "inherited_corners": inherited_corners,
            "min_confidence": min_conf,
            "max_confidence": max_conf,
            "process_time_s": elapsed,
            "similarity_score": similarity_score,
            "square_changed_mask": square_changed_mask,
        }

        # Update cache
        last_corners = corners_used
        last_board = board_img
        last_results = results

    return per_frame


def process_all_videos(
    root_dir: str,
    detector_config_path: str,
    output_root: str,
    similarity_threshold: float = 0.75,
    min_square_confidence: float = 0.95,
    classifier_config: Optional[Dict] = None,
) -> Dict[str, Dict[str, Dict]]:
    os.makedirs(output_root, exist_ok=True)
    results: Dict[str, Dict[str, Dict]] = {}

    # Instantiate classifier once per run using provided configuration
    classifier: Optional[AsyncPieceClassifierService] = None
    if classifier_config is not None:
        cfg = dict(classifier_config)
        model_path = cfg.get('model_path')
        device = cfg.get('device')
        # Treat 'auto' as None to trigger device auto-selection
        if isinstance(device, str) and device.lower() == 'auto':
            device = None
        max_batch_size = cfg.get('max_batch_size', 256)
        batch_timeout_s = cfg.get('batch_timeout_s', 0.010)
        classifier = get_async_classifier(
            model_path=model_path,
            device=device,
            max_batch_size=int(max_batch_size),
            batch_timeout_s=float(batch_timeout_s),
        )
    else:
        classifier = get_async_classifier()

    video_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    video_dirs.sort()

    def _process_single_video(vname: str) -> Tuple[str, Dict[str, Dict]]:
        frames_dir = os.path.join(root_dir, vname)
        out_dir = os.path.join(output_root, vname)
        os.makedirs(out_dir, exist_ok=True)
        res = process_video_folder(
            frames_dir,
            detector_config_path,
            out_dir,
            similarity_threshold=similarity_threshold,
            min_square_confidence=min_square_confidence,
            classifier=classifier,
        )
        # Persist per-video per-frame summary
        with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
            json.dump(res, f, indent=2)

        # Persist per-video aggregated stats
        frames = list(res.values())
        num_frames = len(frames)
        num_success = sum(1 for r in frames if r.get('success'))
        num_saved = sum(1 for r in frames if r.get('saved'))
        num_inherited = sum(1 for r in frames if r.get('inherited_corners'))
        times = [float(r.get('process_time_s', 0.0)) for r in frames if 'process_time_s' in r]
        min_confs = [float(r.get('min_confidence', 0.0)) for r in frames if r.get('success')]
        max_confs = [float(r.get('max_confidence', 0.0)) for r in frames if r.get('success')]
        video_stats = {
            'video': vname,
            'frames': num_frames,
            'success_frames': num_success,
            'saved_frames': num_saved,
            'inherited_corners_frames': num_inherited,
            'avg_process_time_s': (float(sum(times)) / len(times)) if times else 0.0,
            'avg_min_confidence': (float(sum(min_confs)) / len(min_confs)) if min_confs else 0.0,
            'avg_max_confidence': (float(sum(max_confs)) / len(max_confs)) if max_confs else 0.0,
        }
        with open(os.path.join(out_dir, 'video_stats.json'), 'w') as f:
            json.dump(video_stats, f, indent=2)
        # Combined per-video report including per-frame records and summary
        with open(os.path.join(out_dir, 'video_summary.json'), 'w') as f:
            json.dump({
                'summary': video_stats,
                'frames': res,
            }, f, indent=2)

        # Print completion notice
        print(f"Finished video: {vname} | frames={num_frames}, success={num_success}, saved={num_saved}")
        return vname, res

    # Multi-thread across videos, sharing the same classifier
    max_workers = min(8, len(video_dirs)) if len(video_dirs) > 0 else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {executor.submit(_process_single_video, v): v for v in video_dirs}
        for fut in concurrent.futures.as_completed(future_to_video):
            v = future_to_video[fut]
            try:
                vname, res = fut.result()
                results[vname] = res
            except Exception as e:
                # Log error and continue
                print(f"Error processing video {v}: {e}")

    return results 