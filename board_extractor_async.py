import os
import cv2
import json
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import time
import concurrent.futures

from .img2chess.clean_edge_detector import CleanEdgeBasedDetector
from .img2chess.square_extractor import SquareExtractor
from .async_piece_classifier import get_async_classifier, AsyncPieceClassifierService

# New import for frame extraction from YouTube
from src.agents.frame_agent import FrameAgent
import threading
from queue import Queue
from tqdm import tqdm
import chess
import chess.pgn


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


# Helper: convert per-square results to FEN position string (position part only)
def _square_results_to_fen(square_results: Dict[str, List[object]]) -> str:
    """Create FEN position string from per-square results mapping.

    square_results maps like {'a8': ['wp', 0.99], ...}. We only use the labels.
    Returns the FEN position part (8 ranks separated by '/').
    """
    # Initialize empty 8x8 board with '.'
    board: List[List[str]] = [['.' for _ in range(8)] for _ in range(8)]

    for sq, val in square_results.items():
        if not isinstance(sq, str) or len(sq) != 2:
            continue
        file_char, rank_char = sq[0], sq[1]
        if file_char < 'a' or file_char > 'h':
            continue
        try:
            rank = int(rank_char)
        except Exception:
            continue
        if rank < 1 or rank > 8:
            continue
        label = None
        if isinstance(val, (list, tuple)) and len(val) >= 1:
            label = val[0]
        elif isinstance(val, dict) and 'label' in val:
            label = val['label']
        if not isinstance(label, str):
            continue

        col = ord(file_char) - ord('a')
        row = 8 - rank  # rank 8 is top row index 0

        if label == 'empty':
            continue
        piece_char = _label_to_piece_char(label)
        if piece_char:
            board[row][col] = piece_char

    # Convert to FEN rows
    fen_rows: List[str] = []
    for row in board:
        fen_row = ''
        empty_run = 0
        for cell in row:
            if cell == '.':
                empty_run += 1
            else:
                if empty_run > 0:
                    fen_row += str(empty_run)
                    empty_run = 0
                fen_row += cell
        if empty_run > 0:
            fen_row += str(empty_run)
        fen_rows.append(fen_row)

    return '/'.join(fen_rows)


# Helpers for game extraction
def _expand_rank(rank: str) -> List[str]:
    squares: List[str] = []
    for ch in rank:
        if ch.isdigit():
            squares.extend(['.'] * int(ch))
        else:
            squares.append(ch)
    return squares


def _is_valid_start_position(placement: str) -> bool:
    """Check Chess/Chess960-like starting setup by piece counts per back ranks and pawns ranks.

    - Rank 1 (bottom, index 7): 2x R, 2x N, 2x B, 1x K, 1x Q (uppercase)
    - Rank 2 (index 6): 8x P
    - Rank 7 (index 1): 8x p
    - Rank 8 (top, index 0): 2x r, 2x n, 2x b, 1x k, 1x q (lowercase)
    - Ranks 3..6 (indices 2..5): empty
    """
    ranks = placement.split('/')
    if len(ranks) != 8:
        return False
    r8, r7, r6, r5, r4, r3, r2, r1 = [ _expand_rank(r) for r in ranks ]
    # Validate empties
    for r in (r6, r5, r4, r3):
        if any(ch != '.' for ch in r):
            return False
    # Validate pawns
    if not all(ch == 'p' for ch in r7):
        return False
    if not all(ch == 'P' for ch in r2):
        return False
    # Validate back ranks counts
    from collections import Counter
    c8 = Counter(r8)
    c1 = Counter(r1)
    if not (c8.get('r', 0) == 2 and c8.get('n', 0) == 2 and c8.get('b', 0) == 2 and c8.get('k', 0) == 1 and c8.get('q', 0) == 1):
        return False
    if not (c1.get('R', 0) == 2 and c1.get('N', 0) == 2 and c1.get('B', 0) == 2 and c1.get('K', 0) == 1 and c1.get('Q', 0) == 1):
        return False
    return True


def _piece_placement_to_full_fen(placement: str, turn: str, fullmove: int) -> str:
    castling = 'KQkq'
    ep = '-'
    halfmove = 0
    turn_ch = 'w' if turn == 'w' else 'b'
    return f"{placement} {turn_ch} {castling} {ep} {halfmove} {fullmove}"


def _determine_move_between_placements(before: str, after: str, turn: str, fullmove: int) -> Dict[str, object]:
    """Determine a legal move from 'before' to 'after' piece placements.
    Uses python-chess, comparing board_fen only. Assumes Chess960 may be in effect.
    """
    try:
        full_fen = _piece_placement_to_full_fen(before, turn, fullmove)
        board = chess.Board(full_fen, chess960=True)
        after_board_fen = after
        for move in board.legal_moves:
            test = board.copy()
            test.push(move)
            if test.board_fen() == after_board_fen:
                try:
                    san = board.san(move)
                except Exception:
                    san = ''
                return {
                    'success': True,
                    'uci': move.uci(),
                    'san': san,
                }
        return {'success': False, 'uci': '', 'san': ''}
    except Exception as e:
        return {'success': False, 'uci': '', 'san': '', 'error': str(e)}


def _compress_positions(sorted_items: List[Tuple[str, Dict]]) -> List[Dict[str, object]]:
    """Compress consecutive identical FEN placements into segments with first/last frame and time."""
    segments: List[Dict[str, object]] = []
    prev_fen: Optional[str] = None
    for key, info in sorted_items:
        if not info.get('success') or not info.get('all_high'):
            continue
        fen = _square_results_to_fen(info.get('square_results') or {})
        t = float(info.get('time_s', 0.0))
        if prev_fen is None or fen != prev_fen:
            segments.append({
                'fen': fen,
                'first_frame': key,
                'last_frame': key,
                'first_time_s': t,
                'last_time_s': t,
                'count': 1,
            })
            prev_fen = fen
        else:
            segments[-1]['last_frame'] = key
            segments[-1]['last_time_s'] = t
            segments[-1]['count'] = int(segments[-1]['count']) + 1
    return segments


# New helpers: determine bottom color from FEN placement and build annotated PGN

def _infer_bottom_color_from_placement(placement: str) -> str:
    """Return 'white' if bottom rank contains mostly uppercase pieces, else 'black'."""
    try:
        ranks = placement.split('/')
        if len(ranks) != 8:
            return 'white'
        r1_expanded = _expand_rank(ranks[7])
        upper = sum(1 for ch in r1_expanded if isinstance(ch, str) and ch.isalpha() and ch.isupper())
        lower = sum(1 for ch in r1_expanded if isinstance(ch, str) and ch.isalpha() and ch.islower())
        return 'white' if upper >= lower else 'black'
    except Exception:
        return 'white'


def _sanitize_pgn_comment(text: str) -> str:
    """PGN comments are delimited by braces. Replace any braces in text to avoid nesting issues."""
    return text.replace('{', '(').replace('}', ')')


def _collect_captions_for_range(caption_intervals: List[Tuple[float, float, str]], start_s: float, end_s: float) -> List[str]:
    out: List[str] = []
    if start_s is None or end_s is None:
        return out
    for cs, ce, txt in caption_intervals:
        if ce > start_s and cs < end_s:
            out.append(txt)
    return out


def _write_pgn_with_annotations(
    start_placement: str,
    moves: List[Dict[str, object]],
    our_color_letter: str,
    start_time_s: float,
    end_time_s: float,
    caption_intervals: List[Tuple[float, float, str]],
    out_path: str,
) -> None:
    """Construct a PGN from detected moves and annotate our moves with intersecting captions."""
    # Initialize board from placement; side to move is always white at start of a new game in this pipeline
    full_fen = _piece_placement_to_full_fen(start_placement, 'w', 1)
    board = chess.Board(full_fen, chess960=True)

    game = chess.pgn.Game()
    game.setup(board)
    game.headers["SetUp"] = "1"
    game.headers["FEN"] = full_fen

    node = game

    # Precompute indices of our moves in the list and time ranges per our move
    our_indices: List[int] = [i for i, m in enumerate(moves) if m.get('mover') == our_color_letter]

    def _time_range_for_our_move(idx_in_moves: int, idx_in_ours: int) -> Tuple[float, float]:
        # Start at previous our move's appearance, else at game start
        if idx_in_ours == 0:
            start = float(start_time_s)
        else:
            prev_our = our_indices[idx_in_ours - 1]
            start = float(moves[prev_our].get('to_time_s', start_time_s))
        # End when this our move first appears on board
        end = float(moves[idx_in_moves].get('to_time_s', end_time_s))
        # Clamp
        if end < start:
            end = start
        return start, end

    for i, m in enumerate(moves):
        uci = m.get('uci') or ''
        to_fen = m.get('to_fen') or ''
        mv_obj: Optional[chess.Move] = None
        # Prefer provided UCI
        if isinstance(uci, str) and uci:
            try:
                mv_obj = board.parse_uci(uci)
            except Exception:
                mv_obj = None
        # Fallback: search by target placement
        if mv_obj is None and isinstance(to_fen, str) and to_fen:
            target = to_fen
            for mv in board.legal_moves:
                test = board.copy()
                test.push(mv)
                if test.board_fen() == target:
                    mv_obj = mv
                    break
        if mv_obj is None:
            # Stop if we cannot reconcile the move
            break
        node = node.add_main_variation(mv_obj)
        board.push(mv_obj)

        # If this move is ours, attach captions as a PGN comment
        if i in our_indices:
            idx_in_ours = our_indices.index(i)
            start_s, end_s = _time_range_for_our_move(i, idx_in_ours)
            texts = _collect_captions_for_range(caption_intervals, start_s, end_s)
            if texts:
                merged = ' '.join(t.strip() for t in texts if isinstance(t, str) and t.strip())
                if merged:
                    node.comment = _sanitize_pgn_comment(merged)

    with open(out_path, 'w', encoding='utf-8') as f:
        print(game, file=f)


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


def _render_predicted_board_image(results: Dict[str, Tuple[str, float]], size: int = 512, show_confidences: bool = False, confidence_threshold: Optional[float] = None) -> np.ndarray:
    """
    Render the predicted board using piece assets from img2chess/chess_pieces/modern.
    Falls back to letter rendering if an asset is missing.
    Optionally overlays per-square confidence scores.
    Confidence text is colored red when below threshold, green when above/equal.
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
                # Choose color based on threshold (BGR)
                if confidence_threshold is not None:
                    color = (0, 0, 255) if conf < confidence_threshold else (0, 255, 0)
                else:
                    color = (255, 255, 255)
                cv2.putText(img, conf_text, (cx, cy), font, conf_scale, color, conf_thickness, lineType=cv2.LINE_AA)

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
    failed_dir = os.path.join(output_dir, 'failed')
    os.makedirs(failed_dir, exist_ok=True)
    # Directory to save low-confidence square crops for this video
    lowconf_squares_dir = os.path.join(output_dir, 'low_conf_squares')
    os.makedirs(lowconf_squares_dir, exist_ok=True)
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
            reason_text = "read_error: could not read image"
            placeholder = np.zeros((512, 512, 3), dtype=np.uint8)
            header = f"Frame: {fname}"
            lines = ["EXTRACTION FAILED", header, reason_text]
            y = 40
            for line in lines:
                cv2.putText(placeholder, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                y += 30
            cv2.imwrite(os.path.join(failed_dir, fname), placeholder)
            per_frame[fname] = {
                "success": False,
                "reason": reason_text,
                "process_time_s": 0.0,
                "inherited_corners": False,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "similarity_score": None,
                "square_changed_mask": None,
                "square_results": {},
                "low_confidence_squares": [],
            }
            continue

        board_img = None
        corners_used = None
        inherited_corners = False
        similarity_score: Optional[float] = None
        square_changed_mask: Optional[List[List[int]]] = None
        failure_chain: List[str] = []

        # If we have previous corners, try reuse by comparing boards
        if last_corners is not None and last_board is not None:
            board_try = detector.extract_board_with_corners(img, last_corners)
            if board_try is not None:
                square_changed_mask, similarity_score = compute_square_change_mask(last_board, board_try)
                if similarity_score is not None and similarity_score >= similarity_threshold:
                    board_img = board_try
                    corners_used = last_corners
                    inherited_corners = True
                else:
                    failure_chain.append("similarity_below_threshold")
            else:
                failure_chain.append("extract_with_previous_corners_failed")

        # Fallback to detection
        if board_img is None:
            detected_board_img, detected_corners = detector.detect_board_and_corners(img)
            board_img = detected_board_img
            corners_used = detected_corners
            inherited_corners = False
            if board_img is None and detected_corners is None:
                failure_chain.append("detect_board_and_corners_failed")
            elif board_img is None:
                failure_chain.append("detect_board_failed")
            elif detected_corners is None:
                failure_chain.append("detect_corners_failed")

        if board_img is None or corners_used is None:
            # Annotate and save failure image
            fail_img = img if img is not None else np.zeros((512, 512, 3), dtype=np.uint8)
            reason_str = " -> ".join(failure_chain) if failure_chain else "detect_failed"
            y = 40
            cv2.putText(fail_img, "EXTRACTION FAILED", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            y += 35
            cv2.putText(fail_img, f"Frame: {fname}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, lineType=cv2.LINE_AA)
            y += 30
            cv2.putText(fail_img, f"Reason: {reason_str}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.imwrite(os.path.join(failed_dir, fname), fail_img)

            per_frame[fname] = {
                "success": False,
                "reason": reason_str,
                "process_time_s": float(time.perf_counter() - start_t),
                "inherited_corners": False,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "similarity_score": similarity_score,
                "square_changed_mask": square_changed_mask,
                "square_results": {},
                "low_confidence_squares": [],
            }
            continue

        # Classify 64 squares via async service (partial if inherited)
        if inherited_corners and last_results is not None and square_changed_mask is not None:
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
            results, all_high = classify_board_async(board_img, min_square_confidence, classifier=classifier)

        # Compute confidence stats
        conf_values = [conf for (_, conf) in results.values()] if results else []
        min_conf = float(min(conf_values)) if conf_values else 0.0
        max_conf = float(max(conf_values)) if conf_values else 0.0

        # Save side-by-side image and classification if all squares high confidence
        save_ok = False
        h, w = board_img.shape[:2]
        if all_high:
            pred_img = _render_predicted_board_image(results, size=max(h, w), show_confidences=False)
            pred_img = cv2.resize(pred_img, (w, h), interpolation=cv2.INTER_AREA)
            combined = np.concatenate([board_img, pred_img], axis=1)
            out_path = os.path.join(output_dir, f"board_{fname}")
            cv2.imwrite(out_path, combined)
            save_ok = True
        else:
            # Low confidence: still save, with confidences overlayed (red/green by threshold)
            pred_img = _render_predicted_board_image(results, size=max(h, w), show_confidences=True, confidence_threshold=min_square_confidence)
            pred_img = cv2.resize(pred_img, (w, h), interpolation=cv2.INTER_AREA)
            combined = np.concatenate([board_img, pred_img], axis=1)
            out_path = os.path.join(output_dir, f"board_lowconf_{fname}")
            cv2.imwrite(out_path, combined)

            # Save crops of low-confidence squares for debugging/analysis
            try:
                extractor = SquareExtractor(square_size=224)
                squares_map = extractor.extract_squares(board_img)
                low_conf_squares = [name for name, (_, conf) in results.items() if conf < min_square_confidence]
                base = os.path.splitext(fname)[0]
                for sq_name in low_conf_squares:
                    if sq_name in squares_map:
                        sq_img = squares_map[sq_name]
                        cv2.imwrite(os.path.join(lowconf_squares_dir, f"{base}_{sq_name}.png"), sq_img)
            except Exception:
                # Do not crash on debugging artifact generation
                pass

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
            # Persist square-level results for downstream aggregation/visualization
            "square_results": {k: [v[0], float(v[1])] for k, v in results.items()} if results else {},
            "low_confidence_squares": [name for name, (_, conf) in results.items() if conf < min_square_confidence],
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
        # Aggregate failure reasons
        failure_reasons: Dict[str, int] = {}
        for r in frames:
            if not r.get('success'):
                reason_key = r.get('reason', 'unknown')
                failure_reasons[reason_key] = failure_reasons.get(reason_key, 0) + 1
        video_stats = {
            'video': vname,
            'frames': num_frames,
            'success_frames': num_success,
            'saved_frames': num_saved,
            'inherited_corners_frames': num_inherited,
            'avg_process_time_s': (float(sum(times)) / len(times)) if times else 0.0,
            'avg_min_confidence': (float(sum(min_confs)) / len(min_confs)) if min_confs else 0.0,
            'avg_max_confidence': (float(sum(max_confs)) / len(max_confs)) if max_confs else 0.0,
            'failure_reasons': failure_reasons,
        }
        with open(os.path.join(out_dir, 'video_stats.json'), 'w') as f:
            json.dump(video_stats, f, indent=2)
        with open(os.path.join(out_dir, 'video_summary.json'), 'w') as f:
            json.dump({
                'summary': video_stats,
                'frames': res,
            }, f, indent=2)

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


# New high-level method: process a single YouTube video by sampling interval
def process_youtube_video_by_interval(
    youtube_url: str,
    interval_seconds: float,
    output_root: str,
    detector_config_path: str = os.path.join('img2chess', 'detector_config.yaml'),
    similarity_threshold: float = 0.75,
    min_square_confidence: float = 0.95,
    classifier_config: Optional[Dict] = None,
    min_frames_between_resets: int = 10,
) -> Dict:
    """
    Download a YouTube video and sample frames at the given interval, extracting boards.

    - Skips frames that fail extraction or have low confidence squares
    - Segments into multiple games if a position returns to the original position
    - Streams frames without saving them to disk; a producer thread prefetches ~20 frames

    Returns a dict containing per-game segmentation and per-frame metadata.
    """
    os.makedirs(output_root, exist_ok=True)

    # Use FrameAgent to manage video download and metadata
    frame_agent = FrameAgent()
    video_id = frame_agent.extract_video_id(youtube_url) or 'video'
    video_dir = os.path.join(output_root, video_id)
    os.makedirs(video_dir, exist_ok=True)

    # Reuse existing downloaded video if present; else download into video_dir
    local_video_path = None
    for ext in ['mp4', 'webm', 'mkv', 'avi']:
        candidate = os.path.join(video_dir, f"temp_video_{video_id}.{ext}")
        if os.path.exists(candidate):
            local_video_path = candidate
            break
    if not local_video_path:
        try:
            local_video_path = frame_agent._download_video(youtube_url, video_dir)  # type: ignore[attr-defined]
        except Exception:
            local_video_path = None
    if not local_video_path or not os.path.exists(local_video_path):
        raise RuntimeError("Failed to download video for processing.")

    # Determine timestamps based on requested sampling interval
    video_info = frame_agent.get_video_info(youtube_url) or {}
    duration = float(video_info.get('duration', 0.0))
    if duration <= 0.0:
        # Fallback: try OpenCV probe
        cap_probe = cv2.VideoCapture(local_video_path)
        if cap_probe.isOpened():
            fps = cap_probe.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = cap_probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            duration = float(frame_count / fps) if fps > 0 else 0.0
        cap_probe.release()
    if duration <= 0.0:
        raise RuntimeError("Could not determine video duration")

    timestamps: List[float] = []
    t = 0.0
    while t <= duration:
        timestamps.append(t)
        t += float(interval_seconds)

    # Instantiate classifier once
    classifier: Optional[AsyncPieceClassifierService] = None
    if classifier_config is not None:
        cfg = dict(classifier_config)
        model_path = cfg.get('model_path')
        device = cfg.get('device')
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

    # Prepare detector and per-frame results accumulator
    detector = CleanEdgeBasedDetector(detector_config_path)
    per_frame: Dict[str, Dict] = {}

    # Streaming: use a producer thread to prefetch frames into a bounded queue
    queue_size = 20
    q: "Queue[Optional[Tuple[str, float, np.ndarray]]]" = Queue(maxsize=queue_size)
    stop_sentinel: Optional[Tuple[str, float, np.ndarray]] = None  # use None to mark end

    def _producer(video_path: str, ts_list: List[float]) -> None:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                # signal failure
                q.put(None)
                return
            # Prefer frame-accurate seeking by frames
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            for ts in ts_list:
                if fps > 0:
                    frame_idx = int(round(ts * fps))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                else:
                    cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
                ok, frame_bgr = cap.read()
                key = f"{video_id}_{ts:.2f}s"
                if ok and frame_bgr is not None:
                    q.put((key, ts, frame_bgr))
                else:
                    # put a placeholder to allow consumer to continue
                    q.put((key, ts, None))
            cap.release()
        finally:
            q.put(stop_sentinel)

    prod_thread = threading.Thread(target=_producer, args=(local_video_path, timestamps), daemon=True)
    prod_thread.start()

    # Progress bar
    try:
        pbar = tqdm(total=len(timestamps), desc=f"Processing frames ({video_id})", dynamic_ncols=True)
    except Exception:
        pbar = None

    # Process frames as they arrive
    last_corners: Optional[np.ndarray] = None
    last_board: Optional[np.ndarray] = None
    last_results: Optional[Dict[str, Tuple[str, float]]] = None
    attempted = 0

    while True:
        item = q.get()
        if item is None:
            break
        key, ts, frame = item
        if frame is None:
            # record failure and continue
            per_frame[key] = {
                'success': False,
                'reason': 'read_error: could not decode frame',
                'process_time_s': 0.0,
                'inherited_corners': False,
                'min_confidence': 0.0,
                'max_confidence': 0.0,
                'similarity_score': None,
                'square_changed_mask': None,
                'square_results': {},
                'low_confidence_squares': [],
                'time_s': float(ts),
            }
            attempted += 1
            if pbar is not None:
                pbar.update(1)
            continue

        attempted += 1
        start_t = time.perf_counter()
        img = frame

        board_img = None
        corners_used = None
        inherited_corners = False
        similarity_score: Optional[float] = None
        square_changed_mask: Optional[List[List[int]]] = None

        # Try reuse previous corners if possible
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
            detected_board_img, detected_corners = detector.detect_board_and_corners(img)
            board_img = detected_board_img
            corners_used = detected_corners
            inherited_corners = False

        if board_img is None or corners_used is None:
            per_frame[key] = {
                'success': False,
                'reason': 'detect_failed',
                'process_time_s': float(time.perf_counter() - start_t),
                'inherited_corners': False,
                'min_confidence': 0.0,
                'max_confidence': 0.0,
                'similarity_score': similarity_score,
                'square_changed_mask': square_changed_mask,
                'square_results': {},
                'low_confidence_squares': [],
                'time_s': float(ts),
            }
            continue

        # Classify squares (partial if reusing corners)
        if inherited_corners and last_results is not None and square_changed_mask is not None:
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
            results, all_high = classify_board_async(board_img, min_square_confidence, classifier=classifier)

        conf_values = [conf for (_, conf) in results.values()] if results else []
        min_conf = float(min(conf_values)) if conf_values else 0.0
        max_conf = float(max(conf_values)) if conf_values else 0.0

        per_frame[key] = {
            'success': True,
            'all_high': all_high,
            'saved': False,
            'corners': corners_used.tolist() if corners_used is not None else None,
            'inherited_corners': inherited_corners,
            'min_confidence': min_conf,
            'max_confidence': max_conf,
            'process_time_s': float(time.perf_counter() - start_t),
            'similarity_score': similarity_score,
            'square_changed_mask': square_changed_mask,
            'square_results': {k: [v[0], float(v[1])] for k, v in results.items()} if results else {},
            'low_confidence_squares': [name for name, (_, conf) in results.items() if conf < min_square_confidence],
            'time_s': float(ts),
        }

        # update caches
        last_corners = corners_used
        last_board = board_img
        last_results = results

        if pbar is not None:
            pbar.update(1)

    # Build per-frame timeline sorted by timestamp
    sorted_items = sorted(per_frame.items(), key=lambda kv: (kv[1].get('time_s', float('inf'))))

    # Build distinct position segments and persist a positions debug file
    segments = _compress_positions(sorted_items)
    positions_debug = {
        'video_id': video_id,
        'url': youtube_url,
        'interval_seconds': float(interval_seconds),
        'positions': segments,
    }
    with open(os.path.join(video_dir, 'positions_debug.json'), 'w') as f:
        json.dump(positions_debug, f, indent=2)

    # Segment games by valid start positions and compute moves
    games: List[Dict[str, object]] = []
    idx = 0
    game_index = 1
    while idx < len(segments):
        seg = segments[idx]
        if not _is_valid_start_position(seg['fen']):
            idx += 1
            continue
        # Start a game at this segment
        start_seg = seg
        moves: List[Dict[str, object]] = []
        turn = 'w'  # white to move from a valid start position
        fullmove = 1
        j = idx + 1
        last_seg = seg
        while j < len(segments):
            next_seg = segments[j]
            if _is_valid_start_position(next_seg['fen']):
                # next game begins; stop current game before this
                break
            # compute move from last_seg -> next_seg
            move_info = _determine_move_between_placements(last_seg['fen'], next_seg['fen'], turn, fullmove)
            moves.append({
                'from_fen': last_seg['fen'],
                'to_fen': next_seg['fen'],
                'from_time_s': last_seg['first_time_s'],
                'to_time_s': next_seg['first_time_s'],
                'uci': move_info.get('uci', ''),
                'san': move_info.get('san', ''),
                'success': bool(move_info.get('success', False)),
                'mover': turn,
                'fullmove': int(fullmove),
            })
            # advance
            last_seg = next_seg
            # toggle turn and fullmove
            if turn == 'w':
                turn = 'b'
            else:
                turn = 'w'
                fullmove += 1
            j += 1

        # Infer our color based on the bottom side at game start
        our_color_word = _infer_bottom_color_from_placement(start_seg['fen'])
        games.append({
            'game_index': game_index,
            'start_position': start_seg['fen'],
            'start_time_s': start_seg['first_time_s'],
            'end_position': last_seg['fen'],
            'end_time_s': last_seg['last_time_s'],
            'moves': moves,
            'our_color': our_color_word,
        })
        game_index += 1
        idx = j  # continue scanning from next start position

    # NOTE: Caption extraction and PGN generation have been moved out of the
    # video processing flow. Use reconstruct_game_tree.py after extraction to
    # build a full PGN and optionally annotate it with captions.

    # Persist results
    summary = {
        'video_id': video_id,
        'url': youtube_url,
        'interval_seconds': float(interval_seconds),
        'frames_sampled': len(timestamps),
        'games': games,
    }

    with open(os.path.join(video_dir, 'games.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    if pbar is not None:
        pbar.close()

    print(f"Processed video {video_id}: {len(games)} game(s)")
    return summary 