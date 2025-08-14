#!/usr/bin/env python3
import os
import cv2
import json
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import time
import concurrent.futures

from .clean_edge_detector import CleanEdgeBasedDetector
from .square_extractor import SquareExtractor
from .async_piece_classifier import get_async_classifier, AsyncPieceClassifierService

# New import for frame extraction from YouTube
from src.agents.frame_agent import FrameAgent
import threading
from queue import Queue
from tqdm import tqdm
import chess
import chess.pgn

# Re-export the main function
__all__ = ['process_youtube_video_by_interval']

# Import all the functions from the original module
from ..board_extractor_async import (
    boards_similar,
    compute_board_similarity,
    compute_square_change_mask,
    classify_board_async,
    _label_to_piece_char,
    _square_results_to_fen,
    _expand_rank,
    _is_valid_start_position,
    _piece_placement_to_full_fen,
    _determine_move_between_placements,
    _compress_positions,
    _infer_bottom_color_from_placement,
    _sanitize_pgn_comment,
    _collect_captions_for_range,
    _write_pgn_with_annotations,
    _load_piece_image,
    _overlay_bgra,
    _render_predicted_board_image,
    process_video_folder,
    process_all_videos,
    process_youtube_video_by_interval,
)