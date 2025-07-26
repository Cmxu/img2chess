"""
Enhanced chess board detection module for complex video layouts.
Builds on the existing board_detector.py with additional methods for handling
chess streaming layouts, overlays, and multi-element screens.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
import logging
from .board_detector import ChessBoardDetector

logger = logging.getLogger(__name__)

class EnhancedChessBoardDetector(ChessBoardDetector):
    """
    Enhanced chess board detector for complex streaming layouts.
    
    Adds methods specifically for handling chess streaming scenarios:
    - Multiple UI elements on screen
    - Player webcams and overlays
    - Leaderboards and chat windows
    - Different chess.com/lichess layouts
    """
    
    def __init__(self, min_board_area: int = 5000, max_board_area: int = 500000):
        super().__init__(min_board_area, max_board_area)
        # More lenient area thresholds for streaming layouts
        self.streaming_min_area = 5000
        self.streaming_max_area = max_board_area
        
    def detect_board(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Enhanced board detection for streaming layouts with multiple candidate evaluation.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Extracted chess board image or None if no board detected
        """
        # Collect ALL potential board candidates instead of returning first match
        all_candidates = []
        
        # First try the original detection methods (modified to collect candidates)
        candidates = self._collect_standard_candidates(image)
        all_candidates.extend(candidates)
        
        # PRIORITY 5: Region-based detection for streaming layouts
        logger.info("Trying region-based detection for streaming layouts...")
        candidates = self._collect_region_candidates(image)
        all_candidates.extend(candidates)
        
        # PRIORITY 6: Template matching for known layouts
        logger.info("Trying template-based detection...")
        candidates = self._collect_template_candidates(image)
        all_candidates.extend(candidates)
        
        # PRIORITY 7: Color-based detection for chess.com/lichess themes
        logger.info("Trying color-based detection...")
        candidates = self._collect_color_candidates(image)
        all_candidates.extend(candidates)
        
        if not all_candidates:
            logger.warning("❌ No chess board detected using any enhanced method")
            return None
        
        # Evaluate all candidates and select the best one
        best_candidate = self._select_best_candidate(image, all_candidates)
        
        if best_candidate is not None:
            method, corners, score = best_candidate
            logger.info(f"✅ Chess board detected using {method} with score {score:.1f}")
            return self._extract_board(image, corners)
        
        logger.warning("❌ No suitable chess board candidate found")
        return None
    
    def _collect_standard_candidates(self, image: np.ndarray) -> List[Tuple[str, np.ndarray, float]]:
        """Collect candidates using standard detection methods."""
        candidates = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try contour detection
        board_corners = self._detect_via_contours(gray)
        if board_corners is not None:
            score = self._score_chess_candidate(gray, board_corners)
            candidates.append(("contour", board_corners, score))
        
        # Try enhanced contour detection
        from .utils import enhance_image
        enhanced = enhance_image(image)
        enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        board_corners = self._detect_via_contours(enhanced_gray)
        if board_corners is not None:
            score = self._score_chess_candidate(enhanced_gray, board_corners)
            candidates.append(("enhanced_contour", board_corners, score))
        
        # Try corner detection
        board_corners = self._detect_via_corners(gray)
        if board_corners is not None:
            score = self._score_chess_candidate(gray, board_corners)
            candidates.append(("corner", board_corners, score))
        
        # Try enhanced corner detection
        board_corners = self._detect_via_corners(enhanced_gray)
        if board_corners is not None:
            score = self._score_chess_candidate(enhanced_gray, board_corners)
            candidates.append(("enhanced_corner", board_corners, score))
        
        return candidates
    
    def _collect_region_candidates(self, image: np.ndarray) -> List[Tuple[str, np.ndarray, float]]:
        """Collect candidates using region-based detection."""
        candidates = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Define regions where chess boards appear in streaming layouts
        regions = [
            {"name": "right_half", "slice": (0, h, w//2, w)},
            {"name": "center", "slice": (h//6, 5*h//6, w//6, 5*w//6)},
            {"name": "left_half", "slice": (0, h, 0, w//2)},
            {"name": "upper_right", "slice": (0, h//2, w//2, w)},
            {"name": "lower_right", "slice": (h//2, h, w//2, w)},
            {"name": "upper_left", "slice": (0, h//2, 0, w//2)},
            {"name": "lower_left", "slice": (h//2, h, 0, w//2)},
        ]
        
        for region in regions:
            try:
                y1, y2, x1, x2 = region["slice"]
                region_gray = gray[y1:y2, x1:x2]
                region_image = image[y1:y2, x1:x2]
                
                # Collect ALL candidates from this region, not just the first
                region_candidates = self._collect_region_contours_all(region_gray, region_image)
                
                # Adjust coordinates back to full image and add region info
                for corners, score in region_candidates:
                    adjusted_corners = corners.copy()
                    adjusted_corners[:, 0] += x1
                    adjusted_corners[:, 1] += y1
                    
                    method_name = f"region_based_{region['name']}"
                    candidates.append((method_name, adjusted_corners, score))
                    
            except Exception as e:
                logger.debug(f"Region {region['name']} detection failed: {e}")
                continue
        
        return candidates
    
    def _collect_template_candidates(self, image: np.ndarray) -> List[Tuple[str, np.ndarray, float]]:
        """Collect candidates using template matching (placeholder)."""
        # TODO: Implement template matching with common chess.com/lichess layouts
        return []
    
    def _collect_color_candidates(self, image: np.ndarray) -> List[Tuple[str, np.ndarray, float]]:
        """Collect candidates using color-based analysis."""
        candidates = []
        
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for common chess board themes
            color_ranges = [
                ("green", np.array([40, 30, 100]), np.array([80, 255, 255])),
                ("brown", np.array([10, 50, 80]), np.array([20, 255, 200])),
                ("blue", np.array([90, 50, 80]), np.array([130, 255, 255]))
            ]
            
            for color_name, lower, upper in color_ranges:
                try:
                    mask = cv2.inRange(hsv, lower, upper)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
                        area = cv2.contourArea(contour)
                        img_area = image.shape[0] * image.shape[1]
                        
                        if area < img_area * 0.05 or area > img_area * 0.8:
                            continue
                            
                        for epsilon_factor in [0.01, 0.02, 0.03]:
                            epsilon = epsilon_factor * cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, epsilon, True)
                            
                            if len(approx) == 4:
                                corners = approx.reshape(4, 2).astype(np.float32)
                                
                                if self._validate_streaming_corners(corners, image.shape[:2]):
                                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                    score = self._score_streaming_candidate(gray, corners)
                                    method_name = f"color_{color_name}"
                                    candidates.append((method_name, self._order_corners(corners), score))
                                    
                except Exception as e:
                    logger.debug(f"Color analysis for {color_name} failed: {e}")
                    continue
        
        except Exception as e:
            logger.debug(f"Color analysis detection failed: {e}")
        
        return candidates
    
    def _collect_region_contours_all(self, gray: np.ndarray, color_image: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Collect ALL contour candidates from a region, not just the first match.
        Returns list of (corners, score) tuples.
        """
        candidates = []
        
        # More aggressive preprocessing for streaming layouts
        preprocessing_methods = [
            lambda img: img,
            lambda img: cv2.equalizeHist(img),
            lambda img: cv2.GaussianBlur(img, (3, 3), 0),
            lambda img: cv2.bilateralFilter(img, 9, 75, 75),
        ]
        
        threshold_methods = [
            lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2),
            lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 2),
            lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            lambda img: cv2.Canny(img, 20, 80),
            lambda img: cv2.Canny(img, 40, 120),
        ]
        
        for preprocess in preprocessing_methods:
            for threshold_func in threshold_methods:
                try:
                    processed = preprocess(gray)
                    binary = threshold_func(processed)
                    
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Check MORE contours to find all candidates
                    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
                        area = cv2.contourArea(contour)
                        
                        img_area = gray.shape[0] * gray.shape[1]
                        min_area = max(2000, img_area * 0.10)
                        max_area = min(self.streaming_max_area, img_area * 0.90)
                        
                        if area < min_area or area > max_area:
                            continue
                            
                        x, y, w, h = cv2.boundingRect(contour)
                        bbox_aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 999
                        if bbox_aspect > 2.5:
                            continue
                        
                        for epsilon_factor in [0.005, 0.01, 0.02, 0.03, 0.05]:
                            epsilon = epsilon_factor * cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, epsilon, True)
                            
                            if len(approx) == 4:
                                corners = approx.reshape(4, 2).astype(np.float32)
                                
                                if self._validate_streaming_corners(corners, gray.shape):
                                    score = self._score_streaming_candidate(gray, corners)
                                    # Only keep candidates with reasonable scores
                                    if score > 20:
                                        ordered_corners = self._order_corners(corners)
                                        candidates.append((ordered_corners, score))
                except Exception as e:
                    logger.debug(f"Region contour collection failed: {e}")
                    continue
        
        return candidates
    
    def _select_best_candidate(self, image: np.ndarray, candidates: List[Tuple[str, np.ndarray, float]]) -> Optional[Tuple[str, np.ndarray, float]]:
        """
        Select the best candidate from all detected possibilities.
        Prioritizes larger, more complete boards.
        """
        if not candidates:
            return None
        
        logger.info(f"Evaluating {len(candidates)} board candidates...")
        
        # Enhanced scoring that heavily favors larger, more complete boards
        enhanced_candidates = []
        
        for method, corners, base_score in candidates:
            try:
                # Calculate additional metrics for selection
                area = cv2.contourArea(corners)
                
                # Extract board for detailed analysis
                board_size = 200
                dst_corners = np.array([
                    [0, 0], [board_size, 0], 
                    [board_size, board_size], [0, board_size]
                ], dtype=np.float32)
                
                transform_matrix = cv2.getPerspectiveTransform(corners, dst_corners)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                warped = cv2.warpPerspective(gray, transform_matrix, (board_size, board_size))
                
                # CRITICAL: Check if this looks like a complete 8x8 board
                completeness_score = self._score_board_completeness(warped)
                
                # REJECT candidates with very low completeness scores (likely partial boards)
                if completeness_score < 10:  # Much more lenient threshold for complete boards
                    logger.debug(f"Rejecting {method} due to low completeness score: {completeness_score:.1f}")
                    continue
                
                # Calculate comprehensive score
                # Base score (0-100) + area bonus + completeness bonus
                img_area = image.shape[0] * image.shape[1]
                area_ratio = area / img_area
                area_bonus = min(30, area_ratio * 100)  # Up to 30 points for larger area
                
                # Give MUCH higher weight to completeness for final scoring
                total_score = base_score + area_bonus + (completeness_score * 2)  # 2x weight for completeness
                
                enhanced_candidates.append((method, corners, total_score, area, completeness_score))
                
                logger.debug(f"{method}: base={base_score:.1f}, area_bonus={area_bonus:.1f}, completeness={completeness_score:.1f}, total={total_score:.1f}")
                
            except Exception as e:
                logger.debug(f"Failed to evaluate candidate {method}: {e}")
                continue
        
        if not enhanced_candidates:
            return None
        
        # Sort by total score (highest first)
        enhanced_candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Return the best candidate
        best = enhanced_candidates[0]
        method, corners, total_score, area, completeness = best
        
        logger.info(f"Selected {method} with score {total_score:.1f} (area: {area:.0f}, completeness: {completeness:.1f})")
        
        return (method, corners, total_score)
    
    def _score_board_completeness(self, warped_board: np.ndarray) -> float:
        """
        Score how complete/full this board detection is (0-50 points).
        Heavily penalizes partial boards that show only a section of the full board.
        """
        try:
            h, w = warped_board.shape
            
            # Check if we can see board edges/borders
            edge_score = self._check_board_edges(warped_board)
            
            # Check if the grid structure looks like a complete 8x8 board
            grid_score = self._check_complete_grid(warped_board)
            
            # Check piece distribution (complete boards should have pieces across the board)
            distribution_score = self._check_piece_distribution(warped_board)
            
            # Look for board coordinates/labels (a-h, 1-8) which indicate complete boards
            coordinate_score = self._check_board_coordinates(warped_board)
            
            total_score = edge_score + grid_score + distribution_score + coordinate_score
            
            logger.debug(f"Completeness: edges={edge_score:.1f}, grid={grid_score:.1f}, distribution={distribution_score:.1f}, coords={coordinate_score:.1f}")
            
            return min(50, total_score)  # Cap at 50 points
            
        except Exception as e:
            logger.debug(f"Completeness scoring failed: {e}")
            return 0.0
    
    def _check_board_edges(self, board: np.ndarray) -> float:
        """Check if board has clear edges (0-15 points)."""
        try:
            h, w = board.shape
            
            # Look for clear borders around the board
            # Complete boards often have visible borders/edges
            edges = cv2.Canny(board, 50, 150)
            
            # Check edge pixels around perimeter
            top_edge = np.sum(edges[0:3, :]) > 0
            bottom_edge = np.sum(edges[-3:, :]) > 0
            left_edge = np.sum(edges[:, 0:3]) > 0
            right_edge = np.sum(edges[:, -3:]) > 0
            
            edge_count = sum([top_edge, bottom_edge, left_edge, right_edge])
            
            # Complete boards should have at least 3 clear edges
            if edge_count >= 3:
                return 15.0
            elif edge_count >= 2:
                return 10.0
            elif edge_count >= 1:
                return 5.0
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _check_complete_grid(self, board: np.ndarray) -> float:
        """Check if grid structure looks like complete 8x8 (0-20 points)."""
        try:
            h, w = board.shape
            cell_h, cell_w = h // 8, w // 8
            
            if cell_h < 3 or cell_w < 3:  # More reasonable minimum size
                return 0.0  # Too small to be a complete board
            
            # Check that we can identify 8x8 distinct regions
            valid_cells = 0
            brightness_variance = []
            piece_like_cells = 0
            
            for i in range(8):
                for j in range(8):
                    y1, y2 = i * cell_h, (i + 1) * cell_h
                    x1, x2 = j * cell_w, (j + 1) * cell_w
                    
                    if y2 <= h and x2 <= w:  # Make sure cell is within bounds
                        cell = board[y1:y2, x1:x2]
                        if cell.size > 0:
                            cell_var = np.var(cell)
                            brightness_variance.append(cell_var)
                            valid_cells += 1
                            
                            # Check if this cell has piece-like content (high variance + reasonable brightness)
                            cell_mean = np.mean(cell)
                            if cell_var > 50 and 30 < cell_mean < 220:  # More lenient piece detection
                                piece_like_cells += 1
            
            # Complete boards should have all 64 cells visible
            cell_completeness = (valid_cells / 64.0) * 10  # Reduced base score
            
            # Complete boards should have varied cell content
            if brightness_variance:
                variance_score = min(3, np.mean(brightness_variance) / 200)  # More strict variance requirement
            else:
                variance_score = 0
            
            # CRITICAL: Must have reasonable number of pieces visible
            piece_score = min(7, piece_like_cells / 2)  # Up to 7 points for piece presence, more lenient
                
            return cell_completeness + variance_score + piece_score
            
        except Exception:
            return 0.0
    
    def _check_piece_distribution(self, board: np.ndarray) -> float:
        """Check piece distribution across board (0-10 points)."""
        try:
            # Divide board into quadrants and check for content in each
            h, w = board.shape
            mid_h, mid_w = h // 2, w // 2
            
            quadrants = [
                board[0:mid_h, 0:mid_w],          # Top-left
                board[0:mid_h, mid_w:w],          # Top-right  
                board[mid_h:h, 0:mid_w],          # Bottom-left
                board[mid_h:h, mid_w:w]           # Bottom-right
            ]
            
            active_quadrants = 0
            piece_indicators = 0
            
            for quad in quadrants:
                # Look for variation (pieces) in this quadrant
                quad_std = np.std(quad)
                quad_mean = np.mean(quad)
                
                # More sophisticated piece detection
                if quad_std > 30:  # Has content variation - increased threshold
                    active_quadrants += 1
                    
                    # Additional check for piece-like shapes using blob detection
                    try:
                        params = cv2.SimpleBlobDetector_Params()
                        params.filterByArea = True
                        params.minArea = 20
                        params.maxArea = 400
                        params.filterByCircularity = True
                        params.minCircularity = 0.2
                        
                        detector = cv2.SimpleBlobDetector_create(params)
                        keypoints = detector.detect(quad.astype(np.uint8))
                        
                        if len(keypoints) > 0:
                            piece_indicators += 1
                            
                    except Exception:
                        pass  # Fallback to std-based detection
            
            # Complete boards typically have pieces in multiple quadrants
            quadrant_score = (active_quadrants / 4.0) * 7
            piece_detection_score = (piece_indicators / 4.0) * 3
            
            return quadrant_score + piece_detection_score
            
        except Exception:
            return 0.0
    
    def _check_board_coordinates(self, board: np.ndarray) -> float:
        """Check for board coordinate labels (0-5 points)."""
        try:
            # Look for text-like patterns near edges that could be coordinates
            # This is a simple check - could be enhanced with OCR
            
            h, w = board.shape
            
            # Check edges for text-like patterns (high contrast small regions)
            edges = [
                board[0:h//10, :],           # Top edge
                board[-h//10:, :],           # Bottom edge
                board[:, 0:w//10],           # Left edge
                board[:, -w//10:]            # Right edge
            ]
            
            coordinate_indicators = 0
            for edge in edges:
                # Look for high contrast small regions (could be text)
                if edge.size > 0:
                    edge_std = np.std(edge)
                    if edge_std > 30:  # High contrast suggests text
                        coordinate_indicators += 1
            
            return min(5, coordinate_indicators * 1.25)
            
        except Exception:
            return 0.0
    
    def _detect_via_regions(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect chess board by analyzing different regions of the image.
        Useful for streaming layouts where the board is typically in a specific area.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Define common regions where chess boards appear in streaming layouts
        regions = [
            # Right side (common for chess.com layouts)
            {"name": "right_half", "slice": (0, h, w//2, w)},
            # Center region
            {"name": "center", "slice": (h//6, 5*h//6, w//6, 5*w//6)},
            # Left side
            {"name": "left_half", "slice": (0, h, 0, w//2)},
            # Upper right quadrant
            {"name": "upper_right", "slice": (0, h//2, w//2, w)},
            # Lower right quadrant  
            {"name": "lower_right", "slice": (h//2, h, w//2, w)},
            # Upper left quadrant
            {"name": "upper_left", "slice": (0, h//2, 0, w//2)},
            # Lower left quadrant
            {"name": "lower_left", "slice": (h//2, h, 0, w//2)},
        ]
        
        for region in regions:
            try:
                y1, y2, x1, x2 = region["slice"]
                region_gray = gray[y1:y2, x1:x2]
                region_image = image[y1:y2, x1:x2]
                
                logger.debug(f"Testing {region['name']} region ({x1},{y1}) to ({x2},{y2})")
                
                # Try contour detection on this region with more lenient settings
                board_corners = self._detect_region_contours(region_gray, region_image)
                
                if board_corners is not None:
                    # Adjust coordinates back to full image
                    board_corners[:, 0] += x1
                    board_corners[:, 1] += y1
                    
                    logger.info(f"Board detected in {region['name']} region")
                    return board_corners
                    
            except Exception as e:
                logger.debug(f"Region {region['name']} detection failed: {e}")
                continue
                
        return None
    
    def _detect_region_contours(self, gray: np.ndarray, color_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect chess board contours within a specific region with streaming-friendly settings.
        """
        # More aggressive preprocessing for streaming layouts
        preprocessing_methods = [
            # Standard preprocessing
            lambda img: img,
            # High contrast
            lambda img: cv2.equalizeHist(img),
            # Gaussian blur to reduce noise
            lambda img: cv2.GaussianBlur(img, (3, 3), 0),
            # Bilateral filter to preserve edges while reducing noise
            lambda img: cv2.bilateralFilter(img, 9, 75, 75),
        ]
        
        threshold_methods = [
            # More sensitive adaptive thresholding
            lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2),
            lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 2),
            # Multiple Otsu attempts
            lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            # More sensitive edge detection
            lambda img: cv2.Canny(img, 20, 80),
            lambda img: cv2.Canny(img, 40, 120),
        ]
        
        for preprocess in preprocessing_methods:
            for threshold_func in threshold_methods:
                try:
                    processed = preprocess(gray)
                    binary = threshold_func(processed)
                    
                    # Find contours
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # More lenient filtering for streaming layouts
                    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
                        area = cv2.contourArea(contour)
                        
                        # More permissive area requirements
                        img_area = gray.shape[0] * gray.shape[1]
                        min_area = max(2000, img_area * 0.10)  # At least 10% of region
                        max_area = min(self.streaming_max_area, img_area * 0.90)
                        
                        if area < min_area or area > max_area:
                            continue
                            
                        # More lenient aspect ratio
                        x, y, w, h = cv2.boundingRect(contour)
                        bbox_aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 999
                        if bbox_aspect > 2.5:  # More lenient than original 2.0
                            continue
                        
                        # Try polygon approximation with multiple epsilon values
                        for epsilon_factor in [0.005, 0.01, 0.02, 0.03, 0.05]:
                            epsilon = epsilon_factor * cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, epsilon, True)
                            
                            if len(approx) == 4:
                                corners = approx.reshape(4, 2).astype(np.float32)
                                
                                # Very lenient validation
                                if self._validate_streaming_corners(corners, gray.shape):
                                    # Score the candidate with streaming-specific scoring
                                    score = self._score_streaming_candidate(gray, corners)
                                    if score > 30:  # Lower threshold for streaming layouts
                                        logger.info(f"Board detected in region with score {score:.1f}")
                                        return self._order_corners(corners)
                except Exception as e:
                    logger.debug(f"Region contour detection failed: {e}")
                    continue
                    
        return None
    
    def _validate_streaming_corners(self, corners: np.ndarray, image_shape: tuple) -> bool:
        """
        Very lenient validation for streaming layout corners.
        """
        if corners is None or len(corners) != 4:
            return False
            
        # Check minimum area (very permissive)
        area = cv2.contourArea(corners)
        if area < 1000:  # Very low minimum
            return False
            
        # Very lenient aspect ratio
        rect = cv2.minAreaRect(corners)
        width, height = rect[1]
        if width == 0 or height == 0:
            return False
            
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 4.0:  # Very lenient
            return False
            
        return True
    
    def _score_streaming_candidate(self, gray: np.ndarray, corners: np.ndarray) -> float:
        """
        Score candidates with streaming-specific criteria.
        """
        try:
            # Extract region for analysis
            board_size = 200
            dst_corners = np.array([
                [0, 0], [board_size, 0], 
                [board_size, board_size], [0, board_size]
            ], dtype=np.float32)
            
            transform_matrix = cv2.getPerspectiveTransform(corners, dst_corners)
            warped = cv2.warpPerspective(gray, transform_matrix, (board_size, board_size))
            
            score = 0.0
            
            # HIGHEST: Checkerboard pattern (40 points)
            if self._has_checkerboard_pattern(warped):
                score += 40
                
            # HIGH: Grid structure (30 points)
            if self._has_grid_structure(warped):
                score += 30
                
            # MEDIUM: Alternating pattern (up to 25 points)
            alt_score = self._score_streaming_alternating_pattern(warped)
            score += alt_score
            
            # BONUS: Geometric properties (up to 15 points)
            geom_score = self._score_streaming_geometry(corners, gray.shape)
            score += geom_score
            
            # BONUS: Chess piece-like shapes (up to 10 points)
            piece_score = self._score_piece_detection(warped)
            score += piece_score
            
            return score
            
        except Exception as e:
            logger.debug(f"Streaming scoring failed: {e}")
            return 0.0
    
    def _score_streaming_alternating_pattern(self, image: np.ndarray) -> float:
        """Score alternating pattern with streaming-friendly criteria (0-25 points)"""
        h, w = image.shape
        cell_h, cell_w = h // 8, w // 8
        
        if cell_h < 3 or cell_w < 3:  # More lenient minimum cell size
            return 0.0
        
        brightness_values = []
        for i in range(8):
            for j in range(8):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell = image[y1:y2, x1:x2]
                brightness_values.append(np.mean(cell))
        
        brightness_std = np.std(brightness_values)
        brightness_range = np.max(brightness_values) - np.min(brightness_values)
        
        # More lenient contrast requirements
        contrast_score = 0.0
        if brightness_std > 20:  # Lower threshold
            contrast_score += min(12, brightness_std / 5)
        if brightness_range > 60:  # Lower threshold
            contrast_score += min(8, brightness_range / 20)
        
        # Check alternating pattern
        alternating_score = 0
        total_comparisons = 0
        
        for i in range(7):
            for j in range(7):
                current_idx = i * 8 + j
                right_idx = i * 8 + (j + 1)
                bottom_idx = (i + 1) * 8 + j
                
                diff_right = abs(brightness_values[current_idx] - brightness_values[right_idx])
                if diff_right > 15:  # Lower threshold
                    alternating_score += 1
                total_comparisons += 1
                
                diff_bottom = abs(brightness_values[current_idx] - brightness_values[bottom_idx])
                if diff_bottom > 15:  # Lower threshold
                    alternating_score += 1
                total_comparisons += 1
        
        alternating_ratio = alternating_score / total_comparisons
        pattern_score = alternating_ratio * 5
        
        return max(0, min(25, contrast_score + pattern_score))
    
    def _score_streaming_geometry(self, corners: np.ndarray, image_shape: tuple) -> float:
        """Score geometric properties for streaming layouts (0-15 points)"""
        try:
            # Aspect ratio score (0-8 points) - more lenient
            rect = cv2.minAreaRect(corners)
            width, height = rect[1]
            if width == 0 or height == 0:
                return 0.0
            
            aspect_ratio = max(width, height) / min(width, height)
            aspect_score = max(0, 8 - abs(aspect_ratio - 1.0) * 1.5)  # More lenient penalty
            
            # Position score (0-7 points) - prefer not at extreme edges
            img_h, img_w = image_shape
            center_x = np.mean(corners[:, 0])
            center_y = np.mean(corners[:, 1])
            
            norm_x = center_x / img_w
            norm_y = center_y / img_h
            
            # More forgiving edge distance scoring
            edge_distance_x = min(norm_x, 1.0 - norm_x)
            edge_distance_y = min(norm_y, 1.0 - norm_y)
            
            position_score = (min(edge_distance_x, 0.5) + min(edge_distance_y, 0.5)) * 7
            
            return aspect_score + position_score
            
        except Exception:
            return 0.0
    
    def _score_piece_detection(self, warped: np.ndarray) -> float:
        """
        Score based on detection of chess piece-like shapes (0-10 points).
        Look for circular/blob-like shapes that could be pieces.
        """
        try:
            # Use blob detection to find piece-like shapes
            params = cv2.SimpleBlobDetector_Params()
            
            # Filter by area
            params.filterByArea = True
            params.minArea = 50  # Small pieces
            params.maxArea = 2000  # Large pieces
            
            # Filter by circularity (pieces are often round-ish)
            params.filterByCircularity = True
            params.minCircularity = 0.3  # Not too strict
            
            # Filter by convexity
            params.filterByConvexity = True
            params.minConvexity = 0.4
            
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(warped)
            
            # Score based on number of detected blobs
            blob_count = len(keypoints)
            
            # Expect 4-32 pieces on a board
            if 4 <= blob_count <= 32:
                piece_score = min(10, blob_count / 3)  # Up to 10 points
            elif blob_count > 0:
                piece_score = min(5, blob_count / 6)   # Some credit for any pieces
            else:
                piece_score = 0
                
            logger.debug(f"Detected {blob_count} piece-like blobs, score: {piece_score:.1f}")
            return piece_score
            
        except Exception as e:
            logger.debug(f"Piece detection scoring failed: {e}")
            return 0.0
    
    def _detect_via_template_matching(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect chess board using template matching for known chess website layouts.
        This is a placeholder for future implementation with actual templates.
        """
        # TODO: Implement template matching with common chess.com/lichess layouts
        # Would require collecting template images of different board themes
        logger.debug("Template matching not yet implemented")
        return None
    
    def _detect_via_color_analysis(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect chess board using color-based analysis for common chess website themes.
        """
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for common chess board themes
            # Green boards (chess.com default)
            green_light = np.array([40, 30, 100])
            green_dark = np.array([80, 255, 255])
            
            # Brown boards (wood theme)
            brown_light = np.array([10, 50, 80])
            brown_dark = np.array([20, 255, 200])
            
            # Blue boards
            blue_light = np.array([90, 50, 80])
            blue_dark = np.array([130, 255, 255])
            
            color_ranges = [
                ("green", green_light, green_dark),
                ("brown", brown_light, brown_dark), 
                ("blue", blue_light, blue_dark)
            ]
            
            for color_name, lower, upper in color_ranges:
                try:
                    # Create color mask
                    mask = cv2.inRange(hsv, lower, upper)
                    
                    # Find contours in the color mask
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
                        area = cv2.contourArea(contour)
                        
                        # Check if contour is large enough to be a chess board
                        img_area = image.shape[0] * image.shape[1]
                        if area < img_area * 0.05 or area > img_area * 0.8:
                            continue
                            
                        # Try polygon approximation
                        for epsilon_factor in [0.01, 0.02, 0.03]:
                            epsilon = epsilon_factor * cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, epsilon, True)
                            
                            if len(approx) == 4:
                                corners = approx.reshape(4, 2).astype(np.float32)
                                
                                if self._validate_streaming_corners(corners, image.shape[:2]):
                                    logger.info(f"Board detected using {color_name} color analysis")
                                    return self._order_corners(corners)
                                    
                except Exception as e:
                    logger.debug(f"Color analysis for {color_name} failed: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Color analysis detection failed: {e}")
            return None