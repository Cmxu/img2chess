"""
Chess board detection module using computer vision techniques.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class ChessBoardDetector:
    """
    Detects chess boards in images using computer vision techniques.
    
    Uses a combination of corner detection, contour analysis, and geometric
    validation to identify and extract chess board regions from images.
    """
    
    def __init__(self, min_board_area: int = 10000, max_board_area: int = 500000):
        """
        Initialize the chess board detector.
        
        Args:
            min_board_area: Minimum area threshold for valid board detection
            max_board_area: Maximum area threshold for valid board detection
        """
        self.min_board_area = min_board_area
        self.max_board_area = max_board_area
        
    def detect_board(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and extract the chess board from an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Extracted chess board image or None if no board detected
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # PRIORITY 1: Contour detection (best for real chess boards with pieces)
        logger.info("Trying contour detection (best for boards with pieces)...")
        board_corners = self._detect_via_contours(gray)
        
        if board_corners is not None:
            logger.info("✅ Chess board detected using contour detection")
            board = self._extract_board(image, board_corners)
            return board
            
        # PRIORITY 2: Enhanced contour detection
        logger.info("Contour detection failed, trying enhanced image...")
        from .utils import enhance_image
        enhanced = enhance_image(image)
        enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        board_corners = self._detect_via_contours(enhanced_gray)
        if board_corners is not None:
            logger.info("✅ Chess board detected using contour detection on enhanced image")
            board = self._extract_board(image, board_corners)
            return board
        
        # PRIORITY 3: Corner detection (only works for empty boards)
        logger.info("Contour detection failed, trying corner detection...")
        board_corners = self._detect_via_corners(gray)
        
        if board_corners is not None:
            logger.info("✅ Chess board detected using corner detection")
            board = self._extract_board(image, board_corners)
            return board
            
        # PRIORITY 4: Enhanced corner detection
        logger.info("Trying corner detection on enhanced image...")
        board_corners = self._detect_via_corners(enhanced_gray)
        
        if board_corners is not None:
            logger.info("✅ Chess board detected using corner detection on enhanced image")
            board = self._extract_board(image, board_corners)
            return board
        
        logger.warning("❌ No chess board detected in image using any method")
        return None
    
    def _detect_via_corners(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect chess board using corner detection (checkerboard pattern).
        
        Args:
            gray: Grayscale image
            
        Returns:
            Board corner coordinates or None
        """
        # Try different board sizes (internal corners) - more comprehensive
        board_sizes = [(7, 7), (6, 6), (8, 8), (5, 5), (9, 9), (4, 4)]
        
        # Try different detection flags
        flag_combinations = [
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS,
            cv2.CALIB_CB_ADAPTIVE_THRESH,
            cv2.CALIB_CB_NORMALIZE_IMAGE,
            cv2.CALIB_CB_FILTER_QUADS,
            0  # No flags
        ]
        
        for board_size in board_sizes:
            for flags in flag_combinations:
                # Find chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, board_size, flags)
                
                if ret:
                    # Refine corner locations
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    
                    # Get board boundary from internal corners
                    try:
                        board_corners = self._get_board_boundary(corners, board_size)
                        if self._validate_board_corners(board_corners):
                            logger.info(f"Board detected using corner method with size {board_size} and flags {flags}")
                            return board_corners
                    except Exception as e:
                        logger.debug(f"Failed to get board boundary: {e}")
                        continue
                    
        return None
    
    def _detect_via_contours(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect chess board using contour analysis.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Board corner coordinates or None
        """
        # Try multiple thresholding approaches
        threshold_methods = [
            # Adaptive thresholding
            lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
            lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
            # Otsu thresholding
            lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            # Edge detection
            lambda img: cv2.Canny(img, 50, 150),
            lambda img: cv2.Canny(img, 30, 100),
        ]
        
        for threshold_func in threshold_methods:
            try:
                binary = threshold_func(gray)
                
                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours by area and shape
                for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:  # Check top 20
                    area = cv2.contourArea(contour)
                    
                    # Get image dimensions for relative area filtering
                    img_area = gray.shape[0] * gray.shape[1]
                    
                    # Chess board should be significant but not the entire image
                    min_area = max(10000, img_area * 0.15)  # At least 15% of image for better detection
                    max_area = min(self.max_board_area, img_area * 0.85)  # At most 85% of image
                    
                    if area < min_area or area > max_area:
                        logger.debug(f"Contour area {area:.0f} outside valid range [{min_area:.0f}, {max_area:.0f}]")
                        continue
                    
                    # Check bounding box aspect ratio first (quick filter)
                    x, y, w, h = cv2.boundingRect(contour)
                    bbox_aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 999
                    if bbox_aspect > 2.0:  # Chess boards should be roughly square
                        logger.debug(f"Contour bbox aspect ratio {bbox_aspect:.2f} too extreme")
                        continue
                    
                    # Exclude contours that touch image borders (usually entire image outline)
                    img_h, img_w = gray.shape
                    border_margin = 5  # pixels from edge
                    touches_border = (x <= border_margin or y <= border_margin or 
                                    x + w >= img_w - border_margin or y + h >= img_h - border_margin)
                    if touches_border:
                        logger.debug(f"Contour touches image border at ({x},{y},{w},{h}) - likely image outline")
                        continue
                    
                    # Try different epsilon values for polygon approximation
                    for epsilon_factor in [0.01, 0.02, 0.03, 0.04]:
                        epsilon = epsilon_factor * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        
                        # Look for quadrilateral (4 corners)
                        if len(approx) == 4:
                            corners = approx.reshape(4, 2).astype(np.float32)
                            # Use more lenient validation for contour detection
                            if self._validate_board_corners_lenient(corners):
                                # Score this candidate instead of binary validation
                                score = self._score_chess_candidate(gray, corners)
                                if score > 30:  # More lenient acceptance threshold
                                    logger.info(f"Board detected using contour method with score {score:.1f}")
                                    return self._order_corners(corners)
            except Exception as e:
                logger.debug(f"Threshold method failed: {e}")
                continue
                    
        return None
    
    def _get_board_boundary(self, internal_corners: np.ndarray, board_size: tuple) -> np.ndarray:
        """
        Estimate board boundary from internal corners.
        
        Args:
            internal_corners: Internal corner points
            board_size: Size of internal corner grid
            
        Returns:
            Board boundary corner coordinates
        """
        corners = internal_corners.reshape(-1, 2)
        rows, cols = board_size
        
        # Get corner points of the grid
        top_left = corners[0]
        top_right = corners[cols - 1]
        bottom_left = corners[(rows - 1) * cols]
        bottom_right = corners[-1]
        
        # Estimate board boundary by extending from internal corners
        # Calculate grid spacing
        h_spacing = (top_right - top_left) / (cols - 1)
        v_spacing = (bottom_left - top_left) / (rows - 1)
        
        # Extend to board edges
        board_corners = np.array([
            top_left - h_spacing - v_spacing,      # Top-left board corner
            top_right + h_spacing - v_spacing,     # Top-right board corner  
            bottom_right + h_spacing + v_spacing,  # Bottom-right board corner
            bottom_left - h_spacing + v_spacing    # Bottom-left board corner
        ], dtype=np.float32)
        
        return board_corners
    
    def _validate_board_corners(self, corners: np.ndarray) -> bool:
        """
        Validate that detected corners form a reasonable chess board.
        
        Args:
            corners: Array of 4 corner points
            
        Returns:
            True if corners are valid
        """
        if corners is None or len(corners) != 4:
            return False
            
        # Check if corners form a convex quadrilateral
        area = cv2.contourArea(corners)
        if area < self.min_board_area:
            return False
            
        # Check aspect ratio (chess boards are square)
        rect = cv2.minAreaRect(corners)
        width, height = rect[1]
        aspect_ratio = max(width, height) / min(width, height)
        
        if aspect_ratio > 1.5:  # Allow some perspective distortion
            return False
            
        return True
    
    def _validate_board_corners_lenient(self, corners: np.ndarray) -> bool:
        """
        More lenient validation for contour-detected corners.
        
        Args:
            corners: Array of 4 corner points
            
        Returns:
            True if corners are valid
        """
        if corners is None or len(corners) != 4:
            return False
            
        # Check if corners form a convex quadrilateral
        area = cv2.contourArea(corners)
        if area < 500:  # Very permissive minimum area
            return False
            
        # More lenient aspect ratio check
        rect = cv2.minAreaRect(corners)
        width, height = rect[1]
        if width == 0 or height == 0:
            return False
            
        aspect_ratio = max(width, height) / min(width, height)
        
        if aspect_ratio > 3.0:  # Very lenient aspect ratio
            return False
            
        return True
    
    def _validate_chess_pattern(self, gray: np.ndarray, corners: np.ndarray) -> bool:
        """
        Validate that the detected region actually contains a chess board pattern.
        
        Args:
            gray: Grayscale image
            corners: Detected corner points
            
        Returns:
            True if the region contains chess-like patterns
        """
        try:
            # Extract the region for analysis
            board_size = 200  # Temporary size for analysis
            dst_corners = np.array([
                [0, 0], [board_size, 0], 
                [board_size, board_size], [0, board_size]
            ], dtype=np.float32)
            
            transform_matrix = cv2.getPerspectiveTransform(corners, dst_corners)
            warped = cv2.warpPerspective(gray, transform_matrix, (board_size, board_size))
            
            # PRIORITY 1: Actual checkerboard pattern (most reliable)
            if self._has_checkerboard_pattern(warped):
                logger.debug("✅ Found actual checkerboard pattern - high confidence")
                return True
            
            # PRIORITY 2: Strong grid structure + good alternating pattern
            has_grid = self._has_grid_structure(warped)
            has_alternating = self._has_alternating_pattern(warped)
            
            if has_grid and has_alternating:
                logger.debug("✅ Found grid structure + alternating pattern - medium confidence")
                return True
            
            # PRIORITY 3: Very strong alternating pattern alone (but be more strict)
            if has_alternating:
                # Additional validation for alternating-only detection
                if self._validate_strong_chess_pattern(warped):
                    logger.debug("✅ Found very strong alternating pattern - low confidence")
                    return True
                else:
                    logger.debug("❌ Alternating pattern too weak for chess board")
                    return False
            
            logger.debug("❌ No convincing chess patterns found")
            return False
            
        except Exception as e:
            logger.debug(f"Chess pattern validation failed: {e}")
            return False
    
    def _has_checkerboard_pattern(self, image: np.ndarray) -> bool:
        """Check if image contains a checkerboard pattern."""
        # Try to detect corners in the warped image
        for size in [(7, 7), (6, 6), (5, 5), (4, 4)]:
            ret, _ = cv2.findChessboardCorners(
                image, size, 
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if ret:
                logger.debug(f"Found checkerboard pattern of size {size}")
                return True
        return False
    
    def _has_grid_structure(self, image: np.ndarray) -> bool:
        """Check if image has a grid-like structure."""
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Look for horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        
        # Count significant line pixels
        h_pixels = np.sum(horizontal_lines > 0)
        v_pixels = np.sum(vertical_lines > 0)
        
        # Should have reasonable number of grid lines
        min_line_pixels = image.shape[0] * 2  # At least 2 lines worth of pixels
        if h_pixels > min_line_pixels and v_pixels > min_line_pixels:
            logger.debug(f"Found grid structure: h_pixels={h_pixels}, v_pixels={v_pixels}")
            return True
        return False
    
    def _has_alternating_pattern(self, image: np.ndarray) -> bool:
        """Check for alternating light/dark pattern typical of chess boards."""
        # Divide image into 8x8 grid and check variance
        h, w = image.shape
        cell_h, cell_w = h // 8, w // 8
        
        if cell_h < 5 or cell_w < 5:  # Too small to analyze
            return False
        
        brightness_values = []
        for i in range(8):
            for j in range(8):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell = image[y1:y2, x1:x2]
                brightness_values.append(np.mean(cell))
        
        # Check if there's significant variation in brightness (chess pattern)
        brightness_std = np.std(brightness_values)
        brightness_range = np.max(brightness_values) - np.min(brightness_values)
        
        # Much stricter requirements for chess pattern
        # Real chess boards should have very high contrast and clear alternating pattern
        if brightness_std > 40 and brightness_range > 100:
            # Additional check: verify alternating pattern
            # Check if adjacent squares have significantly different brightness
            alternating_score = 0
            total_comparisons = 0
            
            for i in range(7):  # 7x7 comparisons
                for j in range(7):
                    current_idx = i * 8 + j
                    right_idx = i * 8 + (j + 1)
                    bottom_idx = (i + 1) * 8 + j
                    
                    # Compare with right neighbor
                    diff_right = abs(brightness_values[current_idx] - brightness_values[right_idx])
                    if diff_right > 30:  # Significant difference
                        alternating_score += 1
                    total_comparisons += 1
                    
                    # Compare with bottom neighbor  
                    diff_bottom = abs(brightness_values[current_idx] - brightness_values[bottom_idx])
                    if diff_bottom > 30:  # Significant difference
                        alternating_score += 1
                    total_comparisons += 1
            
            alternating_ratio = alternating_score / total_comparisons
            
            if alternating_ratio > 0.3:  # At least 30% of adjacent squares differ significantly
                logger.debug(f"Found strong alternating pattern: std={brightness_std:.1f}, range={brightness_range:.1f}, alternating_ratio={alternating_ratio:.2f}")
                return True
            else:
                logger.debug(f"Weak alternating pattern: std={brightness_std:.1f}, range={brightness_range:.1f}, alternating_ratio={alternating_ratio:.2f}")
        else:
            logger.debug(f"Insufficient contrast for chess pattern: std={brightness_std:.1f}, range={brightness_range:.1f}")
        
        return False
    
    def _validate_strong_chess_pattern(self, image: np.ndarray) -> bool:
        """
        Additional validation for regions that only have alternating patterns.
        This helps reject false positives that have contrast but aren't chess boards.
        """
        h, w = image.shape
        cell_h, cell_w = h // 8, w // 8
        
        if cell_h < 8 or cell_w < 8:  # Need reasonable cell size
            return False
        
        brightness_values = []
        for i in range(8):
            for j in range(8):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell = image[y1:y2, x1:x2]
                brightness_values.append(np.mean(cell))
        
        brightness_std = np.std(brightness_values)
        brightness_range = np.max(brightness_values) - np.min(brightness_values)
        
        # MUCH stricter requirements for alternating-only validation
        if brightness_std < 50 or brightness_range < 120:
            logger.debug(f"Insufficient contrast: std={brightness_std:.1f}, range={brightness_range:.1f}")
            return False
        
        # Check for proper checkerboard alternating pattern
        # In a real chess board, adjacent squares should frequently differ
        alternating_score = 0
        total_comparisons = 0
        
        for i in range(7):
            for j in range(7):
                current_idx = i * 8 + j
                right_idx = i * 8 + (j + 1)
                bottom_idx = (i + 1) * 8 + j
                
                # Compare with right neighbor
                diff_right = abs(brightness_values[current_idx] - brightness_values[right_idx])
                if diff_right > 40:  # Higher threshold
                    alternating_score += 1
                total_comparisons += 1
                
                # Compare with bottom neighbor  
                diff_bottom = abs(brightness_values[current_idx] - brightness_values[bottom_idx])
                if diff_bottom > 40:  # Higher threshold
                    alternating_score += 1
                total_comparisons += 1
        
        alternating_ratio = alternating_score / total_comparisons
        
        # Need very high alternating ratio for alternating-only validation
        if alternating_ratio < 0.4:  # At least 40% of adjacent squares must differ significantly
            logger.debug(f"Insufficient alternating pattern: ratio={alternating_ratio:.2f}")
            return False
        
        # Additional check: verify the pattern has both light and dark regions
        sorted_brightness = sorted(brightness_values)
        median_brightness = sorted_brightness[32]  # Middle value
        
        light_squares = sum(1 for b in brightness_values if b > median_brightness + 20)
        dark_squares = sum(1 for b in brightness_values if b < median_brightness - 20)
        
        # Should have roughly equal numbers of light and dark squares
        if light_squares < 20 or dark_squares < 20:
            logger.debug(f"Unbalanced light/dark squares: light={light_squares}, dark={dark_squares}")
            return False
        
        logger.debug(f"Strong chess pattern validated: std={brightness_std:.1f}, range={brightness_range:.1f}, alternating={alternating_ratio:.2f}, light={light_squares}, dark={dark_squares}")
        return True
    
    def _score_chess_candidate(self, gray: np.ndarray, corners: np.ndarray) -> float:
        """
        Score a candidate region based on how likely it is to be a chess board.
        Higher scores indicate better candidates.
        
        Args:
            gray: Grayscale image
            corners: Detected corner points
            
        Returns:
            Score (0-100, higher is better)
        """
        try:
            # Extract the region for analysis
            board_size = 200  # Temporary size for analysis
            dst_corners = np.array([
                [0, 0], [board_size, 0], 
                [board_size, board_size], [0, board_size]
            ], dtype=np.float32)
            
            transform_matrix = cv2.getPerspectiveTransform(corners, dst_corners)
            warped = cv2.warpPerspective(gray, transform_matrix, (board_size, board_size))
            
            score = 0.0
            
            # HIGHEST PRIORITY: Actual checkerboard pattern detection (50 points)
            if self._has_checkerboard_pattern(warped):
                score += 50
                logger.debug("✅ Checkerboard pattern detected: +50 points")
            
            # HIGH PRIORITY: Grid structure (30 points)
            if self._has_grid_structure(warped):
                score += 30
                logger.debug("✅ Grid structure detected: +30 points")
            
            # MEDIUM PRIORITY: Alternating pattern analysis (up to 20 points)
            alternating_score = self._score_alternating_pattern(warped)
            score += alternating_score
            logger.debug(f"Alternating pattern score: +{alternating_score:.1f} points")
            
            # GEOMETRIC BONUS: Aspect ratio and position (up to 10 points)
            geometric_score = self._score_geometry(corners, gray.shape)
            score += geometric_score
            logger.debug(f"Geometric score: +{geometric_score:.1f} points")
            
            logger.debug(f"Total candidate score: {score:.1f}")
            return score
            
        except Exception as e:
            logger.debug(f"Scoring failed: {e}")
            return 0.0
    
    def _score_alternating_pattern(self, image: np.ndarray) -> float:
        """Score the alternating pattern quality (0-20 points)"""
        h, w = image.shape
        cell_h, cell_w = h // 8, w // 8
        
        if cell_h < 5 or cell_w < 5:
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
        
        # Base score from contrast
        contrast_score = 0.0
        if brightness_std > 30:
            contrast_score += min(10, brightness_std / 10)  # Up to 10 points
        if brightness_range > 80:
            contrast_score += min(5, brightness_range / 40)  # Up to 5 points
        
        # Alternating pattern score
        alternating_score = 0
        total_comparisons = 0
        
        for i in range(7):
            for j in range(7):
                current_idx = i * 8 + j
                right_idx = i * 8 + (j + 1)
                bottom_idx = (i + 1) * 8 + j
                
                # Compare with right neighbor
                diff_right = abs(brightness_values[current_idx] - brightness_values[right_idx])
                if diff_right > 25:
                    alternating_score += 1
                total_comparisons += 1
                
                # Compare with bottom neighbor  
                diff_bottom = abs(brightness_values[current_idx] - brightness_values[bottom_idx])
                if diff_bottom > 25:
                    alternating_score += 1
                total_comparisons += 1
        
        alternating_ratio = alternating_score / total_comparisons
        pattern_score = alternating_ratio * 5  # Up to 5 points
        
        # Penalty for imbalanced light/dark distribution
        sorted_brightness = sorted(brightness_values)
        median_brightness = sorted_brightness[32]
        
        light_squares = sum(1 for b in brightness_values if b > median_brightness + 15)
        dark_squares = sum(1 for b in brightness_values if b < median_brightness - 15)
        
        balance_penalty = 0
        if light_squares < 15 or dark_squares < 15:
            balance_penalty = -5  # Penalty for poor balance
        
        total_score = contrast_score + pattern_score + balance_penalty
        return max(0, min(20, total_score))  # Clamp to 0-20
    
    def _score_geometry(self, corners: np.ndarray, image_shape: tuple) -> float:
        """Score geometric properties (0-10 points)"""
        try:
            # Aspect ratio score (0-5 points)
            rect = cv2.minAreaRect(corners)
            width, height = rect[1]
            if width == 0 or height == 0:
                return 0.0
            
            aspect_ratio = max(width, height) / min(width, height)
            aspect_score = max(0, 5 - abs(aspect_ratio - 1.0) * 2)  # Penalty for non-square
            
            # Position score (0-5 points) - prefer regions not at extreme edges
            img_h, img_w = image_shape
            center_x = np.mean(corners[:, 0])
            center_y = np.mean(corners[:, 1])
            
            # Normalized position (0.0 to 1.0)
            norm_x = center_x / img_w
            norm_y = center_y / img_h
            
            # Prefer regions not too close to edges (0.1 to 0.9 range is good)
            edge_distance_x = min(norm_x, 1.0 - norm_x)
            edge_distance_y = min(norm_y, 1.0 - norm_y)
            
            position_score = (min(edge_distance_x, 0.4) + min(edge_distance_y, 0.4)) * 6.25  # Up to 5 points
            
            return aspect_score + position_score
            
        except Exception:
            return 0.0
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners in consistent order: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            corners: Unordered corner points
            
        Returns:
            Ordered corner points
        """
        # Calculate center point
        center = np.mean(corners, axis=0)
        
        # Sort by angle from center
        def angle_from_center(point):
            return np.arctan2(point[1] - center[1], point[0] - center[0])
        
        # Sort corners by angle (starting from top-left, going clockwise)
        sorted_corners = sorted(corners, key=angle_from_center)
        
        # Ensure we start from top-left
        # Find the corner with minimum sum of coordinates (top-left)
        min_sum_idx = np.argmin([p[0] + p[1] for p in sorted_corners])
        
        # Reorder starting from top-left
        ordered = sorted_corners[min_sum_idx:] + sorted_corners[:min_sum_idx]
        
        return np.array(ordered, dtype=np.float32)
    
    def _extract_board(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Extract and perspective-correct the chess board region.
        
        Args:
            image: Original image
            corners: Board corner coordinates
            
        Returns:
            Perspective-corrected chess board image
        """
        # Define destination points for perspective correction (square board)
        board_size = 640  # Output board size in pixels
        dst_corners = np.array([
            [0, 0],
            [board_size, 0], 
            [board_size, board_size],
            [0, board_size]
        ], dtype=np.float32)
        
        # Calculate perspective transformation matrix
        transform_matrix = cv2.getPerspectiveTransform(corners, dst_corners)
        
        # Apply perspective correction
        board = cv2.warpPerspective(image, transform_matrix, (board_size, board_size))
        
        return board