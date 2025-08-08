"""
Clean Edge-based Chess Board Detector

A simplified version of the edge-based detector that uses YAML configuration
and focuses on detection without extensive logging.
"""

import cv2
import numpy as np
import yaml
from typing import Optional, Tuple, List, Dict, Any
from collections import Counter
import scipy
from numba import njit, prange

class CleanEdgeBasedDetector:
    """
    Clean implementation of edge-based chess board detection.
    
    Loads all hyperparameters from a YAML configuration file and provides
    a simple interface that returns either a board or None.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the detector with configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def detect_board(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and extract chess board from image using optimized distance computations.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Extracted chess board image or None if detection failed
        """
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Apply edge detection
        edges = self._detect_edges(gray)
        
        # Step 3: Find lines using HoughLinesP 
        lines = self._find_lines(edges)
        if lines is None or len(lines) == 0:
            return None
        
        # Step 5: Filter lines by angle (vertical vs horizontal)
        vertical_lines, horizontal_lines = self._filter_lines_by_angle(lines)
        
        # Step 6: Precompute line data
        line_data = self._precompute_line_data(vertical_lines, horizontal_lines)
        if not line_data['all_distances']:
            return None
        
        # Step 8: Find frequent distances 
        frequent_distances = self._find_frequent_distances(line_data['all_distances'])
        if not frequent_distances:
            return None
        
        # Step 9: Validate distance multiples
        valid_square_side_length = self._find_valid_square_size_optimized(line_data, frequent_distances)
        if valid_square_side_length is None:
            return None
        
        # Step 10: Select lines matching chess square pattern (using precomputed data)
        selected_vertical_lines = self._select_lines_matching_pattern_optimized(
            vertical_lines, line_data['v_coords'], line_data['v_distance_matrix'], 'vertical', valid_square_side_length)
        selected_horizontal_lines = self._select_lines_matching_pattern_optimized(
            horizontal_lines, line_data['h_coords'], line_data['h_distance_matrix'], 'horizontal', valid_square_side_length)
        
        # Step 11: Filter lines by boundaries of the other direction (iteratively)
        if len(selected_vertical_lines) > 0 and len(selected_horizontal_lines) > 0:
            buffer_pixels = self.config['line_selection']['boundary_buffer_pixels']
            
            for iteration in range(10):
                prev_h_count = len(selected_horizontal_lines)
                prev_v_count = len(selected_vertical_lines)
                
                selected_horizontal_lines, selected_vertical_lines = self._filter_lines_by_boundaries(
                    selected_horizontal_lines, selected_vertical_lines, buffer_pixels)
                
                if len(selected_horizontal_lines) == prev_h_count and len(selected_vertical_lines) == prev_v_count:
                    break
        
        # Step 12: Smart duplicate removal (at the end)
        distance_threshold = self.config['duplicate_removal']['distance_threshold']
        if len(selected_vertical_lines) > 0:
            selected_vertical_lines = self._remove_smart_duplicates_optimized(
                selected_vertical_lines, 'vertical', valid_square_side_length, distance_threshold)
        if len(selected_horizontal_lines) > 0:
            selected_horizontal_lines = self._remove_smart_duplicates_optimized(
                selected_horizontal_lines, 'horizontal', valid_square_side_length, distance_threshold)
        
        # Final validation - check if we have enough lines
        min_lines = self.config['board_validation']['min_lines_required']
        if len(selected_vertical_lines) < min_lines or len(selected_horizontal_lines) < min_lines:
            return None
        
        # Calculate board corners from line intersections
        corners = self._calculate_board_corners(selected_vertical_lines, selected_horizontal_lines)
        if corners is None:
            return None
        
        # Step 14.5: Adjust corners if board is not exactly 8x8
        adjusted_corners = self._adjust_corners_for_8x8(corners, selected_vertical_lines, selected_horizontal_lines, valid_square_side_length)
        if adjusted_corners is not None:
            corners = adjusted_corners
        
        # Step 15: Validate corner geometry (but don't fail on violations, just like original)
        corner_validation = self._validate_corner_geometry_detailed(corners, valid_square_side_length)
        
        # Step 15.5: Try to fix violations using iterative board adjustment
        if corner_validation['has_violations']:
            adjusted_corners_result = self._iterative_board_adjustment_with_corners(
                image, corners, valid_square_side_length
            )
            if adjusted_corners_result is not None:
                adjusted_validation = self._validate_corner_geometry_detailed(adjusted_corners_result, valid_square_side_length)
                if adjusted_validation['violation_count'] < corner_validation['violation_count']:
                    corners = adjusted_corners_result
        
        # Extract and return the board using final corners
        board = self._extract_board(image, corners)
        return board
 
    def detect_board_and_corners(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect board and also return its corners.
        Returns (board_image, corners) where corners are float32 [[x,y], ...] in TL, TR, BR, BL order.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = self._detect_edges(gray)
        lines = self._find_lines(edges)
        if lines is None or len(lines) == 0:
            return None, None
        vertical_lines, horizontal_lines = self._filter_lines_by_angle(lines)
        line_data = self._precompute_line_data(vertical_lines, horizontal_lines)
        if not line_data['all_distances']:
            return None, None
        frequent_distances = self._find_frequent_distances(line_data['all_distances'])
        if not frequent_distances:
            return None, None
        valid_square_side_length = self._find_valid_square_size_optimized(line_data, frequent_distances)
        if valid_square_side_length is None:
            return None, None
        selected_vertical_lines = self._select_lines_matching_pattern_optimized(
            vertical_lines, line_data['v_coords'], line_data['v_distance_matrix'], 'vertical', valid_square_side_length)
        selected_horizontal_lines = self._select_lines_matching_pattern_optimized(
            horizontal_lines, line_data['h_coords'], line_data['h_distance_matrix'], 'horizontal', valid_square_side_length)
        if len(selected_vertical_lines) > 0 and len(selected_horizontal_lines) > 0:
            buffer_pixels = self.config['line_selection']['boundary_buffer_pixels']
            for _ in range(10):
                prev_h_count = len(selected_horizontal_lines)
                prev_v_count = len(selected_vertical_lines)
                selected_horizontal_lines, selected_vertical_lines = self._filter_lines_by_boundaries(
                    selected_horizontal_lines, selected_vertical_lines, buffer_pixels)
                if len(selected_horizontal_lines) == prev_h_count and len(selected_vertical_lines) == prev_v_count:
                    break
        distance_threshold = self.config['duplicate_removal']['distance_threshold']
        if len(selected_vertical_lines) > 0:
            selected_vertical_lines = self._remove_smart_duplicates_optimized(
                selected_vertical_lines, 'vertical', valid_square_side_length, distance_threshold)
        if len(selected_horizontal_lines) > 0:
            selected_horizontal_lines = self._remove_smart_duplicates_optimized(
                selected_horizontal_lines, 'horizontal', valid_square_side_length, distance_threshold)
        min_lines = self.config['board_validation']['min_lines_required']
        if len(selected_vertical_lines) < min_lines or len(selected_horizontal_lines) < min_lines:
            return None, None
        corners = self._calculate_board_corners(selected_vertical_lines, selected_horizontal_lines)
        if corners is None:
            return None, None
        adjusted_corners = self._adjust_corners_for_8x8(corners, selected_vertical_lines, selected_horizontal_lines, valid_square_side_length)
        if adjusted_corners is not None:
            corners = adjusted_corners
        corner_validation = self._validate_corner_geometry_detailed(corners, valid_square_side_length)
        if corner_validation['has_violations']:
            adjusted_corners_result = self._iterative_board_adjustment_with_corners(image, corners, valid_square_side_length)
            if adjusted_corners_result is not None:
                adjusted_validation = self._validate_corner_geometry_detailed(adjusted_corners_result, valid_square_side_length)
                if adjusted_validation['violation_count'] < corner_validation['violation_count']:
                    corners = adjusted_corners_result
        board = self._extract_board(image, corners)
        return board, corners.astype(np.float32)

    def extract_board_with_corners(self, image: np.ndarray, corners: np.ndarray) -> Optional[np.ndarray]:
        """Public helper to extract the board using provided corners."""
        if corners is None:
            return None
        return self._extract_board(image, corners.astype(np.float32))
    
    def _adjust_corners_for_8x8(self, corners: np.ndarray, vertical_lines: np.ndarray, horizontal_lines: np.ndarray, square_side_length: float) -> Optional[np.ndarray]:
        """Adjust corners for optimal 8x8 board extraction - simplified version."""
        # Simple implementation - just return None to skip adjustment
        return None
    
    def _validate_corner_geometry_detailed(self, corners: np.ndarray, square_side_length: float) -> dict:
        """Detailed corner geometry validation that returns violation info."""
        tolerance_percent = self.config['board_validation']['corner_geometry_tolerance_percent']
        angle_tolerance = self.config['board_validation']['angle_tolerance_degrees']
        
        expected_size = 8 * square_side_length
        tolerance = expected_size * (tolerance_percent / 100.0)
        
        violations = []
        
        # Check side lengths
        top_left, top_right, bottom_right, bottom_left = corners
        sides = [
            np.linalg.norm(top_right - top_left),
            np.linalg.norm(bottom_right - top_right),
            np.linalg.norm(bottom_left - bottom_right),
            np.linalg.norm(top_left - bottom_left)
        ]
        
        for i, side in enumerate(sides):
            if abs(side - expected_size) > tolerance:
                violations.append(f"Side {i} length {side:.1f} deviates from expected {expected_size:.1f} by {abs(side - expected_size):.1f} pixels")
        
        # Check angles (should be ~90 degrees)
        def calculate_angle(p1, vertex, p2):
            v1, v2 = p1 - vertex, p2 - vertex
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return np.degrees(np.arccos(cos_angle))
        
        angles = [
            calculate_angle(bottom_left, top_left, top_right),
            calculate_angle(top_left, top_right, bottom_right),
            calculate_angle(top_right, bottom_right, bottom_left),
            calculate_angle(bottom_right, bottom_left, top_left)
        ]
        
        for i, angle in enumerate(angles):
            if abs(angle - 90.0) > angle_tolerance:
                violations.append(f"Angle {i} is {angle:.1f}°, deviates from 90° by {abs(angle - 90.0):.1f}°")
        
        return {
            'has_violations': len(violations) > 0,
            'violation_count': len(violations),
            'violations': violations
        }
    
    def _iterative_board_adjustment_with_corners(self, image: np.ndarray, corners: np.ndarray, square_side_length: float) -> Optional[np.ndarray]:
        """Try to adjust corners iteratively - simplified version."""
        # Simple implementation - just return None to skip adjustment
        return None
    
    def _precompute_line_data(self, vertical_lines: np.ndarray, horizontal_lines: np.ndarray) -> dict:
        """Precompute line coordinates and distance matrices using Numba for efficiency."""
        # Extract coordinates
        v_coords = np.mean(vertical_lines[:, [0, 2]], axis=1) if len(vertical_lines) > 0 else np.array([], dtype=np.float64)
        h_coords = np.mean(horizontal_lines[:, [1, 3]], axis=1) if len(horizontal_lines) > 0 else np.array([], dtype=np.float64)
        
        # Compute distance matrices and condensed vectors via Numba
        if v_coords.size >= 2:
            v_distance_matrix, v_condensed = self.distance_numba_half(v_coords.astype(np.float64))
        else:
            v_distance_matrix = np.empty((0, 0), dtype=np.float64)
            v_condensed = np.empty((0,), dtype=np.float64)
        if h_coords.size >= 2:
            h_distance_matrix, h_condensed = self.distance_numba_half(h_coords.astype(np.float64))
        else:
            h_distance_matrix = np.empty((0, 0), dtype=np.float64)
            h_condensed = np.empty((0,), dtype=np.float64)
        
        all_distances: List[float] = []
        if v_condensed.size > 0:
            all_distances.extend(v_condensed.tolist())
        if h_condensed.size > 0:
            all_distances.extend(h_condensed.tolist())
        
        return {
            'v_coords': v_coords,
            'h_coords': h_coords,
            'v_distance_matrix': v_distance_matrix,
            'h_distance_matrix': h_distance_matrix,
            'all_distances': all_distances
        }
    
    def _find_valid_square_size_optimized(self, line_data: dict, frequent_distances: List[float]) -> Optional[float]:
        """Find valid chess square side length using precomputed distance data."""
        min_square_size = self.config['distance_analysis']['min_square_size']
        tolerance = self.config['distance_analysis']['validation_tolerance_pixels']
        
        valid_candidates = []
        distances_array = np.array(line_data['all_distances'])
        
        for candidate in frequent_distances:
            if self._validate_distance_multiples_optimized(distances_array, candidate, tolerance):
                valid_candidates.append(candidate)
        
        # Prefer reasonable sizes
        reasonable_candidates = [c for c in valid_candidates if c >= min_square_size]
        if reasonable_candidates:
            return reasonable_candidates[0]  # Smallest reasonable candidate
        elif valid_candidates:
            return valid_candidates[-1]  # Largest candidate if none are reasonable
        
        return None
    
    def _validate_distance_multiples_optimized(self, distances: np.ndarray, candidate: float, tolerance: float) -> bool:
        """Validate that multiples of candidate distance appear in the distances using vectorized operations."""
        expected_multiples = [2, 3, 4]
        found_multiples = 0
        
        for multiple in expected_multiples:
            expected_distance = candidate * multiple
            if np.any(np.abs(distances - expected_distance) <= tolerance):
                found_multiples += 1
        
        return found_multiples >= 2
    
    def _select_lines_matching_pattern_optimized(self, lines: np.ndarray, coords: np.ndarray, 
                                                 distance_matrix: np.ndarray, direction: str, 
                                                 square_side_length: float) -> np.ndarray:
        """Select lines that match chess square spacing pattern using precomputed distance matrix."""
        if len(lines) == 0:
            return np.array([])
        
        tolerance = self.config['line_selection']['tolerance_pixels']
        tolerance_mult = self.config['line_selection']['tolerance_multiplier']
        min_matching = self.config['line_selection']['min_matching_distances']
        min_small_matching = self.config['line_selection']['min_small_matching_distances']
        
        # Sort lines by coordinate
        sorted_indices = np.argsort(coords)
        sorted_lines = lines[sorted_indices]
        sorted_coords = coords[sorted_indices]
        
        selected_lines = []
        
        for i, (line, coord) in enumerate(zip(sorted_lines, sorted_coords)):
            # Get distances from precomputed matrix
            distances = distance_matrix[sorted_indices[i]] if distance_matrix.size > 0 else np.array([])
            matching_distances = 0
            small_matching_distances = 0
            
            for distance in distances:
                if distance == 0:
                    continue
                
                multiple = round(distance / square_side_length)
                expected_distance = multiple * square_side_length
                
                if multiple > 0 and abs(distance - expected_distance) <= multiple * tolerance_mult * tolerance:
                    matching_distances += 1
                    if multiple <= 3:
                        small_matching_distances += 1
            
            if matching_distances >= min_matching and small_matching_distances >= min_small_matching:
                selected_lines.append(line)
        
        return np.array(selected_lines)
    
    def _remove_smart_duplicates_optimized(self, lines: np.ndarray, direction: str, 
                                         square_side_length: float, distance_threshold: float = 5.0) -> np.ndarray:
        """Remove duplicate lines intelligently using optimized distance computation."""
        if len(lines) == 0:
            return lines
        
        # Extract coordinates
        if direction == 'vertical':
            coords = np.array([(line[0] + line[2]) / 2 for line in lines])
        else:
            coords = np.array([(line[1] + line[3]) / 2 for line in lines])
        
        # Compute distance matrix once
        distance_matrix = np.abs(coords[:, np.newaxis] - coords)
        
        unique_lines = []
        used_indices = set()
        
        for i, line in enumerate(lines):
            if i in used_indices:
                continue
            
            # Find close lines using precomputed distances
            close_indices = []
            for j in range(len(lines)):
                if j != i and j not in used_indices and distance_matrix[i, j] <= distance_threshold:
                    close_indices.append(j)
            
            if not close_indices:
                unique_lines.append(line)
                used_indices.add(i)
            else:
                # Choose best line based on spacing quality
                candidates = [i] + close_indices
                best_idx = self._evaluate_line_spacing_quality_optimized(
                    coords[candidates], square_side_length, coords, distance_matrix, candidates)
                
                unique_lines.append(lines[candidates[best_idx]])
                for idx in candidates:
                    used_indices.add(idx)
        
        return np.array(unique_lines)
    
    def _evaluate_line_spacing_quality_optimized(self, candidate_coords: np.ndarray, 
                                               square_side_length: float, all_coords: np.ndarray,
                                               distance_matrix: np.ndarray, candidate_indices: List[int]) -> int:
        """Evaluate which line has best spacing alignment using precomputed distances."""
        best_score = float('inf')
        best_index = 0
        
        for i, candidate_idx in enumerate(candidate_indices):
            # Get distances from precomputed matrix
            distances = distance_matrix[candidate_idx]
            spacing_errors = []
            
            for distance in distances:
                if distance == 0:
                    continue
                
                multiple = round(distance / square_side_length)
                if multiple > 0:
                    expected_distance = multiple * square_side_length
                    error = abs(distance - expected_distance)
                    spacing_errors.append(error)
            
            if spacing_errors:
                average_error = np.mean(spacing_errors)
                if average_error < best_score:
                    best_score = average_error
                    best_index = i
        
        return best_index
    
    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection with adaptive parameters."""
        # Apply Gaussian blur
        kernel_size = tuple(self.config['edge_detection']['gaussian_blur']['kernel_size'])
        sigma = self.config['edge_detection']['gaussian_blur']['sigma']
        blurred = cv2.GaussianBlur(gray, kernel_size, sigma)
        
        # Calculate adaptive thresholds
        median = np.median(blurred)
        lower = int(max(0, 0.5 * median))
        upper = int(min(255, 0.9 * median))
        
        # Apply Canny edge detection
        aperture_size = self.config['edge_detection']['canny']['aperture_size']
        edges = cv2.Canny(blurred, lower, upper, apertureSize=aperture_size)
        
        return edges
    
    def _find_lines(self, edges: np.ndarray) -> Optional[np.ndarray]:
        """Find lines using HoughLinesP."""
        params = self.config['line_detection']['hough_lines']
        
        lines = cv2.HoughLinesP(
            edges,
            rho=params['rho'],
            theta=np.pi / params['theta_resolution'],
            threshold=params['threshold'],
            minLineLength=params['min_line_length'],
            maxLineGap=params['max_line_gap']
        )
        
        if lines is None:
            return None
        
        return lines.reshape(-1, 4)
    
    def _filter_lines_by_angle(self, lines: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filter lines into vertical and horizontal based on angle."""
        vertical_lines = []
        horizontal_lines = []
        angle_tolerance = self.config['line_filtering']['angle_tolerance']
        
        for line in lines:
            x1, y1, x2, y2 = line
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if 90 - angle_tolerance < angle < 90 + angle_tolerance:
                vertical_lines.append(line)
            elif angle < angle_tolerance or angle > 180 - angle_tolerance:
                horizontal_lines.append(line)
        
        return np.array(vertical_lines), np.array(horizontal_lines)
    
    @staticmethod
    @njit(fastmath=True)
    def distance_numba_half(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return symmetric distance matrix (upper triangular filled) and condensed upper-tri distances.
        The condensed vector is in row-major order over the upper triangle (i<j).
        """
        n = coords.size
        # Handle empty or single-element cases explicitly
        if n == 0:
            return np.empty((0, 0), dtype=coords.dtype), np.empty((0,), dtype=coords.dtype)
        if n == 1:
            return np.zeros((1, 1), dtype=coords.dtype), np.empty((0,), dtype=coords.dtype)
        
        matrix = np.zeros((n, n), dtype=coords.dtype)
        condensed = np.empty(n * (n - 1) // 2, dtype=coords.dtype)
        k = 0
        for i in prange(n - 1):
            for j in range(i + 1, n):
                d = abs(coords[j] - coords[i])
                matrix[i, j] = d
                matrix[j, i] = d
                condensed[k] = d
                k += 1
        return matrix, condensed
    
    def _find_frequent_distances(self, distances: List[float]) -> List[float]:
        """Find frequently occurring distances."""
        if not distances:
            return []
        
        tolerance = self.config['distance_analysis']['tolerance_pixels']
        min_instances = self.config['distance_analysis']['min_instances']
        
        sorted_distances = sorted(distances)
        distance_groups = []
        current_group = [sorted_distances[0]]
        
        for i in range(1, len(sorted_distances)):
            dist = sorted_distances[i]
            # Use mode of current group as center
            values, counts = np.unique(current_group, return_counts=True)
            group_center = values[np.argmax(counts)]
            
            if abs(dist - group_center) <= tolerance:
                current_group.append(dist)
            else:
                distance_groups.append(current_group)
                current_group = [dist]
        
        distance_groups.append(current_group)
        
        # Get frequent distances (mode of each group)
        frequent_distances = []
        for group in distance_groups:
            if len(group) >= min_instances:
                values, counts = np.unique(group, return_counts=True)
                mode = values[np.argmax(counts)]
                frequent_distances.append(mode)
        
        return sorted(frequent_distances)
    
    def _find_valid_square_size(self, all_distances: List[float], frequent_distances: List[float]) -> Optional[float]:
        """Find valid chess square side length."""
        min_square_size = self.config['distance_analysis']['min_square_size']
        tolerance = self.config['distance_analysis']['validation_tolerance_pixels']
        
        valid_candidates = []
        distances_array = np.array(all_distances)
        
        for candidate in frequent_distances:
            if self._validate_distance_multiples(distances_array, candidate, tolerance):
                valid_candidates.append(candidate)
        
        # Prefer reasonable sizes
        reasonable_candidates = [c for c in valid_candidates if c >= min_square_size]
        if reasonable_candidates:
            return reasonable_candidates[0]  # Smallest reasonable candidate
        elif valid_candidates:
            return valid_candidates[-1]  # Largest candidate if none are reasonable
        
        return None
    
    def _validate_distance_multiples(self, distances: np.ndarray, candidate: float, tolerance: float) -> bool:
        """Validate that multiples of candidate distance appear in the distances."""
        expected_multiples = [2, 3, 4]
        found_multiples = 0
        
        for multiple in expected_multiples:
            expected_distance = candidate * multiple
            close_distances = distances[np.abs(distances - expected_distance) <= tolerance]
            if len(close_distances) > 0:
                found_multiples += 1
        
        return found_multiples >= 2
    
    def _select_lines_matching_pattern(self, lines: np.ndarray, direction: str, square_side_length: float) -> np.ndarray:
        """Select lines that match chess square spacing pattern."""
        if len(lines) == 0:
            return np.array([])
        
        tolerance = self.config['line_selection']['tolerance_pixels']
        tolerance_mult = self.config['line_selection']['tolerance_multiplier']
        min_matching = self.config['line_selection']['min_matching_distances']
        min_small_matching = self.config['line_selection']['min_small_matching_distances']
        
        # Extract and sort coordinates
        if direction == 'vertical':
            coords = [(line[0] + line[2]) / 2 for line in lines]
        else:
            coords = [(line[1] + line[3]) / 2 for line in lines]
        
        coords_array = np.array(coords)
        sorted_indices = np.argsort(coords_array)
        sorted_lines = lines[sorted_indices]
        sorted_coords = coords_array[sorted_indices]
        
        selected_lines = []
        
        for line, coord in zip(sorted_lines, sorted_coords):
            distances = np.abs(sorted_coords - coord)
            matching_distances = 0
            small_matching_distances = 0
            
            for distance in distances:
                if distance == 0:
                    continue
                
                multiple = round(distance / square_side_length)
                expected_distance = multiple * square_side_length
                
                if multiple > 0 and abs(distance - expected_distance) <= multiple * tolerance_mult * tolerance:
                    matching_distances += 1
                    if multiple <= 3:
                        small_matching_distances += 1
            
            if matching_distances >= min_matching and small_matching_distances >= min_small_matching:
                selected_lines.append(line)
        
        return np.array(selected_lines)
    
    def _filter_lines_by_boundaries(self, horizontal_lines: np.ndarray, vertical_lines: np.ndarray, 
                                   buffer_pixels: float) -> Tuple[np.ndarray, np.ndarray]:
        """Filter lines by boundaries of the other direction."""
        if len(vertical_lines) == 0 or len(horizontal_lines) == 0:
            return horizontal_lines, vertical_lines
        
        # Get boundaries
        v_coords = [(line[0] + line[2]) / 2 for line in vertical_lines]
        min_x, max_x = min(v_coords) - buffer_pixels, max(v_coords) + buffer_pixels
        
        h_coords = [(line[1] + line[3]) / 2 for line in horizontal_lines]
        min_y, max_y = min(h_coords) - buffer_pixels, max(h_coords) + buffer_pixels
        
        # Filter horizontal lines
        filtered_horizontal = []
        for line in horizontal_lines:
            line_x_min, line_x_max = min(line[0], line[2]), max(line[0], line[2])
            if not (line_x_max < min_x or line_x_min > max_x):
                filtered_horizontal.append(line)
        
        # Filter vertical lines
        filtered_vertical = []
        for line in vertical_lines:
            line_y_min, line_y_max = min(line[1], line[3]), max(line[1], line[3])
            if not (line_y_max < min_y or line_y_min > max_y):
                filtered_vertical.append(line)
        
        return np.array(filtered_horizontal), np.array(filtered_vertical)
    
    def _remove_smart_duplicates(self, lines: np.ndarray, direction: str, square_side_length: float, distance_threshold: float = 5.0) -> np.ndarray:
        """Remove duplicate lines intelligently."""
        if len(lines) == 0:
            return lines
        
        # Extract coordinates
        if direction == 'vertical':
            coords = [(line[0] + line[2]) / 2 for line in lines]
        else:
            coords = [(line[1] + line[3]) / 2 for line in lines]
        
        coords_array = np.array(coords)
        unique_lines = []
        used_indices = set()
        
        for i, (line, coord) in enumerate(zip(lines, coords_array)):
            if i in used_indices:
                continue
            
            # Find close lines
            close_indices = []
            for j, other_coord in enumerate(coords_array):
                if j != i and j not in used_indices and abs(coord - other_coord) <= distance_threshold:
                    close_indices.append(j)
            
            if not close_indices:
                unique_lines.append(line)
                used_indices.add(i)
            else:
                # Choose best line based on spacing quality
                candidates = [i] + close_indices
                best_idx = self._evaluate_line_spacing_quality(
                    coords_array[candidates], square_side_length, coords_array)
                
                unique_lines.append(lines[candidates[best_idx]])
                for idx in candidates:
                    used_indices.add(idx)
        
        return np.array(unique_lines)
    
    def _evaluate_line_spacing_quality(self, candidate_coords: np.ndarray, 
                                     square_side_length: float, all_coords: np.ndarray) -> int:
        """Evaluate which line has best spacing alignment."""
        best_score = float('inf')
        best_index = 0
        
        for i, coord in enumerate(candidate_coords):
            distances = np.abs(all_coords - coord)
            spacing_errors = []
            
            for distance in distances:
                if distance == 0:
                    continue
                
                multiple = round(distance / square_side_length)
                if multiple > 0:
                    expected_distance = multiple * square_side_length
                    error = abs(distance - expected_distance)
                    spacing_errors.append(error)
            
            if spacing_errors:
                average_error = np.mean(spacing_errors)
                if average_error < best_score:
                    best_score = average_error
                    best_index = i
        
        return best_index
    
    def _calculate_board_corners(self, vertical_lines: np.ndarray, horizontal_lines: np.ndarray) -> Optional[np.ndarray]:
        """Calculate board corners from outermost lines."""
        if len(vertical_lines) < 2 or len(horizontal_lines) < 2:
            return None
        
        # Get outermost lines
        v_coords = [(line[0] + line[2]) / 2 for line in vertical_lines]
        h_coords = [(line[1] + line[3]) / 2 for line in horizontal_lines]
        
        min_v_idx = np.argmin(v_coords)
        max_v_idx = np.argmax(v_coords)
        min_h_idx = np.argmin(h_coords)
        max_h_idx = np.argmax(h_coords)
        
        left_line = vertical_lines[min_v_idx]
        right_line = vertical_lines[max_v_idx]
        top_line = horizontal_lines[min_h_idx]
        bottom_line = horizontal_lines[max_h_idx]
        
        # Calculate intersections
        corners = []
        for v_line, h_line in [(left_line, top_line), (right_line, top_line), 
                              (right_line, bottom_line), (left_line, bottom_line)]:
            intersection = self._line_intersection(v_line, h_line)
            if intersection is None:
                return None
            corners.append(intersection)
        
        return np.array(corners, dtype=np.float32)
    
    def _line_intersection(self, line1: np.ndarray, line2: np.ndarray) -> Optional[np.ndarray]:
        """Calculate intersection point of two lines."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return np.array([x, y])
    
    def _validate_corner_geometry(self, corners: np.ndarray, square_side_length: float) -> bool:
        """Validate that corners form a reasonable square."""
        tolerance_percent = self.config['board_validation']['corner_geometry_tolerance_percent']
        angle_tolerance = self.config['board_validation']['angle_tolerance_degrees']
        
        expected_size = 8 * square_side_length
        tolerance = expected_size * (tolerance_percent / 100.0)
        
        # Check side lengths
        top_left, top_right, bottom_right, bottom_left = corners
        sides = [
            np.linalg.norm(top_right - top_left),
            np.linalg.norm(bottom_right - top_right),
            np.linalg.norm(bottom_left - bottom_right),
            np.linalg.norm(top_left - bottom_left)
        ]
        
        for side in sides:
            if abs(side - expected_size) > tolerance:
                return False
        
        # Check angles (should be ~90 degrees)
        def calculate_angle(p1, vertex, p2):
            v1, v2 = p1 - vertex, p2 - vertex
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return np.degrees(np.arccos(cos_angle))
        
        angles = [
            calculate_angle(bottom_left, top_left, top_right),
            calculate_angle(top_left, top_right, bottom_right),
            calculate_angle(top_right, bottom_right, bottom_left),
            calculate_angle(bottom_right, bottom_left, top_left)
        ]
        
        for angle in angles:
            if abs(angle - 90.0) > angle_tolerance:
                return False
        
        return True
    
    def _extract_board(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Extract the chess board using perspective transformation."""
        board_size = self.config['board_extraction']['output_size']
        
        target_corners = np.array([
            [0, 0],
            [board_size, 0],
            [board_size, board_size],
            [0, board_size]
        ], dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(corners, target_corners)
        board = cv2.warpPerspective(image, matrix, (board_size, board_size))
        
        return board