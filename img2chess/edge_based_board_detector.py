"""
Edge-based Chess Board Detector - Updated based on notebook analysis

This module implements a refined approach for detecting chess boards in images
using edge detection and line filtering techniques based on detailed step-by-step analysis.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import os
import logging
from collections import Counter

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

class EdgeBasedChessBoardDetector:
    """
    Detects chess boards using edge detection and line filtering.
    
    This approach finds all edges in the image, filters for nearly vertical/horizontal
    lines, then identifies equally spaced sets of lines in each direction to
    determine the board corners.
    """
    
    def __init__(self, debug_mode: bool = False, debug_output_dir: str = "debug_output"):
        """
        Initialize the detector.
        
        Args:
            debug_mode: If True, save intermediate steps for debugging
            debug_output_dir: Directory to save debug output
        """
        self.debug_mode = debug_mode
        self.debug_output_dir = debug_output_dir
    
    def detect_board(self, image: np.ndarray, frame_name: str, return_log: bool = False) -> Optional[np.ndarray]:
        """
        Detect and extract the chess board from an image using edge detection.
        
        Args:
            image: Input image as numpy array (BGR format)
            return_log: If True, return tuple (board, log) instead of just board
            
        Returns:
            Extracted chess board image or None if no board detected
            If return_log=True, returns tuple (board, log_dict)
        """
        # Initialize log dictionary
        log = {
            'image_shape': image.shape,
            'steps': [],
            'success': False,
            'failure_reason': None,
            'stats': {}
        }
        # Create debug directory if needed
        if self.debug_mode and not os.path.exists(os.path.join(self.debug_output_dir, frame_name)):
            os.makedirs(os.path.join(self.debug_output_dir, frame_name))
        
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        log['steps'].append({'step': 1, 'name': 'Convert to grayscale', 'status': 'completed'})
        
        if self.debug_mode:
            cv2.imwrite(os.path.join(self.debug_output_dir, frame_name, "01_grayscale.jpg"), gray)
        
        # Step 2: Apply edge detection
        edges = self._detect_edges(gray)
        edge_pixels = int(np.count_nonzero(edges))
        log['steps'].append({'step': 2, 'name': 'Edge detection', 'status': 'completed', 'edge_pixels': edge_pixels})
        log['stats']['edge_pixels'] = edge_pixels
        
        if self.debug_mode:
            cv2.imwrite(os.path.join(self.debug_output_dir, frame_name, "02_edges.jpg"), edges)
        
        # Step 3: Find lines using HoughLinesP 
        lines = self._find_lines(edges)
        if lines is None or len(lines) == 0:
            log['failure_reason'] = 'No lines detected in image'
            log['steps'].append({'step': 3, 'name': 'Find lines', 'status': 'failed', 'lines_found': 0})
            logger.warning("No lines detected in image")
            return (None, log) if return_log else None
        
        lines_count = len(lines)
        log['steps'].append({'step': 3, 'name': 'Find lines', 'status': 'completed', 'lines_found': lines_count})
        log['stats']['total_lines'] = lines_count
            
        if self.debug_mode:
            line_img = self._draw_lines(image.copy(), lines)
            cv2.imwrite(os.path.join(self.debug_output_dir, frame_name, "03_all_lines.jpg"), line_img)
        
        # Step 4: REMOVED (no early duplicate removal)
        
        # Step 5: Filter lines by angle (vertical vs horizontal)
        vertical_lines, horizontal_lines = self._filter_lines_by_angle(lines)
        rejected_lines = len(lines) - len(vertical_lines) - len(horizontal_lines)
        log['steps'].append({'step': 5, 'name': 'Filter by angle', 'status': 'completed',
                           'vertical_lines': len(vertical_lines), 'horizontal_lines': len(horizontal_lines),
                           'rejected_lines': rejected_lines})
        log['stats']['vertical_lines'] = len(vertical_lines)
        log['stats']['horizontal_lines'] = len(horizontal_lines)
        log['stats']['rejected_lines'] = rejected_lines
        
        if self.debug_mode:
            if len(vertical_lines) > 0:
                v_line_img = self._draw_lines(image.copy(), vertical_lines, color=(0, 255, 0))
                cv2.imwrite(os.path.join(self.debug_output_dir, frame_name, "05_vertical_lines.jpg"), v_line_img)
            
            if len(horizontal_lines) > 0:
                h_line_img = self._draw_lines(image.copy(), horizontal_lines, color=(255, 0, 0))
                cv2.imwrite(os.path.join(self.debug_output_dir, frame_name, "06_horizontal_lines.jpg"), h_line_img)
        
        # Step 6: REMOVED (no minimum distance filtering)
        
        # Step 7: Compute all pairwise distances
        all_distances, distance_details = self._compute_all_distances(vertical_lines, horizontal_lines)
        
        if not all_distances:
            log['failure_reason'] = 'No distances computed from lines'
            log['steps'].append({'step': 7, 'name': 'Compute distances', 'status': 'failed', 'distances_found': 0})
            logger.warning("No distances computed from lines")
            return (None, log) if return_log else None
        
        log['steps'].append({'step': 7, 'name': 'Compute distances', 'status': 'completed', 
                           'distances_found': len(all_distances), 'distance_details': distance_details})
        log['stats']['total_distances'] = len(all_distances)
        log['stats']['distance_details'] = distance_details
        if all_distances:
            log['stats']['distance_range'] = {'min': float(min(all_distances)), 'max': float(max(all_distances))}
        
        # Step 8: Find frequent distances (using mode instead of mean, pixels instead of percentage)
        frequent_distances, distance_analysis, all_distance_groups = self._find_frequent_distances(all_distances, tolerance_pixels=2.0, min_instances=15)
        
        if not frequent_distances:
            log['failure_reason'] = 'No frequent distances found with 20+ instances'
            log['steps'].append({'step': 8, 'name': 'Find frequent distances', 'status': 'failed', 
                               'frequent_distances': 0, 'distance_analysis': distance_analysis})
            logger.warning("No frequent distances found with 20+ instances")
            return (None, log) if return_log else None
        
        log['steps'].append({'step': 8, 'name': 'Find frequent distances', 'status': 'completed', 
                           'frequent_distances': len(frequent_distances), 'distance_analysis': distance_analysis})
        log['stats']['frequent_distances'] = [float(d) for d in frequent_distances]
        log['stats']['distance_analysis'] = distance_analysis
        
        # Step 9: Validate distance multiples
        valid_square_side_length = None
        valid_candidates = []
        
        # Find all valid candidates
        for candidate_distance in frequent_distances:
            if self._validate_distance_multiples(all_distances, candidate_distance, tolerance_pixels=3.0):
                valid_candidates.append(candidate_distance)
        
        # Prefer larger distances (more reasonable for chess squares)
        # A chess square should be at least 30 pixels for reasonable detection
        reasonable_candidates = [c for c in valid_candidates if c >= 30.0]
        if reasonable_candidates:
            valid_square_side_length = reasonable_candidates[0]  # First reasonable candidate (smallest among reasonable)
        elif valid_candidates:
            valid_square_side_length = valid_candidates[-1]  # If no reasonable candidates, take the largest
        
        if valid_square_side_length is None:
            log['failure_reason'] = 'No valid chess square side length found'
            log['steps'].append({'step': 9, 'name': 'Validate multiples', 'status': 'failed', 
                               'valid_candidates': len(valid_candidates)})
            logger.warning("No valid chess square side length found")
            return (None, log) if return_log else None
        
        log['steps'].append({'step': 9, 'name': 'Validate multiples', 'status': 'completed',
                           'valid_candidates': len(valid_candidates), 'square_side_length': float(valid_square_side_length)})
        log['stats']['valid_candidates'] = [float(c) for c in valid_candidates]
        log['stats']['square_side_length'] = float(valid_square_side_length)
        
        # Step 10: Select lines matching chess square pattern (completely changed)
        selected_vertical_lines, vertical_line_details = self._select_lines_matching_pattern(
            vertical_lines, 'vertical', valid_square_side_length, tolerance_pixels=2.0)
        selected_horizontal_lines, horizontal_line_details = self._select_lines_matching_pattern(
            horizontal_lines, 'horizontal', valid_square_side_length, tolerance_pixels=2.0)
        
        # Log final selection summary
        logger.info(f"\nFinal selection:")
        logger.info(f"Vertical lines: {len(selected_vertical_lines)}")
        logger.info(f"Horizontal lines: {len(selected_horizontal_lines)}")
        
        # Check if we have enough lines for a valid chess board
        min_required = 7
        vertical_sufficient = len(selected_vertical_lines) >= min_required
        horizontal_sufficient = len(selected_horizontal_lines) >= min_required
        
        logger.info(f"\nValidation:")
        logger.info(f"Vertical lines sufficient (≥{min_required}): {'✓' if vertical_sufficient else '✗'}")
        logger.info(f"Horizontal lines sufficient (≥{min_required}): {'✓' if horizontal_sufficient else '✗'}")
        
        chess_board_detected = vertical_sufficient and horizontal_sufficient
        logger.info(f"Chess board detected: {'✓ YES' if chess_board_detected else '✗ NO'}")
        
        log['steps'].append({'step': 10, 'name': 'Select matching lines', 'status': 'completed',
                           'selected_vertical': len(selected_vertical_lines), 'selected_horizontal': len(selected_horizontal_lines),
                           'vertical_line_details': vertical_line_details, 'horizontal_line_details': horizontal_line_details,
                           'chess_board_detected': chess_board_detected})
        log['stats']['selected_vertical_lines'] = len(selected_vertical_lines)
        log['stats']['selected_horizontal_lines'] = len(selected_horizontal_lines)
        log['stats']['vertical_line_details'] = vertical_line_details
        log['stats']['horizontal_line_details'] = horizontal_line_details
        
        # Step 11: Filter lines by boundaries of the other direction (iteratively)
        if len(selected_vertical_lines) > 0 and len(selected_horizontal_lines) > 0:
            original_h_count = len(selected_horizontal_lines)
            original_v_count = len(selected_vertical_lines)
            
            # Repeat filtering until no more lines are removed
            iteration = 0
            total_filtered_h = 0
            total_filtered_v = 0
            iterations_log = []
            
            while True:
                iteration += 1
                prev_h_count = len(selected_horizontal_lines)
                prev_v_count = len(selected_vertical_lines)
                
                # Filter both directions by boundaries
                selected_horizontal_lines, selected_vertical_lines = self._filter_lines_by_boundaries(
                    selected_horizontal_lines, selected_vertical_lines, buffer_pixels=200)
                
                filtered_h_this_iter = prev_h_count - len(selected_horizontal_lines)
                filtered_v_this_iter = prev_v_count - len(selected_vertical_lines)
                total_filtered_h += filtered_h_this_iter
                total_filtered_v += filtered_v_this_iter
                
                iteration_info = {
                    'iteration': iteration,
                    'filtered_horizontal': filtered_h_this_iter,
                    'filtered_vertical': filtered_v_this_iter,
                    'remaining_horizontal': len(selected_horizontal_lines),
                    'remaining_vertical': len(selected_vertical_lines)
                }
                iterations_log.append(iteration_info)
                
                logger.info(f"Boundary filtering iteration {iteration}: filtered H={filtered_h_this_iter}, V={filtered_v_this_iter}")
                logger.info(f"  Remaining: H={len(selected_horizontal_lines)}, V={len(selected_vertical_lines)}")
                
                # Stop if no lines were filtered in this iteration
                if filtered_h_this_iter == 0 and filtered_v_this_iter == 0:
                    logger.info(f"Boundary filtering converged after {iteration} iterations")
                    break
                
                # Safety check to prevent infinite loops
                if iteration >= 10:
                    logger.warning(f"Boundary filtering stopped after {iteration} iterations (safety limit)")
                    break
            
            log['steps'].append({'step': 11, 'name': 'Iterative boundary filtering', 'status': 'completed',
                               'total_iterations': iteration,
                               'total_filtered_horizontal': total_filtered_h, 'total_filtered_vertical': total_filtered_v,
                               'final_horizontal': len(selected_horizontal_lines), 'final_vertical': len(selected_vertical_lines),
                               'iterations_log': iterations_log})
            log['stats']['final_horizontal_lines'] = len(selected_horizontal_lines)
            log['stats']['final_vertical_lines'] = len(selected_vertical_lines)
            log['stats']['boundary_filtering_iterations'] = iteration
            log['stats']['boundary_filtering_log'] = iterations_log
        
        # Step 12: Smart duplicate removal (at the end)
        pre_dedup_v_count = len(selected_vertical_lines)
        pre_dedup_h_count = len(selected_horizontal_lines)
        
        if len(selected_vertical_lines) > 0:
            selected_vertical_lines = self._remove_smart_duplicates(
                selected_vertical_lines, 'vertical', valid_square_side_length, distance_threshold=5.0)
        
        if len(selected_horizontal_lines) > 0:
            selected_horizontal_lines = self._remove_smart_duplicates(
                selected_horizontal_lines, 'horizontal', valid_square_side_length, distance_threshold=5.0)
        
        dedup_v_removed = pre_dedup_v_count - len(selected_vertical_lines)
        dedup_h_removed = pre_dedup_h_count - len(selected_horizontal_lines)
        
        log['steps'].append({'step': 12, 'name': 'Smart duplicate removal', 'status': 'completed',
                           'vertical_removed': dedup_v_removed, 'horizontal_removed': dedup_h_removed,
                           'final_vertical': len(selected_vertical_lines), 'final_horizontal': len(selected_horizontal_lines)})
        log['stats']['dedup_vertical_removed'] = dedup_v_removed
        log['stats']['dedup_horizontal_removed'] = dedup_h_removed
        log['stats']['post_dedup_vertical_lines'] = len(selected_vertical_lines)
        log['stats']['post_dedup_horizontal_lines'] = len(selected_horizontal_lines)
        
        # Final validation
        if len(selected_vertical_lines) < 7 or len(selected_horizontal_lines) < 7:
            log['failure_reason'] = f'Not enough lines selected: vertical={len(selected_vertical_lines)}, horizontal={len(selected_horizontal_lines)}'
            log['steps'].append({'step': 13, 'name': 'Final validation', 'status': 'failed',
                               'final_vertical': len(selected_vertical_lines), 'final_horizontal': len(selected_horizontal_lines)})
            logger.warning(f"Not enough lines selected: vertical={len(selected_vertical_lines)}, horizontal={len(selected_horizontal_lines)}")
            return (None, log) if return_log else None
        
        if self.debug_mode:
            final_img = image.copy()
            if len(selected_vertical_lines) > 0:
                final_img = self._draw_lines(final_img, selected_vertical_lines, color=(0, 255, 0), thickness=3)
            if len(selected_horizontal_lines) > 0:
                final_img = self._draw_lines(final_img, selected_horizontal_lines, color=(0, 0, 255), thickness=3)
            cv2.imwrite(os.path.join(self.debug_output_dir, frame_name, "07_final_lines.jpg"), final_img)
        
        log['steps'].append({'step': 13, 'name': 'Final validation', 'status': 'completed',
                           'final_vertical': len(selected_vertical_lines), 'final_horizontal': len(selected_horizontal_lines)})
        
        # Calculate board corners from line intersections
        corners = self._calculate_board_corners(selected_vertical_lines, selected_horizontal_lines)
        if corners is None:
            log['failure_reason'] = 'Could not calculate board corners from lines'
            log['steps'].append({'step': 14, 'name': 'Calculate corners', 'status': 'failed'})
            logger.warning("Could not calculate board corners from lines")
            return (None, log) if return_log else None
        
        log['steps'].append({'step': 14, 'name': 'Calculate corners', 'status': 'completed'})
        log['stats']['corners'] = corners.tolist()
        
        # Step 15: Validate corner geometry
        corner_validation = self._validate_corner_geometry(corners, valid_square_side_length)
        log['steps'].append({'step': 15, 'name': 'Validate corner geometry', 'status': 'completed',
                           'corner_validation': corner_validation, 'violation_count': corner_validation['violation_count']})
        log['stats']['corner_validation'] = corner_validation
        log['stats']['geometry_violation_count'] = corner_validation['violation_count']
        
        # Log violations if any
        if corner_validation['has_violations']:
            logger.warning(f"Corner geometry violations detected ({corner_validation['violation_count']} violations):")
            for violation in corner_validation['violations']:
                logger.warning(f"  {violation}")
        else:
            logger.info(f"Corner geometry validation passed - board forms proper square (0 violations)")
            
        if self.debug_mode:
            corner_img = image.copy()
            for point in corners:
                cv2.circle(corner_img, tuple(point.astype(int)), 5, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(self.debug_output_dir, frame_name, "08_board_corners.jpg"), corner_img)
        
        # Extract and return the board
        board = self._extract_board(image, corners)
        log['steps'].append({'step': 16, 'name': 'Extract board', 'status': 'completed'})
        log['success'] = True
        log['stats']['output_board_shape'] = board.shape
        
        return (board, log) if return_log else board

    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection with adaptive parameters."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        median = np.median(blurred)
        lower = int(max(0, 0.7 * median))
        upper = int(min(255, 1.3 * median))
        edges = cv2.Canny(blurred, lower, upper, apertureSize=3)
        return edges

    def _find_lines(self, edges: np.ndarray) -> Optional[np.ndarray]:
        """Find lines using HoughLinesP with multiple parameter sets."""
        all_lines = []
        
        param_sets = [
            # {'threshold': 50, 'minLineLength': 30, 'maxLineGap': 10},
            # {'threshold': 80, 'minLineLength': 50, 'maxLineGap': 15},
            {'threshold': 100, 'minLineLength': 90, 'maxLineGap': 20},
        ]
        
        for params in param_sets:
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                **params
            )
            
            if lines is not None:
                all_lines.extend(lines.reshape(-1, 4))
        
        if not all_lines:
            return None
            
        return np.array(all_lines)

    def _remove_smart_duplicates(self, lines: np.ndarray, direction: str, square_side_length: float, distance_threshold: float = 5.0) -> np.ndarray:
        """Remove duplicate lines by keeping the one with better spacing alignment to square_side_length."""
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
                
            # Find all lines within distance_threshold of this line
            close_indices = []
            for j, other_coord in enumerate(coords_array):
                if j != i and j not in used_indices:
                    if abs(coord - other_coord) <= distance_threshold:
                        close_indices.append(j)
            
            if not close_indices:
                # No duplicates, keep this line
                unique_lines.append(line)
                used_indices.add(i)
            else:
                # Multiple candidates, evaluate which has better spacing
                candidates = [i] + close_indices
                best_index = self._evaluate_line_spacing_quality(
                    coords_array[candidates], square_side_length, coords_array
                )
                
                # Keep the best line and mark others as used
                unique_lines.append(lines[candidates[best_index]])
                for idx in candidates:
                    used_indices.add(idx)
        
        logger.info(f"Smart duplicate removal ({direction}): {len(lines)} -> {len(unique_lines)} lines")
        return np.array(unique_lines)
    
    def _evaluate_line_spacing_quality(self, candidate_coords: np.ndarray, 
                                     square_side_length: float, all_coords: np.ndarray) -> int:
        """Evaluate which line has the best spacing alignment with expected chess square pattern."""
        best_score = float('inf')
        best_index = 0
        
        for i, coord in enumerate(candidate_coords):
            # Calculate distances to all other lines
            distances = np.abs(all_coords - coord)
            
            # Evaluate how well the distances match multiples of square_side_length
            spacing_errors = []
            for distance in distances:
                if distance == 0:  # Skip self-distance
                    continue
                    
                # Find nearest multiple of square_side_length
                multiple = round(distance / square_side_length)
                if multiple > 0:
                    expected_distance = multiple * square_side_length
                    error = abs(distance - expected_distance)
                    spacing_errors.append(error)
            
            # Score is average spacing error (lower is better)
            if spacing_errors:
                average_error = np.mean(spacing_errors)
                if average_error < best_score:
                    best_score = average_error
                    best_index = i
        
        return best_index

    def _validate_corner_geometry(self, corners: np.ndarray, square_side_length: float, tolerance_percent: float = 2.0) -> dict:
        """Validate that corners form a square with sides of 8 * square_side_length."""
        expected_board_size = 8 * square_side_length
        tolerance = expected_board_size * (tolerance_percent / 100.0)
        
        # corners are in order: [top_left, top_right, bottom_right, bottom_left]
        top_left, top_right, bottom_right, bottom_left = corners
        
        # Calculate side lengths
        top_side = np.linalg.norm(top_right - top_left)
        right_side = np.linalg.norm(bottom_right - top_right)
        bottom_side = np.linalg.norm(bottom_left - bottom_right)
        left_side = np.linalg.norm(top_left - bottom_left)
        
        # Calculate diagonal lengths
        diag1 = np.linalg.norm(bottom_right - top_left)
        diag2 = np.linalg.norm(bottom_left - top_right)
        
        # Check for violations
        violations = []
        side_errors = []
        
        # Check each side length
        for side_name, side_length in [('top', top_side), ('right', right_side), 
                                     ('bottom', bottom_side), ('left', left_side)]:
            error = abs(side_length - expected_board_size)
            error_percent = (error / expected_board_size) * 100
            side_errors.append(error_percent)
            
            if error > tolerance:
                violations.append(f"{side_name} side: {side_length:.1f}px vs expected {expected_board_size:.1f}px "
                                f"(error: {error_percent:.1f}%)")
        
        # Check diagonal equality (should be equal for a proper rectangle)
        diag_diff = abs(diag1 - diag2)
        diag_error_percent = (diag_diff / max(diag1, diag2)) * 100
        if diag_error_percent > tolerance_percent:
            violations.append(f"Diagonal mismatch: {diag1:.1f}px vs {diag2:.1f}px "
                            f"(difference: {diag_error_percent:.1f}%)")
        
        # Check if it's reasonably square (diagonals should be sqrt(2) * side_length for perfect square)
        expected_diagonal = expected_board_size * np.sqrt(2)
        avg_diagonal = (diag1 + diag2) / 2
        diagonal_error = abs(avg_diagonal - expected_diagonal)
        diagonal_error_percent = (diagonal_error / expected_diagonal) * 100
        
        if diagonal_error_percent > tolerance_percent:
            violations.append(f"Diagonal length: {avg_diagonal:.1f}px vs expected {expected_diagonal:.1f}px "
                            f"(error: {diagonal_error_percent:.1f}%)")
        
        # Calculate corner angles (should be ~90 degrees)
        def calculate_angle(p1, vertex, p2):
            v1 = p1 - vertex
            v2 = p2 - vertex
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
            angle_rad = np.arccos(cos_angle)
            return np.degrees(angle_rad)
        
        corner_angles = [
            calculate_angle(bottom_left, top_left, top_right),      # top-left angle
            calculate_angle(top_left, top_right, bottom_right),     # top-right angle  
            calculate_angle(top_right, bottom_right, bottom_left), # bottom-right angle
            calculate_angle(bottom_right, bottom_left, top_left)   # bottom-left angle
        ]
        
        angle_violations = []
        for i, angle in enumerate(corner_angles):
            angle_error = abs(angle - 90.0)
            if angle_error > 2.0:  # Allow 2 degree tolerance for angles
                corner_names = ['top-left', 'top-right', 'bottom-right', 'bottom-left']
                angle_violations.append(f"{corner_names[i]} angle: {angle:.1f}° vs expected 90° "
                                      f"(error: {angle_error:.1f}°)")
        
        violations.extend(angle_violations)
        
        return {
            'expected_board_size': float(expected_board_size),
            'tolerance_pixels': float(tolerance),
            'tolerance_percent': tolerance_percent,
            'side_lengths': {
                'top': float(top_side),
                'right': float(right_side), 
                'bottom': float(bottom_side),
                'left': float(left_side)
            },
            'side_errors_percent': [float(e) for e in side_errors],
            'diagonal_lengths': {'diagonal1': float(diag1), 'diagonal2': float(diag2)},
            'diagonal_difference_percent': float(diag_error_percent),
            'corner_angles': [float(a) for a in corner_angles],
            'has_violations': len(violations) > 0,
            'violation_count': 1 if len(violations) > 0 else 0,  # Binary: 0 or 1
            'violations': violations,
            'max_side_error_percent': float(max(side_errors)) if side_errors else 0.0
        }

    def _filter_lines_by_angle(self, lines: np.ndarray, angle_tolerance: float = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Filter lines into vertical and horizontal based on their angle."""
        vertical_lines = []
        horizontal_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Very strict tolerance: 2.5 degrees
            if 90 - angle_tolerance < angle < 90 + angle_tolerance:  # Nearly vertical
                vertical_lines.append(line)
            elif angle < angle_tolerance or angle > 180 - angle_tolerance:  # Nearly horizontal
                horizontal_lines.append(line)
                
        return np.array(vertical_lines), np.array(horizontal_lines)

    def _compute_all_distances(self, vertical_lines: np.ndarray, horizontal_lines: np.ndarray) -> tuple:
        """Compute all pairwise distances between lines with detailed analysis."""
        vertical_distances = self.compute_all_distances_dir(vertical_lines, 'vertical')
        horizontal_distances = self.compute_all_distances_dir(horizontal_lines, 'horizontal')
        
        all_distances = vertical_distances + horizontal_distances
        
        # Log distance computation details
        logger.info(f"Vertical line distances: {len(vertical_distances)}")
        logger.info(f"Horizontal line distances: {len(horizontal_distances)}")
        logger.info(f"Total distances: {len(all_distances)}")
        
        distance_details = {
            'vertical_distances': len(vertical_distances),
            'horizontal_distances': len(horizontal_distances),
            'total_distances': len(all_distances)
        }
        
        if all_distances:
            min_dist = min(all_distances)
            max_dist = max(all_distances)
            logger.info(f"Distance range: {min_dist:.1f} - {max_dist:.1f}")
            
            distance_details.update({
                'distance_range': {'min': float(min_dist), 'max': float(max_dist)}
            })
            
            # Generate histogram if matplotlib is available and debug mode is on
            # (disabled for performance)
        else:
            logger.info("No distances computed!")
            
        return all_distances, distance_details

    def _generate_distance_histogram(self, all_distances: List[float], frame_name: str) -> None:
        """Generate and save distance histogram plots."""
        try:
            plt.figure(figsize=(15, 5))
            
            # Regular histogram
            plt.subplot(1, 2, 1)
            plt.hist(all_distances, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Distance (pixels)')
            plt.ylabel('Frequency')
            plt.title('Distribution of All Line Distances')
            plt.grid(True, alpha=0.3)
            
            # Log scale histogram
            plt.subplot(1, 2, 2)
            plt.hist(all_distances, bins=50, alpha=0.7, edgecolor='black', log=True)
            plt.xlabel('Distance (pixels)')
            plt.ylabel('Frequency (log scale)')
            plt.title('Distribution of All Line Distances (Log Scale)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the plot
            output_path = os.path.join(self.debug_output_dir, frame_name, "distance_histogram.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Distance histogram saved to: {output_path}")
            
        except Exception as e:
            logger.warning(f"Failed to generate distance histogram: {e}")
    
    def compute_all_distances_dir(self, lines, direction):
        """Compute all pairwise distances between lines."""
        if len(lines) < 2:
            return []
            
        # Extract coordinates
        if direction == 'vertical':
            coords = [(line[0] + line[2]) / 2 for line in lines]
        else:
            coords = [(line[1] + line[3]) / 2 for line in lines]
        
        sorted_coords = np.sort(coords)
        
        # Get all pairwise distances
        all_distances = []
        for i in range(len(sorted_coords)):
            for j in range(i + 1, len(sorted_coords)):
                distance = abs(sorted_coords[j] - sorted_coords[i])
                if distance > 0:
                    all_distances.append(distance)
        
        return all_distances

    def _find_frequent_distances(self, distances: List[float], tolerance_pixels: float = 2.0, min_instances: int = 20) -> tuple:
        """Find frequently occurring distances using pixel-based tolerance and mode."""
        if len(distances) == 0:
            return [], {}, []
            
        sorted_distances = np.sort(distances)
        
        # Group distances within pixel tolerance
        distance_groups = []
        current_group = [sorted_distances[0]]
        
        for i in range(1, len(sorted_distances)):
            dist = sorted_distances[i]
            values, counts = np.unique(current_group, return_counts=True)
            group_center = values[np.argmax(counts)]  # Get mode
            
            if abs(dist - group_center) <= tolerance_pixels:
                current_group.append(dist)
            else:
                distance_groups.append(current_group)
                current_group = [dist]
        
        distance_groups.append(current_group)
        
        # Filter groups with min_instances+ instances and use mode
        frequent_distances = []
        frequent_groups = []
        for group in distance_groups:
            if len(group) >= min_instances:
                # Use mode instead of mean as the representative distance
                values, counts = np.unique(group, return_counts=True)
                mode_idx = np.argmax(counts)
                frequent_distances.append(values[mode_idx])
                frequent_groups.append(group)
        
        # Sort by distance value (smallest first)
        if frequent_distances:
            sorted_indices = np.argsort(frequent_distances)
            frequent_distances = [frequent_distances[i] for i in sorted_indices]
            frequent_groups = [frequent_groups[i] for i in sorted_indices]
        
        # Create detailed analysis
        logger.info(f"\nDistance Analysis:")
        logger.info(f"Total distance groups: {len(distance_groups)}")
        logger.info(f"Groups with {min_instances}+ instances: {len(frequent_groups)}")
        
        # Show all groups sorted by frequency
        all_groups_with_freq = [(len(group), np.mean(group)) for group in distance_groups]
        all_groups_with_freq.sort(reverse=True)  # Sort by frequency (descending)
        
        logger.info(f"\nTop 20 distance groups (by frequency):")
        for i, (freq, mean_dist) in enumerate(all_groups_with_freq[:20]):
            marker = " *** SELECTED ***" if freq >= min_instances else ""
            logger.info(f"{i+1:2d}. Distance: {mean_dist:6.1f} px, Frequency: {freq:3d}{marker}")
        
        if frequent_distances:
            logger.info(f"\nFrequent distances ({min_instances}+ instances, sorted smallest first):")
            for i, (distance, group) in enumerate(zip(frequent_distances, frequent_groups)):
                logger.info(f"{i+1}. Distance: {distance:.1f} px, Instances: {len(group)}")
        else:
            logger.info(f"\nNo distances found with {min_instances}+ instances!")
            logger.info("This suggests the chess board pattern is not clearly detectable.")
        
        # Create detailed info for JSON log
        distance_analysis = {
            'total_groups': len(distance_groups),
            'frequent_groups_count': len(frequent_groups),
            'min_instances': min_instances,
            'tolerance_pixels': tolerance_pixels,
            'top_groups': [{'distance': float(mean_dist), 'frequency': freq, 'selected': freq >= min_instances} 
                          for freq, mean_dist in all_groups_with_freq[:20]],
            'frequent_distances': [{'distance': float(distance), 'instances': len(group)} 
                                  for distance, group in zip(frequent_distances, frequent_groups)]
        }
        
        return frequent_distances, distance_analysis, distance_groups

    def _validate_distance_multiples(self, distances: List[float], candidate_distance: float, tolerance_pixels: float = 4) -> bool:
        """Validate that multiples of the candidate distance appear."""
        distances_array = np.array(distances)
        expected_multiples = [2, 3, 4]
        found_multiples = 0
        
        for multiple in expected_multiples:
            expected_distance = candidate_distance * multiple
            close_distances = distances_array[np.abs(distances_array - expected_distance) <= tolerance_pixels]
            if len(close_distances) > 0:
                found_multiples += 1
        
        return found_multiples >= 2

    def _select_lines_matching_pattern(self, lines: np.ndarray, direction: str, square_side_length: float, tolerance_pixels: float = 1.0) -> tuple:
        """Select lines that match the chess square spacing pattern."""
        if len(lines) == 0 or square_side_length is None:
            return np.array([]), []
            
        # Extract coordinates and sort
        if direction == 'vertical':
            coords = [(line[0] + line[2]) / 2 for line in lines]
        else:
            coords = [(line[1] + line[3]) / 2 for line in lines]
        
        coords_array = np.array(coords)
        sorted_indices = np.argsort(coords_array)
        sorted_lines = lines[sorted_indices]
        sorted_coords = coords_array[sorted_indices]
        
        selected_lines = []
        line_details = []
        
        logger.info(f"\nSelecting {direction} lines matching pattern (tolerance: ±{tolerance_pixels:.1f} px):")
        
        # For each line, compute distances to all other lines
        for line, coord in zip(sorted_lines, sorted_coords):
            # Get distances to all other lines
            distances = np.abs(sorted_coords - coord)
            
            # Count how many distances match multiples of square_side_length
            matching_distances = 0
            
            for distance in distances:
                if distance == 0:  # Skip self-distance
                    continue
                    
                # Find nearest multiple of square_side_length
                multiple = round(distance / square_side_length)
                expected_distance = multiple * square_side_length
                
                if multiple > 0 and abs(distance - expected_distance) <= tolerance_pixels:
                    matching_distances += 1
            
            # Create line detail record
            line_detail = {
                'coordinate': float(coord),
                'matching_distances': matching_distances,
                'selected': matching_distances >= 4
            }
            line_details.append(line_detail)
            
            # If line has at least 4 matching distances, select it
            if matching_distances >= 4:
                selected_lines.append(line)
                logger.info(f"  Line {len(selected_lines)}: {coord:.1f} px ({matching_distances} matching distances)")
            else:
                logger.info(f"  Line rejected: {coord:.1f} px (only {matching_distances} matching distances)")
        
        logger.info(f"  Selected {len(selected_lines)} out of {len(lines)} {direction} lines")
        return np.array(selected_lines), line_details

    def _filter_lines_by_boundaries(self, horizontal_lines: np.ndarray, vertical_lines: np.ndarray, buffer_pixels: float = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Filter both horizontal and vertical lines by boundaries of the other direction."""
        if len(vertical_lines) == 0 or len(horizontal_lines) == 0:
            return horizontal_lines, vertical_lines
        
        # Get vertical line x-coordinates for horizontal filtering
        v_coords = [(line[0] + line[2]) / 2 for line in vertical_lines]
        min_x = min(v_coords) - buffer_pixels
        max_x = max(v_coords) + buffer_pixels
        
        # Get horizontal line y-coordinates for vertical filtering
        h_coords = [(line[1] + line[3]) / 2 for line in horizontal_lines]
        min_y = min(h_coords) - buffer_pixels
        max_y = max(h_coords) + buffer_pixels
        
        # Filter horizontal lines by vertical boundaries
        filtered_horizontal = []
        for line in horizontal_lines:
            line_x_min = min(line[0], line[2])
            line_x_max = max(line[0], line[2])
            
            # Keep lines that have some overlap with the vertical boundary region
            if not (line_x_max < min_x or line_x_min > max_x):
                filtered_horizontal.append(line)
        
        # Filter vertical lines by horizontal boundaries
        filtered_vertical = []
        for line in vertical_lines:
            line_y_min = min(line[1], line[3])
            line_y_max = max(line[1], line[3])
            
            # Keep lines that have some overlap with the horizontal boundary region
            if not (line_y_max < min_y or line_y_min > max_y):
                filtered_vertical.append(line)
        
        return np.array(filtered_horizontal), np.array(filtered_vertical)

    def _draw_lines(self, image: np.ndarray, lines: np.ndarray, color=(0, 255, 0), thickness=2) -> np.ndarray:
        """Draw lines on an image for visualization."""
        result = image.copy()
        if lines is not None and len(lines) > 0:
            for line in lines:
                x1, y1, x2, y2 = line.astype(int)
                cv2.line(result, (x1, y1), (x2, y2), color, thickness)
        return result

    def _calculate_board_corners(self, vertical_lines: np.ndarray, horizontal_lines: np.ndarray) -> Optional[np.ndarray]:
        """Calculate board corners from line intersections."""
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
        top_left = self._line_intersection(left_line, top_line)
        top_right = self._line_intersection(right_line, top_line)
        bottom_left = self._line_intersection(left_line, bottom_line)
        bottom_right = self._line_intersection(right_line, bottom_line)
        
        if any(corner is None for corner in [top_left, top_right, bottom_left, bottom_right]):
            return None
        
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

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

    def _extract_board(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Extract the chess board using perspective transformation."""
        # Define target size for the extracted board
        board_size = 400
        target_corners = np.array([
            [0, 0],
            [board_size, 0],
            [board_size, board_size],
            [0, board_size]
        ], dtype=np.float32)
        
        # Calculate perspective transform
        matrix = cv2.getPerspectiveTransform(corners, target_corners)
        
        # Apply transform
        board = cv2.warpPerspective(image, matrix, (board_size, board_size))
        
        return board