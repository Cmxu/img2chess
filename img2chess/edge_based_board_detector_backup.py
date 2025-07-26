"""
Edge-based Chess Board Detector

This module implements a new approach for detecting chess boards in images
using edge detection and line filtering techniques.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import os
import logging

logger = logging.getLogger(__name__)

class EdgeBasedChessBoardDetector:
    """
    Detects chess boards using edge detection and line filtering.
    
    This approach finds all edges in the image, filters for nearly vertical/horizontal
    lines, then identifies equally spaced sets of 8 lines in each direction to
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
    
    def detect_board(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and extract the chess board from an image using edge detection.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Extracted chess board image or None if no board detected
        """
        # Create debug directory if needed
        if self.debug_mode and not os.path.exists(self.debug_output_dir):
            os.makedirs(self.debug_output_dir)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if self.debug_mode:
            cv2.imwrite(os.path.join(self.debug_output_dir, "01_original_gray.jpg"), gray)
        
        # Apply edge detection with multiple approaches
        edges = self._detect_edges(gray)
        if self.debug_mode:
            cv2.imwrite(os.path.join(self.debug_output_dir, "02_edges.jpg"), edges)
        
        # Find lines using HoughLinesP with multiple parameter sets
        lines = self._find_lines(edges)
        if lines is None:
            logger.warning("No lines detected in image")
            return None
            
        if self.debug_mode:
            line_img = self._draw_lines(image.copy(), lines)
            cv2.imwrite(os.path.join(self.debug_output_dir, "03_all_lines.jpg"), line_img)
        
        # Filter lines into vertical and horizontal
        vertical_lines, horizontal_lines = self._filter_lines(lines)
        
        if self.debug_mode:
            v_line_img = self._draw_lines(image.copy(), vertical_lines)
            cv2.imwrite(os.path.join(self.debug_output_dir, "04_vertical_lines.jpg"), v_line_img)
            
            h_line_img = self._draw_lines(image.copy(), horizontal_lines)
            cv2.imwrite(os.path.join(self.debug_output_dir, "05_horizontal_lines.jpg"), h_line_img)
        
        # New approach: Find chess square side length from both directions combined
        # Reset any previous state
        if hasattr(self, '_all_distances'):
            delattr(self, '_all_distances')
        if hasattr(self, '_directions_processed'):
            delattr(self, '_directions_processed')
            
        # Process vertical lines first (just collects distances)
        self._find_chess_square_side_length(vertical_lines, [(line[0] + line[2]) / 2 for line in vertical_lines], 'vertical')
        
        # Process horizontal lines second (analyzes combined distances)
        square_side_length = self._find_chess_square_side_length(horizontal_lines, [(line[1] + line[3]) / 2 for line in horizontal_lines], 'horizontal')
        
        if square_side_length is None:
            logger.warning("Could not determine chess square side length from line distances")
            return None
            
        # Select lines that match the discovered square side length pattern
        final_v_lines = self._create_grid_from_square_side_length(vertical_lines, [(line[0] + line[2]) / 2 for line in vertical_lines], 'vertical', 9, square_side_length)
        final_h_lines = self._create_grid_from_square_side_length(horizontal_lines, [(line[1] + line[3]) / 2 for line in horizontal_lines], 'horizontal', 9, square_side_length)
        
        if final_v_lines is None or final_h_lines is None or len(final_v_lines) < 7 or len(final_h_lines) < 7:
            logger.warning("Could not find enough lines matching the chess square side length pattern")
            return None
            
        if self.debug_mode:
            v_final_img = self._draw_lines(image.copy(), final_v_lines)
            cv2.imwrite(os.path.join(self.debug_output_dir, "06_final_vertical_lines.jpg"), v_final_img)
            
            h_final_img = self._draw_lines(image.copy(), final_h_lines)
            cv2.imwrite(os.path.join(self.debug_output_dir, "07_final_horizontal_lines.jpg"), h_final_img)
            
            all_final_img = self._draw_lines(image.copy(), np.vstack([final_v_lines, final_h_lines]))
            cv2.imwrite(os.path.join(self.debug_output_dir, "08_final_lines.jpg"), all_final_img)
        
        # Calculate board corners from line intersections
        corners = self._calculate_board_corners(final_v_lines, final_h_lines)
        if corners is None:
            logger.warning("Could not calculate board corners from lines")
            return None
            
        if self.debug_mode:
            corner_img = image.copy()
            for point in corners:
                cv2.circle(corner_img, tuple(point.astype(int)), 5, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(self.debug_output_dir, "09_board_corners.jpg"), corner_img)
        
        # Extract and return the board
        board = self._extract_board(image, corners)
        return board

    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection to the grayscale image with adaptive parameters."""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Compute median intensity to set adaptive thresholds
        median = np.median(blurred)
        lower = int(max(0, 0.7 * median))
        upper = int(min(255, 1.3 * median))
        
        # Apply Canny edge detection with adaptive thresholds
        edges = cv2.Canny(blurred, lower, upper, apertureSize=3)
        return edges

    def _find_lines(self, edges: np.ndarray) -> Optional[np.ndarray]:
        """Find lines in the edge image using HoughLinesP with multiple parameter sets."""
        all_lines = []
        
        # Try different parameter combinations
        param_sets = [
            {'threshold': 50, 'minLineLength': 30, 'maxLineGap': 10},
            {'threshold': 80, 'minLineLength': 50, 'maxLineGap': 15},
            {'threshold': 100, 'minLineLength': 70, 'maxLineGap': 20},
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
            
        # Remove duplicate lines
        unique_lines = self._remove_duplicate_lines(np.array(all_lines))
        return unique_lines

    def _remove_duplicate_lines(self, lines: np.ndarray, distance_threshold: float = 10) -> np.ndarray:
        """Remove duplicate or very similar lines."""
        if len(lines) == 0:
            return lines
            
        unique_lines = []
        for line in lines:
            is_duplicate = False
            for existing_line in unique_lines:
                # Calculate distance between line midpoints
                mid1 = np.array([(line[0] + line[2]) / 2, (line[1] + line[3]) / 2])
                mid2 = np.array([(existing_line[0] + existing_line[2]) / 2, (existing_line[1] + existing_line[3]) / 2])
                distance = np.linalg.norm(mid1 - mid2)
                
                # Calculate angle difference
                angle1 = np.arctan2(line[3] - line[1], line[2] - line[0])
                angle2 = np.arctan2(existing_line[3] - existing_line[1], existing_line[2] - existing_line[0])
                angle_diff = np.abs(angle1 - angle2)
                
                # If lines are close and have similar angles, consider them duplicates
                if distance < distance_threshold and angle_diff < np.pi/18:  # 10 degrees
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_lines.append(line)
                
        return np.array(unique_lines)

    def _filter_lines(self, lines: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter lines into vertical and horizontal based on their angle.
        
        Args:
            lines: Array of lines in (x1, y1, x2, y2) format
            
        Returns:
            Tuple of (vertical_lines, horizontal_lines)
        """
        vertical_lines = []
        horizontal_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Calculate angle in degrees (relative to horizontal axis)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Classify as vertical or horizontal (with very strict tolerance)
            # Vertical lines have angles close to 90 degrees
            if 87.5 < angle < 92.5:  # Nearly vertical (2.5 degree tolerance)
                vertical_lines.append(line)
            # Horizontal lines have angles close to 0 or 180 degrees
            elif angle < 2.5 or angle > 177.5:  # Nearly horizontal (2.5 degree tolerance)
                horizontal_lines.append(line)
                
        return np.array(vertical_lines), np.array(horizontal_lines)

    def _filter_lines_by_min_distance(self, lines: np.ndarray, direction: str, min_distance_ratio: float = 0.05) -> np.ndarray:
        """
        Filter lines to ensure minimum distance between them.
        This helps reject lines that are too close together and improves evenness.
        
        Args:
            lines: Array of lines in (x1, y1, x2, y2) format
            direction: 'vertical' or 'horizontal'
            min_distance_ratio: Minimum distance as ratio of image dimension
            
        Returns:
            Filtered array of lines with minimum distance enforcement
        """
        if len(lines) <= 1:
            return lines
            
        # Extract relevant coordinate for sorting (x for vertical, y for horizontal)
        if direction == 'vertical':
            coords = [(line[0] + line[2]) / 2 for line in lines]
            # Estimate image width for minimum distance calculation
            img_width = max(max(line[0], line[2]) for line in lines) - min(min(line[0], line[2]) for line in lines)
            img_height = max(max(line[1], line[3]) for line in lines) - min(min(line[1], line[3]) for line in lines)
            img_dimension = min(img_width, img_height)  # Use smaller dimension
        else:  # horizontal
            coords = [(line[1] + line[3]) / 2 for line in lines]
            # Estimate image dimensions for minimum distance calculation
            img_width = max(max(line[0], line[2]) for line in lines) - min(min(line[0], line[2]) for line in lines)
            img_height = max(max(line[1], line[3]) for line in lines) - min(min(line[1], line[3]) for line in lines)
            img_dimension = min(img_width, img_height)  # Use smaller dimension
            
        if img_dimension == 0:
            return lines
            
        min_distance = img_dimension * min_distance_ratio
        
        # Sort lines by coordinate
        sorted_indices = np.argsort(coords)
        sorted_lines = lines[sorted_indices]
        sorted_coords = np.array(coords)[sorted_indices]
        
        # Filter to maintain minimum distance
        filtered_lines = [sorted_lines[0]]  # Always keep the first line
        filtered_coords = [sorted_coords[0]]
        
        for i in range(1, len(sorted_lines)):
            # Check distance to the last kept line
            distance = sorted_coords[i] - filtered_coords[-1]
            if distance >= min_distance:
                filtered_lines.append(sorted_lines[i])
                filtered_coords.append(sorted_coords[i])
                
        return np.array(filtered_lines)

    def _find_equally_spaced_lines(self, lines: np.ndarray, direction: str) -> Optional[np.ndarray]:
        """
        Find 9 equally spaced lines from the given set of lines (for 8x8 chessboard).
        Robust to partial detection - can work with 7+ lines and extrapolate missing ones.
        Applies minimum distance filtering to ensure lines are well-spaced.
        
        Args:
            lines: Array of lines in (x1, y1, x2, y2) format
            direction: 'vertical' or 'horizontal'
            
        Returns:
            Array of 9 equally spaced lines or None if not found
        """
        if len(lines) < 7:
            # Need at least 7 lines to reliably detect the pattern
            return None
            
        target_count = 9  # 9 lines for 8x8 chessboard
        
        # Apply minimum distance filtering first
        filtered_lines = self._filter_lines_by_min_distance(lines, direction)
        if len(filtered_lines) < 5:  # Relaxed from 7 to allow more detection
            return None
        lines = filtered_lines
            
        # Extract relevant coordinate for sorting (x for vertical, y for horizontal)
        if direction == 'vertical':
            # Use average x coordinate of line endpoints
            coords = [(line[0] + line[2]) / 2 for line in lines]
        else:  # horizontal
            # Use average y coordinate of line endpoints
            coords = [(line[1] + line[3]) / 2 for line in lines]
            
        # Sort lines by their position
        sorted_indices = np.argsort(coords)
        sorted_lines = lines[sorted_indices]
        sorted_coords = np.array(coords)[sorted_indices]
        
        # If we have exactly the right number, return them
        if len(sorted_lines) == target_count:
            return sorted_lines
            
        # If we have more than needed, find the best subset
        if len(sorted_lines) > target_count:
            # Try to find target_count lines with approximately equal spacing
            best_lines = None
            best_score = float('inf')
            
            # Try different combinations of target_count lines, not just consecutive ones
            # This allows for better selection of evenly spaced lines
            from itertools import combinations
            
            # Limit combinations to avoid excessive computation
            max_combinations = min(100, len(list(combinations(range(len(sorted_lines)), target_count))))
            combinations_tried = 0
            
            for indices in combinations(range(len(sorted_lines)), target_count):
                if combinations_tried >= max_combinations:
                    break
                combinations_tried += 1
                
                candidate_lines = sorted_lines[list(indices)]
                candidate_coords = sorted_coords[list(indices)]
                
                # Calculate spacings between consecutive lines
                spacings = np.diff(candidate_coords)
                
                # Calculate how consistent the spacing is
                mean_spacing = np.mean(spacings)
                if mean_spacing == 0:
                    continue
                    
                # Score based on coefficient of variation (lower is better)
                spacing_std = np.std(spacings)
                cv = spacing_std / mean_spacing  # Coefficient of variation
                
                # Also consider how well the lines span the available space
                total_span = candidate_coords[-1] - candidate_coords[0]
                available_span = sorted_coords[-1] - sorted_coords[0]
                span_ratio = total_span / available_span if available_span > 0 else 0
                
                # Combined score: prioritize consistent spacing and good coverage
                score = cv + (1 - span_ratio) * 0.3  # Weight span coverage less
                
                if score < best_score:
                    best_score = score
                    best_lines = candidate_lines
                    
            # Apply a threshold for acceptable spacing consistency
            if best_score > 0.2:  # Stricter threshold for better quality
                # If no good equally spaced set found, try extrapolation
                return self._extrapolate_missing_lines(sorted_lines, sorted_coords, direction, target_count)
                
            return best_lines if best_lines is not None else None
            
        # If we have fewer lines, try to extrapolate to get the full set
        return self._extrapolate_missing_lines(sorted_lines, sorted_coords, direction, target_count)
    
    def _extrapolate_missing_lines(self, lines: np.ndarray, coords: np.ndarray, direction: str, target_count: int) -> Optional[np.ndarray]:
        """
        Extrapolate missing lines to complete the chessboard grid.
        
        Args:
            lines: Available lines sorted by position
            coords: Coordinates of the lines (x for vertical, y for horizontal)
            direction: 'vertical' or 'horizontal'
            target_count: Number of lines needed (should be 9)
            
        Returns:
            Array of target_count lines or None if extrapolation fails
        """
        if len(lines) < 2:
            return None
            
        # Calculate average spacing from available lines
        if len(coords) >= 3:
            # Use multiple spacings to get better average
            spacings = np.diff(coords)
            # Remove outliers (spacings that are much different)
            median_spacing = np.median(spacings)
            good_spacings = spacings[np.abs(spacings - median_spacing) < median_spacing * 0.5]
            if len(good_spacings) > 0:
                avg_spacing = np.mean(good_spacings)
            else:
                avg_spacing = median_spacing
        else:
            # Only 2 points, use their spacing
            avg_spacing = coords[-1] - coords[0]
            
        if avg_spacing <= 0:
            return None
            
        # New approach: Find chess square side length by analyzing distance patterns
        
        # Step 1: Find the most frequent distance (chess square side length) from all line pairs
        square_side_length = self._find_chess_square_side_length(lines, coords, direction)
        
        if square_side_length is None:
            return None
            
        # This method is now called from the main detect_board method
        # Individual direction processing is handled there
        return None
    
    def _find_chess_square_side_length(self, lines: np.ndarray, coords: np.ndarray, direction: str) -> Optional[float]:
        """
        Find the chess square side length by analyzing distance patterns between all lines.
        This method will be called for both vertical and horizontal lines, and results combined.
        """
        if len(coords) < 2:
            return None
            
        # Get all pairwise distances between lines
        all_distances = []
        sorted_coords = np.sort(coords)
        
        for i in range(len(sorted_coords)):
            for j in range(i + 1, len(sorted_coords)):
                distance = abs(sorted_coords[j] - sorted_coords[i])
                if distance > 0:  # Ignore zero distances
                    all_distances.append(distance)
        
        if not all_distances:
            return None
        
        # Store distances for combining with other direction later
        if not hasattr(self, '_all_distances'):
            self._all_distances = []
        self._all_distances.extend(all_distances)
        
        # If this is the second direction, analyze combined distances
        if hasattr(self, '_directions_processed'):
            return self._analyze_combined_distances()
        else:
            # Mark that we've processed one direction
            self._directions_processed = True
            return None
    
    def _analyze_combined_distances(self) -> Optional[float]:
        """
        Analyze combined distances from both vertical and horizontal lines to find chess square side length.
        """
        if not hasattr(self, '_all_distances') or not self._all_distances:
            return None
            
        distances = np.array(self._all_distances)
        
        # Find the most frequent distance (with some tolerance for floating point errors)
        distance_candidates = self._find_frequent_distances(distances)
        
        # Test each candidate to see if it has the expected multiples
        for candidate_distance in distance_candidates:
            if self._validate_distance_multiples(distances, candidate_distance):
                # Clean up temporary storage
                delattr(self, '_all_distances')
                delattr(self, '_directions_processed')
                return candidate_distance
        
        # Clean up temporary storage
        delattr(self, '_all_distances')
        delattr(self, '_directions_processed')
        return None
    
    def _find_frequent_distances(self, distances: np.ndarray, tolerance_ratio: float = 0.02) -> List[float]:
        """
        Find the most frequently occurring distances with some tolerance.
        """
        if len(distances) == 0:
            return []
            
        # Sort distances
        sorted_distances = np.sort(distances)
        
        # Group distances that are within tolerance of each other
        distance_groups = []
        current_group = [sorted_distances[0]]
        
        for i in range(1, len(sorted_distances)):
            dist = sorted_distances[i]
            # Check if this distance is close to the current group
            group_center = np.mean(current_group)
            tolerance = group_center * tolerance_ratio
            
            if abs(dist - group_center) <= tolerance:
                current_group.append(dist)
            else:
                # Start a new group
                distance_groups.append(current_group)
                current_group = [dist]
        
        # Add the last group
        distance_groups.append(current_group)
        
        # Sort groups by frequency (most frequent first)
        distance_groups.sort(key=len, reverse=True)
        
        # Filter groups to keep only those with 25+ instances, then sort by distance value (smallest first)
        frequent_distances = []
        for group in distance_groups:
            if len(group) >= 25:  # Only consider distances that appear at least 25 times
                frequent_distances.append(np.mean(group))
        
        # Sort by distance value (smallest first) rather than frequency
        frequent_distances.sort()
        
        return frequent_distances
    
    def _validate_distance_multiples(self, distances: np.ndarray, candidate_distance: float, tolerance_ratio: float = 0.02) -> bool:
        """
        Validate that multiples of the candidate distance (2x, 3x, 4x) also appear in the distance list.
        This confirms it's likely the chess square side length.
        """
        tolerance = candidate_distance * tolerance_ratio
        
        # Check for multiples: 2x, 3x, 4x (and possibly more)
        expected_multiples = [2, 3, 4]  # We expect to see these multiples for chess boards
        found_multiples = 0
        
        for multiple in expected_multiples:
            expected_distance = candidate_distance * multiple
            
            # Check if this multiple exists in our distances
            close_distances = distances[np.abs(distances - expected_distance) <= tolerance]
            if len(close_distances) > 0:
                found_multiples += 1
        
        # We should find at least 2 out of 3 expected multiples for a valid chess square side length
        return found_multiples >= 2
    
    def _score_grid_fit(self, sorted_coords: np.ndarray, spacing: float, target_count: int) -> float:
        """
        Score how well the detected lines fit a regular grid with the given spacing.
        Lower scores are better.
        """
        if spacing <= 0:
            return float('inf')
            
        # Try different starting positions for the grid to find the best fit
        min_coord = sorted_coords[0]
        max_coord = sorted_coords[-1]
        
        best_score = float('inf')
        
        # Try grid starting positions that would place the grid around our detected lines
        for start_offset in np.arange(-2 * spacing, 2 * spacing, spacing / 4):
            grid_start = min_coord - 2 * spacing + start_offset  # Allow grid to extend beyond detected lines
            
            # Generate grid positions
            grid_positions = [grid_start + i * spacing for i in range(target_count)]
            
            # Calculate how well our detected lines match this grid
            total_distance = 0
            matched_lines = 0
            
            for coord in sorted_coords:
                # Find the closest grid position
                distances = [abs(coord - grid_pos) for grid_pos in grid_positions]
                min_distance = min(distances)
                
                if min_distance < spacing * 0.5:  # Must be reasonably close to grid position (relaxed)
                    total_distance += min_distance
                    matched_lines += 1
            
            if matched_lines >= max(2, len(sorted_coords) * 0.4):  # At least 40% of lines should match (relaxed)
                # Score based on average distance and coverage
                avg_distance = total_distance / matched_lines if matched_lines > 0 else float('inf')
                coverage_penalty = (len(sorted_coords) - matched_lines) * spacing  # Penalty for unmatched lines
                score = avg_distance + coverage_penalty
                
                if score < best_score:
                    best_score = score
        
        return best_score
    
    def _create_perfect_grid_with_original_lines(self, lines: np.ndarray, coords: np.ndarray, direction: str, target_count: int, optimal_spacing: float) -> np.ndarray:
        """
        Create a perfectly spaced grid for chess board, using original lines where possible.
        All lines will have EXACTLY equal spacing as required for a square chess board.
        """
        # Sort original lines by coordinate
        sorted_indices = np.argsort(coords)
        sorted_lines = lines[sorted_indices]
        sorted_coords = coords[sorted_indices]
        
        # Find the best position for the perfect grid that maximizes use of original lines
        best_grid_start = self._find_optimal_grid_start(sorted_coords, optimal_spacing, target_count)
        
        # Generate the perfect grid positions with exactly equal spacing
        perfect_grid_positions = [best_grid_start + i * optimal_spacing for i in range(target_count)]
        
        # Create the final grid by matching original lines to perfect positions
        result_lines = []
        used_original_indices = set()
        
        for grid_pos in perfect_grid_positions:
            # Find the closest unused original line to this perfect grid position
            best_original_idx = None
            best_distance = float('inf')
            
            for i, coord in enumerate(sorted_coords):
                if i not in used_original_indices:
                    distance = abs(coord - grid_pos)
                    if distance < best_distance:
                        best_distance = distance
                        best_original_idx = i
            
            # Use original line if it's close enough to the perfect position
            if best_original_idx is not None and best_distance < optimal_spacing * 0.4:  # Reasonable tolerance for perfect grid
                # Use the original line but place it at the EXACT grid position for perfect spacing
                original_line = sorted_lines[best_original_idx].copy()
                
                if direction == 'vertical':
                    # Adjust x coordinates to exact grid position
                    original_line[0] = grid_pos
                    original_line[2] = grid_pos
                else:  # horizontal
                    # Adjust y coordinates to exact grid position
                    original_line[1] = grid_pos
                    original_line[3] = grid_pos
                    
                result_lines.append(original_line)
                used_original_indices.add(best_original_idx)
            else:
                # Create a new line at the exact grid position
                # Use the closest original line as a template for length and orientation
                if len(sorted_lines) > 0:
                    template_idx = np.argmin(np.abs(sorted_coords - grid_pos))
                    template_line = sorted_lines[template_idx]
                    
                    if direction == 'vertical':
                        new_line = np.array([
                            grid_pos, template_line[1],
                            grid_pos, template_line[3]
                        ])
                    else:  # horizontal
                        new_line = np.array([
                            template_line[0], grid_pos,
                            template_line[2], grid_pos
                        ])
                    result_lines.append(new_line)
        
        return np.array(result_lines)
    
    def _find_optimal_grid_start(self, sorted_coords: np.ndarray, spacing: float, target_count: int) -> float:
        """
        Find the optimal starting position for the perfect grid that maximizes use of original lines.
        """
        if len(sorted_coords) == 0:
            return 0.0
            
        # Try different grid starting positions and score them
        min_coord = sorted_coords[0]
        max_coord = sorted_coords[-1]
        
        best_start = min_coord
        best_score = float('inf')
        
        # Try grid starts that would place the grid around our detected lines
        search_range = max(2 * spacing, (max_coord - min_coord) * 0.5)
        for start_candidate in np.arange(min_coord - search_range, min_coord + search_range, spacing / 8):
            
            # Generate grid positions for this starting point
            grid_positions = [start_candidate + i * spacing for i in range(target_count)]
            
            # Score this grid by how many original lines it can use
            total_distance = 0
            matched_count = 0
            
            # For each original line, find its closest grid position
            for coord in sorted_coords:
                distances = [abs(coord - grid_pos) for grid_pos in grid_positions]
                min_distance = min(distances)
                
                if min_distance < spacing * 0.4:  # Must be close to count as matched
                    total_distance += min_distance
                    matched_count += 1
            
            # Score favors grids that can use more original lines with smaller adjustments
            if matched_count > 0:
                avg_distance = total_distance / matched_count
                # Prioritize using more original lines, then minimize adjustment distance
                score = -matched_count * 100 + avg_distance  # Negative for maximization
            else:
                score = float('inf')  # No matches is very bad
            
            if score < best_score:
                best_score = score
                best_start = start_candidate
        
        return best_start
    def _create_grid_from_square_side_length(self, lines: np.ndarray, coords: np.ndarray, direction: str, target_count: int,  square_side_length: float) -> Optional[np.ndarray]:
        """
        Create a perfect 9-line grid using the discovered chess square side length.
        Select original lines that best fit this spacing pattern.
        """
        if len(lines) == 0:
            return None
            
        # Sort lines by coordinate
        coords_array = np.array(coords)
        sorted_indices = np.argsort(coords_array)
        sorted_lines = lines[sorted_indices]
        sorted_coords = coords_array[sorted_indices]
        
        # Find the best set of lines that match the square side length pattern
        selected_lines = self._select_lines_matching_square_pattern(sorted_lines, sorted_coords, square_side_length)
        
        if len(selected_lines) < 7:  # Need at least 7 lines for a valid chess board
            return None
            
        # Extract coordinates of selected lines
        if direction == 'vertical':
            selected_coords = [(line[0] + line[2]) / 2 for line in selected_lines]
        else:
            selected_coords = [(line[1] + line[3]) / 2 for line in selected_lines]
            
        # Return the selected lines that match the chess square pattern (no grid creation)
        return np.array(selected_lines)
    
    def _select_lines_matching_square_pattern(self, sorted_lines: np.ndarray, sorted_coords: np.ndarray, square_side_length: float) -> List:
        """
        Select lines from the detected set that best match the chess square spacing pattern.
        """
        if len(sorted_lines) == 0:
            return []
            
        # Find lines that are spaced at multiples of the square side length
        selected_lines = []
        selected_coords = []
        tolerance = square_side_length * 0.02  # 2% tolerance (much stricter)
        
        # Start with the first line
        selected_lines.append(sorted_lines[0])
        selected_coords.append(sorted_coords[0])
        
        # For each subsequent line, check if it's at an appropriate multiple distance
        for i in range(1, len(sorted_lines)):
            line = sorted_lines[i]
            coord = sorted_coords[i]
            
            # Check distance to any of the already selected lines
            for selected_coord in selected_coords:
                distance = abs(coord - selected_coord) 
                # Check if this distance is close to a multiple of square_side_length
                multiple = round(distance / square_side_length)
                expected_distance = multiple * square_side_length
                
                if multiple > 0 and abs(distance - expected_distance) <= tolerance:
                    # This line fits the pattern
                    selected_lines.append(line)
                    selected_coords.append(coord)
                    break
        
        return selected_lines
    
    def _create_perfect_equal_grid(self, selected_lines: List, selected_coords: List[float], direction: str, target_count: int, square_side_length: float) -> np.ndarray:
        """
        Create a perfect grid with exactly equal spacing using the square side length.
        """
        if len(selected_lines) == 0:
            return None
            
        # Find the best starting position for the perfect grid
        min_coord = min(selected_coords)
        max_coord = max(selected_coords)
        
        # Try different starting positions to maximize use of original lines
        best_grid_start = min_coord
        best_original_usage = 0
        
        # Just return the selected lines that match the pattern - no perfect grid needed
        # Sort by coordinate to maintain order
        sorted_indices = np.argsort(selected_coords)
        return np.array([selected_lines[i] for i in sorted_indices])

    def _calculate_board_corners(self, vertical_lines: np.ndarray, horizontal_lines: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate board corners from intersections of vertical and horizontal lines.
        
        Args:
            vertical_lines: Array of vertical lines
            horizontal_lines: Array of horizontal lines
            
        Returns:
            Array of 4 corner points (top-left, top-right, bottom-right, bottom-left)
        """
        if len(vertical_lines) < 2 or len(horizontal_lines) < 2:
            return None
            
        # Get the extreme lines (outermost)
        # For vertical lines, use x coordinate
        v_coords = [(line[0] + line[2]) / 2 for line in vertical_lines]
        left_v_idx = np.argmin(v_coords)
        right_v_idx = np.argmax(v_coords)
        
        # For horizontal lines, use y coordinate
        h_coords = [(line[1] + line[3]) / 2 for line in horizontal_lines]
        top_h_idx = np.argmin(h_coords)
        bottom_h_idx = np.argmax(h_coords)
        
        # Get the four corner lines
        left_line = vertical_lines[left_v_idx]
        right_line = vertical_lines[right_v_idx]
        top_line = horizontal_lines[top_h_idx]
        bottom_line = horizontal_lines[bottom_h_idx]
        
        # Calculate intersections
        # Top-left corner (left line with top line)
        tl = self._line_intersection(left_line, top_line)
        # Top-right corner (right line with top line)
        tr = self._line_intersection(right_line, top_line)
        # Bottom-right corner (right line with bottom line)
        br = self._line_intersection(right_line, bottom_line)
        # Bottom-left corner (left line with bottom line)
        bl = self._line_intersection(left_line, bottom_line)
        
        if any(point is None for point in [tl, tr, br, bl]):
            return None
            
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _line_intersection(self, line1: np.ndarray, line2: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate intersection point of two lines.
        
        Args:
            line1, line2: Lines in (x1, y1, x2, y2) format
            
        Returns:
            Intersection point (x, y) or None if lines are parallel
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Calculate denominator
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:  # Lines are parallel
            return None
            
        # Calculate intersection point
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        
        return np.array([px, py])

    def _extract_board(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Extract and perspective-correct the chess board region.
        
        Args:
            image: Original image
            corners: Board corner coordinates (tl, tr, br, bl)
            
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

    def _draw_lines(self, image: np.ndarray, lines: np.ndarray) -> np.ndarray:
        """Draw lines on an image for debugging purposes."""
        if lines is None or len(lines) == 0:
            return image
            
        result = image.copy()
        for line in lines:
            x1, y1, x2, y2 = line.astype(int)
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return result