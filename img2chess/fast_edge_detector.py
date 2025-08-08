import cv2
import numpy as np
import yaml
from typing import Optional, Tuple, List, Dict, Any


class FastEdgeBasedDetector:
    """
    Faster edge-based chess board detector.

    Key speedups vs CleanEdgeBasedDetector:
    - Optional downscaled detection (resolves on smaller image, then maps to original)
    - Vectorized line selection using precomputed distance matrices
    - Simplified duplicate removal via greedy clustering by coordinate
    - Fewer boundary filtering iterations

    Uses the same YAML configuration file as CleanEdgeBasedDetector.
    Extra optional fast settings can be provided under config['fast_settings']:
      fast_settings:
        max_detection_dim: 720           # max dimension for downscaled detection
        max_lines_per_direction: 60      # limit number of lines per direction by length
        boundary_filter_iterations: 2    # fewer iterations than the clean version
        use_downscale: true              # enable downscaled detection
    """

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        fast_defaults = {
            'max_detection_dim': 720,
            'max_lines_per_direction': 60,
            'boundary_filter_iterations': 2,
            'use_downscale': True,
        }
        self.fast = {**fast_defaults, **self.config.get('fast_settings', {})}

    def detect_board(self, image: np.ndarray) -> Optional[np.ndarray]:
        # Determine downscale
        scale = 1.0
        resized = image
        if self.fast['use_downscale']:
            h, w = image.shape[:2]
            max_dim = max(h, w)
            if max_dim > self.fast['max_detection_dim']:
                scale = self.fast['max_detection_dim'] / max_dim
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Grayscale and edges on resized
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edges = self._detect_edges(gray)

        # Lines
        lines = self._find_lines(edges, scale)
        if lines is None or len(lines) == 0:
            return None

        # Filter by angle
        vertical_lines, horizontal_lines = self._filter_lines_by_angle(lines)
        if len(vertical_lines) == 0 or len(horizontal_lines) == 0:
            return None

        # Keep only top-K longest lines per direction
        max_k = int(self.fast['max_lines_per_direction'])
        vertical_lines = self._keep_top_k_by_length(vertical_lines, max_k)
        horizontal_lines = self._keep_top_k_by_length(horizontal_lines, max_k)

        # Precompute coords and distance matrices
        line_data = self._precompute_line_data(vertical_lines, horizontal_lines)
        if line_data['all_distances'] is None or len(line_data['all_distances']) == 0:
            return None

        # Frequent distances (histogram-based, vectorized)
        frequent_distances = self._find_frequent_distances_fast(line_data['all_distances'])
        if not frequent_distances:
            return None

        # Validate distance multiples to infer square size
        valid_square_side_length = self._find_valid_square_size_optimized(line_data, frequent_distances)
        if valid_square_side_length is None:
            return None

        # Vectorized line selection using distance matrices
        selected_vertical_lines = self._select_lines_vectorized(
            vertical_lines,
            line_data['v_distance_matrix'],
            valid_square_side_length,
        )
        selected_horizontal_lines = self._select_lines_vectorized(
            horizontal_lines,
            line_data['h_distance_matrix'],
            valid_square_side_length,
        )

        # Boundary filtering, fewer iterations
        if len(selected_vertical_lines) > 0 and len(selected_horizontal_lines) > 0:
            buffer_pixels = self.config['line_selection']['boundary_buffer_pixels']
            for _ in range(int(self.fast['boundary_filter_iterations'])):
                prev_h = len(selected_horizontal_lines)
                prev_v = len(selected_vertical_lines)
                selected_horizontal_lines, selected_vertical_lines = self._filter_lines_by_boundaries(
                    selected_horizontal_lines, selected_vertical_lines, buffer_pixels
                )
                if len(selected_horizontal_lines) == prev_h and len(selected_vertical_lines) == prev_v:
                    break

        # Duplicate removal (greedy coordinate clustering)
        distance_threshold = self.config['duplicate_removal']['distance_threshold']
        if len(selected_vertical_lines) > 0:
            selected_vertical_lines = self._remove_duplicates_fast(selected_vertical_lines, direction='vertical', distance_threshold=distance_threshold)
        if len(selected_horizontal_lines) > 0:
            selected_horizontal_lines = self._remove_duplicates_fast(selected_horizontal_lines, direction='horizontal', distance_threshold=distance_threshold)

        # Final validation
        min_lines = self.config['board_validation']['min_lines_required']
        if len(selected_vertical_lines) < min_lines or len(selected_horizontal_lines) < min_lines:
            return None

        # Compute corners on resized space
        corners_resized = self._calculate_board_corners(selected_vertical_lines, selected_horizontal_lines)
        if corners_resized is None:
            return None

        # Map corners back to original resolution
        if scale != 1.0:
            corners = (corners_resized.astype(np.float32) / scale).astype(np.float32)
        else:
            corners = corners_resized.astype(np.float32)

        # Extract on original image
        board = self._extract_board(image, corners)
        return board

    # ---------- Helpers ----------

    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        kernel_size = tuple(self.config['edge_detection']['gaussian_blur']['kernel_size'])
        sigma = self.config['edge_detection']['gaussian_blur']['sigma']
        blurred = cv2.GaussianBlur(gray, kernel_size, sigma)
        median = float(np.median(blurred))
        lower = int(max(0, 0.5 * median))
        upper = int(min(255, 0.9 * median))
        aperture_size = int(self.config['edge_detection']['canny']['aperture_size'])
        edges = cv2.Canny(blurred, lower, upper, apertureSize=aperture_size)
        return edges

    def _find_lines(self, edges: np.ndarray, scale: float) -> Optional[np.ndarray]:
        params = self.config['line_detection']['hough_lines']
        # Scale-dependent params for resized image
        min_line_length = max(1, int(round(params['min_line_length'] * scale)))
        max_line_gap = max(1, int(round(params['max_line_gap'] * scale)))

        lines = cv2.HoughLinesP(
            edges,
            rho=params['rho'],
            theta=np.pi / params['theta_resolution'],
            threshold=params['threshold'],
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )
        if lines is None:
            return None
        return lines.reshape(-1, 4)

    def _filter_lines_by_angle(self, lines: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vertical_lines = []
        horizontal_lines = []
        angle_tolerance = self.config['line_filtering']['angle_tolerance']
        for line in lines:
            x1, y1, x2, y2 = line
            angle = float(abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi))
            if 90 - angle_tolerance < angle < 90 + angle_tolerance:
                vertical_lines.append(line)
            elif angle < angle_tolerance or angle > 180 - angle_tolerance:
                horizontal_lines.append(line)
        return np.array(vertical_lines), np.array(horizontal_lines)

    def _keep_top_k_by_length(self, lines: np.ndarray, k: int) -> np.ndarray:
        if len(lines) <= k:
            return lines
        # Compute lengths
        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        lengths = np.sqrt(dx * dx + dy * dy)
        idx = np.argsort(lengths)[-k:]
        return lines[idx]

    def _precompute_line_data(self, vertical_lines: np.ndarray, horizontal_lines: np.ndarray) -> dict:
        v_coords = np.array([(line[0] + line[2]) / 2 for line in vertical_lines]) if len(vertical_lines) > 0 else np.array([])
        h_coords = np.array([(line[1] + line[3]) / 2 for line in horizontal_lines]) if len(horizontal_lines) > 0 else np.array([])

        v_distance_matrix = np.abs(v_coords[:, None] - v_coords) if len(v_coords) > 0 else np.array([[]])
        h_distance_matrix = np.abs(h_coords[:, None] - h_coords) if len(h_coords) > 0 else np.array([[]])

        all_distances = []
        if len(v_coords) >= 2:
            v_distances = v_distance_matrix[np.triu_indices_from(v_distance_matrix, k=1)]
            all_distances.extend(v_distances[v_distances > 0])
        if len(h_coords) >= 2:
            h_distances = h_distance_matrix[np.triu_indices_from(h_distance_matrix, k=1)]
            all_distances.extend(h_distances[h_distances > 0])

        return {
            'v_coords': v_coords,
            'h_coords': h_coords,
            'v_distance_matrix': v_distance_matrix,
            'h_distance_matrix': h_distance_matrix,
            'all_distances': np.array(all_distances, dtype=np.float32),
        }

    def _find_frequent_distances_fast(self, distances: np.ndarray) -> List[float]:
        if distances is None or len(distances) == 0:
            return []
        tol = float(self.config['distance_analysis']['tolerance_pixels'])
        min_instances = int(self.config['distance_analysis']['min_instances'])
        # Quantize distances into bins of size ~ tolerance
        bins = np.round(distances / tol)
        values, counts = np.unique(bins, return_counts=True)
        good_bins = values[counts >= min_instances]
        # Map bins back to representative distances (center of bin)
        frequent = list(sorted((good_bins * tol).tolist()))
        return frequent

    def _find_valid_square_size_optimized(self, line_data: dict, frequent_distances: List[float]) -> Optional[float]:
        min_square_size = self.config['distance_analysis']['min_square_size']
        tolerance = self.config['distance_analysis']['validation_tolerance_pixels']
        valid_candidates = []
        distances_array = np.array(line_data['all_distances'])
        for candidate in frequent_distances:
            if self._validate_distance_multiples_optimized(distances_array, candidate, tolerance):
                valid_candidates.append(candidate)
        reasonable_candidates = [c for c in valid_candidates if c >= min_square_size]
        if reasonable_candidates:
            return float(reasonable_candidates[0])
        elif valid_candidates:
            return float(valid_candidates[-1])
        return None

    def _validate_distance_multiples_optimized(self, distances: np.ndarray, candidate: float, tolerance: float) -> bool:
        expected_multiples = [2, 3, 4]
        found = 0
        for m in expected_multiples:
            expected = candidate * m
            if np.any(np.abs(distances - expected) <= tolerance):
                found += 1
        return found >= 2

    def _select_lines_vectorized(self, lines: np.ndarray, distance_matrix: np.ndarray, square_side_length: float) -> np.ndarray:
        if len(lines) == 0:
            return np.array([])
        tol = float(self.config['line_selection']['tolerance_pixels'])
        tol_mult = float(self.config['line_selection']['tolerance_multiplier'])
        min_matching = int(self.config['line_selection']['min_matching_distances'])
        min_small_matching = int(self.config['line_selection']['min_small_matching_distances'])

        D = distance_matrix.astype(np.float32)
        # multiples rounded
        with np.errstate(divide='ignore', invalid='ignore'):
            M = np.rint(D / float(square_side_length))
        positive_mask = (M > 0) & (D > 0)
        expected = M * float(square_side_length)
        error = np.abs(D - expected)
        allowed = positive_mask & (error <= (np.maximum(M, 1) * tol_mult * tol))
        # Counts per line (row)
        counts_all = allowed.sum(axis=1)
        counts_small = (allowed & (M <= 3)).sum(axis=1)
        keep_mask = (counts_all >= min_matching) & (counts_small >= min_small_matching)
        return lines[keep_mask]

    def _filter_lines_by_boundaries(self, horizontal_lines: np.ndarray, vertical_lines: np.ndarray, buffer_pixels: float) -> Tuple[np.ndarray, np.ndarray]:
        if len(vertical_lines) == 0 or len(horizontal_lines) == 0:
            return horizontal_lines, vertical_lines
        v_coords = [(line[0] + line[2]) / 2 for line in vertical_lines]
        min_x, max_x = min(v_coords) - buffer_pixels, max(v_coords) + buffer_pixels
        h_coords = [(line[1] + line[3]) / 2 for line in horizontal_lines]
        min_y, max_y = min(h_coords) - buffer_pixels, max(h_coords) + buffer_pixels
        # Filter horizontal
        h_keep = []
        for line in horizontal_lines:
            line_x_min, line_x_max = min(line[0], line[2]), max(line[0], line[2])
            if not (line_x_max < min_x or line_x_min > max_x):
                h_keep.append(line)
        # Filter vertical
        v_keep = []
        for line in vertical_lines:
            line_y_min, line_y_max = min(line[1], line[3]), max(line[1], line[3])
            if not (line_y_max < min_y or line_y_min > max_y):
                v_keep.append(line)
        return np.array(h_keep), np.array(v_keep)

    def _remove_duplicates_fast(self, lines: np.ndarray, direction: str, distance_threshold: float = 5.0) -> np.ndarray:
        if len(lines) == 0:
            return lines
        # Coordinate per direction
        if direction == 'vertical':
            coords = np.array([(line[0] + line[2]) / 2 for line in lines])
        else:
            coords = np.array([(line[1] + line[3]) / 2 for line in lines])
        # Lengths for tie-breaker
        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        lengths = np.sqrt(dx * dx + dy * dy)

        order = np.argsort(coords)
        coords_sorted = coords[order]
        lines_sorted = lines[order]
        lengths_sorted = lengths[order]

        kept = []
        i = 0
        n = len(lines_sorted)
        while i < n:
            # Start new cluster
            cluster_indices = [i]
            j = i + 1
            while j < n and abs(coords_sorted[j] - coords_sorted[i]) <= distance_threshold:
                cluster_indices.append(j)
                j += 1
            # Pick the longest line in the cluster
            cluster_lengths = lengths_sorted[cluster_indices]
            best_local_idx = int(cluster_indices[int(np.argmax(cluster_lengths))])
            kept.append(lines_sorted[best_local_idx])
            i = j
        return np.array(kept)

    def _calculate_board_corners(self, vertical_lines: np.ndarray, horizontal_lines: np.ndarray) -> Optional[np.ndarray]:
        if len(vertical_lines) < 2 or len(horizontal_lines) < 2:
            return None
        v_coords = [(line[0] + line[2]) / 2 for line in vertical_lines]
        h_coords = [(line[1] + line[3]) / 2 for line in horizontal_lines]
        min_v_idx = int(np.argmin(v_coords))
        max_v_idx = int(np.argmax(v_coords))
        min_h_idx = int(np.argmin(h_coords))
        max_h_idx = int(np.argmax(h_coords))
        left_line = vertical_lines[min_v_idx]
        right_line = vertical_lines[max_v_idx]
        top_line = horizontal_lines[min_h_idx]
        bottom_line = horizontal_lines[max_h_idx]
        corners = []
        for v_line, h_line in [
            (left_line, top_line),
            (right_line, top_line),
            (right_line, bottom_line),
            (left_line, bottom_line),
        ]:
            inter = self._line_intersection(v_line, h_line)
            if inter is None:
                return None
            corners.append(inter)
        return np.array(corners, dtype=np.float32)

    def _line_intersection(self, line1: np.ndarray, line2: np.ndarray) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return np.array([x, y], dtype=np.float32)

    def _extract_board(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        board_size = int(self.config['board_extraction']['output_size'])
        target = np.array([
            [0, 0],
            [board_size, 0],
            [board_size, board_size],
            [0, board_size],
        ], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target)
        board = cv2.warpPerspective(image, matrix, (board_size, board_size))
        return board 