"""
Unified Scenario Generator for CrowdES Benchmark

Generates scenario maps for multiple group behavior types from a single config.
Supports: Queue, Cohesive Walking, Gathering
"""

import cv2
import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy.spatial import cKDTree


class UnifiedScenarioGenerator:
    """
    Generate scenario maps for all enabled group behaviors.

    Usage:
        generator = UnifiedScenarioGenerator('configs/scenarios.yaml')
        scenario_maps = generator.generate_from_carla(semantic_map, walkable_mask)
        # Returns: {'queue': map, 'cohesive': map, 'gathering': map}
    """

    def __init__(self, config_path: str):
        """Load unified config."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.enabled_scenarios = {
            name: params['enabled']
            for name, params in self.config['scenarios'].items()
        }


    def generate_from_carla(self,
                           semantic_map: np.ndarray,
                           walkable_mask: np.ndarray,
                           pixel_per_meter: float = 10.0,
                           world=None,
                           camera_transform=None,
                           ortho_size: float = None,
                           image_resolution: int = None,
                           buildings: list = None,
                           bus_stops: list = None,
                           food_carts: list = None,
                           crosswalks: list = None) -> Dict[str, np.ndarray]:
        """
        Generate scenario maps from CARLA semantic segmentation.

        Args:
            semantic_map: (H, W) CARLA semantic segmentation (class IDs)
            walkable_mask: (H, W) binary mask of walkable areas
            pixel_per_meter: Scale factor
            buildings: DEPRECATED - Not used (bbox inaccurate)
            bus_stops: List of queue anchor dicts with 'goal_point', 'direction', 'length_px' (bus stops + food carts)
            food_carts: Unused — food carts are now routed to bus_stops as queue anchors

        Returns:
            scenario_maps: Dict with keys ['queue', 'cohesive', 'gathering']
                          Each value is (H, W) float32 density map [0, 1]
            metadata: Dict with scenario location metadata
        """
        H, W = walkable_mask.shape
        scenario_maps = {}
        metadata = {}

        # Queue scenarios
        if self.enabled_scenarios.get('queue', False):
            queue_map, queue_meta = self._generate_queue_map(
                semantic_map, walkable_mask, pixel_per_meter,
                camera_transform=camera_transform,
                ortho_size=ortho_size, image_resolution=image_resolution,
                buildings=buildings, bus_stops=bus_stops
            )
            scenario_maps['queue'] = queue_map
            metadata['queue'] = queue_meta
        else:
            scenario_maps['queue'] = np.zeros((H, W), dtype=np.float32)
            metadata['queue'] = []

        # Gathering scenarios
        if self.enabled_scenarios.get('gathering', False):
            gathering_map, gathering_meta = self._generate_gathering_map(
                semantic_map, walkable_mask, pixel_per_meter,
                food_carts=food_carts
            )
            scenario_maps['gathering'] = gathering_map
            metadata['gathering'] = gathering_meta
        else:
            scenario_maps['gathering'] = np.zeros((H, W), dtype=np.float32)
            metadata['gathering'] = []

        # Cohesive spawn areas (DEPRECATED - no longer generated)
        # Grouping is now done dynamically by GroupGenerator based on spawn/goal proximity
        metadata['cohesive'] = []

        # Crosswalk metadata (pass through from detection)
        metadata['crosswalk'] = crosswalks or []

        # Post-generation filtering: Remove gatherings that overlap with queues
        # Priority: queue > gathering (keep queue, remove overlapping gathering)
        metadata = self._filter_overlapping_scenarios(metadata, pixel_per_meter)

        return scenario_maps, metadata

    def _generate_queue_map(self,
                           semantic_map: np.ndarray,
                           walkable_mask: np.ndarray,
                           pixel_per_meter: float,
                           camera_transform=None,
                           ortho_size: float = None,
                           image_resolution: int = None,
                           buildings: list = None,
                           bus_stops: list = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate queue scenario map from bus stops (buildings DEPRECATED).

        Args:
            buildings: DEPRECATED - Not used (bbox inaccurate)
            bus_stops: List of bus stop dicts (already in pixel coords)

        Returns:
            queue_map: (H, W) density map with queue lines
            metadata: List of queue info dicts
        """
        H, W = walkable_mask.shape
        queue_map = np.zeros((H, W), dtype=np.float32)
        metadata = []

        config = self.config['queue']
        detection_mode = config.get('detection_mode', 'bus_stop')
        line_thickness = int(config.get('line_thickness_meters', 0.6) * pixel_per_meter)
        queue_length_meters = config.get('queue_length_meters', 10.0)
        queue_length_px = int(queue_length_meters * pixel_per_meter)

        # Check camera parameters (kept for future use, though not needed now)
        if camera_transform is None or ortho_size is None or image_resolution is None:
            pass
            return queue_map, metadata

        # Build KDTree for walkable pixel snapping
        walkable_coords = np.argwhere(walkable_mask > 0)
        if len(walkable_coords) == 0:
            pass
            return queue_map, metadata
        walkable_coords_xy = walkable_coords[:, [1, 0]]  # [y, x] to [x, y]
        kdtree = cKDTree(walkable_coords_xy)

        # Helper: Draw queue line with walkable check
        def draw_queue_line(start_x, start_y, direction, max_length, source_name):
            # Snap start point to walkable if needed
            if start_y < 0 or start_y >= H or start_x < 0 or start_x >= W:
                return start_x, start_y, 0

            if walkable_mask[start_y, start_x] == 0:
                dist, idx = kdtree.query([start_x, start_y])
                start_x, start_y = int(walkable_coords_xy[idx][0]), int(walkable_coords_xy[idx][1])

            # Walk along direction, stop at non-walkable
            current_x, current_y = float(start_x), float(start_y)
            step_size = 1.0  # pixel per step
            actual_length = 0

            for step_idx in range(int(max_length)):
                next_x = current_x + direction[0] * step_size
                next_y = current_y + direction[1] * step_size

                # Check bounds
                if not (0 <= int(next_x) < W and 0 <= int(next_y) < H):
                    break

                # Check walkable
                if walkable_mask[int(next_y), int(next_x)] == 0:
                    break

                current_x, current_y = next_x, next_y
                actual_length += step_size

            end_x, end_y = int(current_x), int(current_y)

            if actual_length > 0:
                cv2.line(queue_map, (start_x, start_y), (end_x, end_y),
                        color=1.0, thickness=line_thickness)

            return end_x, end_y, actual_length, start_x, start_y

        # 1. Process building entrances: DEPRECATED (bbox inaccurate)
        # Buildings are skipped - only bus stops and buses are used for queue

        # 2. Process bus stops (already in pixel coords from extract_carla_maps.py)
        if detection_mode in ['bus_stop', 'both'] and bus_stops is not None and len(bus_stops) > 0:
            for idx, bus_stop in enumerate(bus_stops):
                px, py = bus_stop['goal_point']
                direction = np.array(bus_stop['direction'])
                requested_length = bus_stop.get('length_px', queue_length_px)
                source_type = bus_stop.get('source', 'bus_stop')

                end_x, end_y, actual_length, snapped_x, snapped_y = draw_queue_line(
                    px, py, direction, requested_length, f"{source_type} '{bus_stop['name']}'"
                )

                if actual_length > 0:
                    metadata.append({
                        'type': 'queue',
                        'source': source_type,
                        'goal_point': [snapped_x, snapped_y],
                        'direction': direction.tolist(),
                        'length_px': actual_length,
                        'requested_length_px': requested_length,
                        'queue_id': bus_stop['name'],
                        'pixel_per_meter': bus_stop.get('pixel_per_meter', pixel_per_meter)
                    })

        pass  # queue counts logged in scene summary
        return queue_map, metadata

    def _generate_gathering_map(self,
                               semantic_map: np.ndarray,
                               walkable_mask: np.ndarray,
                               pixel_per_meter: float,
                               food_carts: list = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate gathering scenario map (open plaza/gathering points via 2D convolution).

        Food carts are now handled as queue anchors in extract_carla_maps.py.

        Args:
            semantic_map: (H, W) semantic segmentation
            walkable_mask: (H, W) binary mask
            pixel_per_meter: Scale factor
            food_carts: Unused (food carts are now queue anchors)

        Returns:
            gathering_map: (H, W) density map with circles
            metadata: List of gathering point info
        """
        H, W = walkable_mask.shape
        gathering_map = np.zeros((H, W), dtype=np.float32)
        metadata = []

        config = self.config['gathering']
        # Default visualization radius (will be replaced by adaptive radius during validation)
        default_radius = int(2.5 * pixel_per_meter)  # 2.5m default

        # Plaza detection using 2D convolution: find locations with max local walkable area
        from scipy.ndimage import maximum_filter
        from scipy.signal import fftconvolve  # FFT-based convolution (much faster for large kernels)

        # Get plaza detection parameters
        search_radius_m = config.get('search_radius_meters', 5.0)  # Search with 5m radius circle
        min_coverage = config.get('min_coverage_ratio', 0.7)  # At least 70% of kernel must be walkable
        max_gatherings = config.get('max_gathering_points', 5)  # Max gathering points per scene
        min_distance_m = config.get('min_distance_meters', 10.0)  # Min distance between gatherings

        search_radius_px = int(search_radius_m * pixel_per_meter)
        min_distance_px = int(min_distance_m * pixel_per_meter)

        pass  # parameters logged in scene summary

        # Create circular kernel
        y_grid, x_grid = np.ogrid[-search_radius_px:search_radius_px+1, -search_radius_px:search_radius_px+1]
        kernel = (x_grid**2 + y_grid**2 <= search_radius_px**2).astype(float)
        kernel_area = kernel.sum()

        # Convolve walkable mask with circular kernel to find local walkable density
        # Use FFT convolution (much faster: O(N log N) vs O(N²))
        walkable_float = (walkable_mask > 0).astype(float)
        coverage_map = fftconvolve(walkable_float, kernel, mode='same')

        # Normalize by kernel area to get coverage ratio [0, 1]
        coverage_ratio = coverage_map / kernel_area

        # Only consider locations with sufficient coverage
        valid_mask = coverage_ratio >= min_coverage

        # Find local maxima in coverage map
        local_max_map = maximum_filter(coverage_map, size=min_distance_px)
        is_local_max = (coverage_map == local_max_map) & valid_mask

        # Get candidate gathering centers sorted by coverage (descending)
        candidates_y, candidates_x = np.where(is_local_max)
        candidates_coverage = coverage_ratio[candidates_y, candidates_x]

        # Sort by coverage descending
        sorted_indices = np.argsort(candidates_coverage)[::-1]
        candidates_y = candidates_y[sorted_indices]
        candidates_x = candidates_x[sorted_indices]
        candidates_coverage = candidates_coverage[sorted_indices]

        # Select top gatherings with minimum distance constraint
        selected_centers = []
        for i in range(len(candidates_y)):
            if len(selected_centers) >= max_gatherings:
                break
            cy, cx = candidates_y[i], candidates_x[i]
            coverage = candidates_coverage[i]

            # Check minimum distance from already selected centers
            too_close = False
            for prev_cy, prev_cx, _ in selected_centers:
                dist = np.sqrt((cy - prev_cy)**2 + (cx - prev_cx)**2)
                if dist < min_distance_px:
                    too_close = True
                    break

            if not too_close:
                selected_centers.append((cy, cx, coverage))

        # Create gathering points from selected centers
        for cy, cx, coverage in selected_centers:
            # Draw placeholder circle (will be redrawn with adaptive radius after validation)
            cv2.circle(gathering_map, (cx, cy), default_radius,
                      color=1.0, thickness=-1)

            metadata.append({
                'type': 'gathering',
                'source': 'convolution',
                'center': [cx, cy],
                'pixel_per_meter': pixel_per_meter,
                'coverage_ratio': float(coverage),
                # Note: spawn_radius will be calculated during validation
            })

        pass  # gathering counts logged in scene summary
        return gathering_map, metadata

    def _filter_overlapping_scenarios(self, metadata: Dict[str, List[Dict]],
                                      pixel_per_meter: float) -> Dict[str, List[Dict]]:
        """
        Remove gatherings that overlap with queues.
        Priority: queue > gathering (keep queue, remove overlapping gathering)

        Args:
            metadata: Dict with keys 'queue', 'gathering', etc.
            pixel_per_meter: Scale factor for distance calculation

        Returns:
            Filtered metadata dict
        """
        queue_list = metadata.get('queue', [])
        gathering_list = metadata.get('gathering', [])

        if len(queue_list) == 0 or len(gathering_list) == 0:
            return metadata

        # Minimum distance threshold (meters)
        min_distance_meters = self.config.get('gathering', {}).get('queue_overlap_clearance_meters', 5.0)
        min_distance_px = min_distance_meters * pixel_per_meter

        def _point_to_segment_dist(point, seg_start, seg_end):
            seg = seg_end - seg_start
            seg_len_sq = np.dot(seg, seg)
            if seg_len_sq < 1e-6:
                return np.linalg.norm(point - seg_start)
            t = np.clip(np.dot(point - seg_start, seg) / seg_len_sq, 0.0, 1.0)
            return np.linalg.norm(point - (seg_start + t * seg))

        # Filter gatherings: remove any gathering too close to a queue line
        filtered_gatherings = []
        removed_count = 0

        for gather_info in gathering_list:
            gather_center = np.array(gather_info.get('center', [0, 0]), dtype=float)

            too_close = False
            for queue_info in queue_list:
                queue_goal = np.array(queue_info.get('goal_point', [0, 0]), dtype=float)
                direction  = np.array(queue_info.get('direction', [1, 0]), dtype=float)
                # Use walkable_length_px (set by validator) — the actual usable queue extent
                tail_px = float(queue_info.get('walkable_length_px',
                                               queue_info.get('length_px', 0)))
                queue_tail = queue_goal + direction * tail_px

                distance = _point_to_segment_dist(gather_center, queue_goal, queue_tail)
                if distance < min_distance_px:
                    too_close = True
                    removed_count += 1
                    break

            if not too_close:
                filtered_gatherings.append(gather_info)

        if removed_count > 0:
            print(f"  [Filter] Removed {removed_count}/{len(gathering_list)} gatherings due to queue overlap")

        metadata['gathering'] = filtered_gatherings
        return metadata

    def combine_scenario_maps(self,
                            scenario_maps: Dict[str, np.ndarray],
                            weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Combine multiple scenario maps into single density map.

        Args:
            scenario_maps: Dict of {scenario_type: density_map}
            weights: Optional weights for each scenario type

        Returns:
            combined: (H, W) combined density map
        """
        if weights is None:
            weights = {k: 1.0 for k in scenario_maps.keys()}

        H, W = next(iter(scenario_maps.values())).shape
        combined = np.zeros((H, W), dtype=np.float32)

        for scenario_type, density_map in scenario_maps.items():
            weight = weights.get(scenario_type, 1.0)
            combined += weight * density_map

        # Normalize to [0, 1]
        if combined.max() > 0:
            combined = combined / combined.max()

        return combined


# Example usage
if __name__ == "__main__":
    print("UnifiedScenarioGenerator Test")
    print("=" * 60)

    # Load generator
    generator = UnifiedScenarioGenerator('./topourbanbench/generation/crowdes/configs/scenarios.yaml')

    # Dummy CARLA data
    H, W = 512, 512
    semantic_map = np.zeros((H, W), dtype=np.uint8)
    walkable_mask = np.ones((H, W), dtype=np.uint8)

    # Add fake sidewalk (buildings not used anymore)
    semantic_map[200:400, 50:450] = 2    # Sidewalk

    # Generate
    print("\nGenerating scenario maps...")
    scenario_maps, metadata = generator.generate_from_carla(
        semantic_map, walkable_mask, pixel_per_meter=10.0
    )

    print("\nGenerated maps:")
    for name, map_data in scenario_maps.items():
        print(f"  {name}: shape={map_data.shape}, max={map_data.max():.3f}")

    print("\nMetadata:")
    for name, meta in metadata.items():
        print(f"  {name}: {len(meta)} locations")

    print("\n" + "=" * 60)
    print("✓ UnifiedScenarioGenerator ready")
