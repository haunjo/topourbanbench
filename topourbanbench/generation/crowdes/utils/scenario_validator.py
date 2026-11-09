"""
Scenario Validator for CrowdES

Validates queue and gathering points against walkable mask and calculates
adaptive gathering radius based on local walkable area.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple


def calculate_adaptive_radius(center_x: int, center_y: int,
                              walkable_mask: np.ndarray,
                              min_radius_px: float,
                              max_radius_px: float,
                              safety_margin: float = 0.7) -> Optional[int]:
    """
    Calculate adaptive gathering radius by sampling walkable area in 8 directions.

    Args:
        center_x, center_y: Center point coordinates (pixels)
        walkable_mask: Binary walkable mask (H, W)
        min_radius_px: Minimum acceptable radius (pixels)
        max_radius_px: Maximum radius limit (pixels)
        safety_margin: Use this fraction of available space (default 0.7 = 70%)

    Returns:
        adaptive_radius: Calculated radius (pixels), or None if insufficient space
    """
    H, W = walkable_mask.shape

    # Sample 8 directions
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)

    walkable_distances = []
    for angle in angles:
        dx = np.cos(angle)
        dy = np.sin(angle)

        # Ray cast in this direction until hitting non-walkable
        max_search = int(max_radius_px + 50)
        for r in range(1, max_search):
            x = int(center_x + dx * r)
            y = int(center_y + dy * r)

            # Bounds check
            if not (0 <= y < H and 0 <= x < W):
                walkable_distances.append(r)
                break

            # Walkable check
            if walkable_mask[y, x] == 0:
                walkable_distances.append(r)
                break
        else:
            # Reached max_search without hitting wall
            walkable_distances.append(max_search)

    if len(walkable_distances) == 0:
        return None

    # Use minimum distance (narrowest direction determines max safe radius)
    min_distance = min(walkable_distances)

    # Check if meets minimum requirement
    if min_distance < min_radius_px:
        return None

    # Apply safety margin
    adaptive_radius = int(min_distance * safety_margin)

    # Clamp to max
    adaptive_radius = min(adaptive_radius, int(max_radius_px))

    return adaptive_radius


def validate_scenario_points(scenario_metadata: Dict,
                             walkable_mask: np.ndarray,
                             pixel_per_meter: float,
                             gathering_config: Dict,
                             queue_config: Dict = None) -> Dict:
    """
    Validate queue and gathering points against walkable mask.

    Queue validation:
        - Goal point must be walkable
        - Direction must have minimum walkable length

    Gathering validation:
        - Center must be walkable
        - Calculate adaptive radius based on local walkable area

    Args:
        scenario_metadata: Dict with 'queue', 'gathering', 'crosswalk' lists
        walkable_mask: Binary walkable mask (H, W)
        pixel_per_meter: Scale factor
        gathering_config: Gathering config dict from scenarios.yaml

    Returns:
        validated_metadata: Filtered and updated metadata dict
    """
    validated = {
        'queue': [],
        'gathering': [],
        'crosswalk': scenario_metadata.get('crosswalk', [])
    }

    H, W = walkable_mask.shape

    # ==================== Queue Validation ====================
    pass

    _q_cfg = queue_config or {}
    q_safety_margin    = float(_q_cfg.get('safety_margin_ratio', 0.6))
    q_min_length_px    = float(_q_cfg.get('min_length_meters', 1.6)) * pixel_per_meter
    q_max_length_px    = float(_q_cfg.get('max_length_meters', 8.0)) * pixel_per_meter
    q_gap_tolerance_px = int(float(_q_cfg.get('scan_gap_tolerance_meters', 0.2)) * pixel_per_meter)

    for queue in scenario_metadata.get('queue', []):
        queue_id = queue.get('queue_id', queue.get('name', 'unknown'))
        goal_point = queue['goal_point']
        direction = np.array(queue['direction'])
        length_px = queue.get('length_px', queue.get('requested_length_px', 400))

        x_start, y_start = int(goal_point[0]), int(goal_point[1])

        # Bounds check for start point
        if not (0 <= y_start < H and 0 <= x_start < W):
            pass
            continue

        # Create queue line mask (draw the queue line)
        queue_mask = np.zeros((H, W), dtype=np.uint8)
        x_end = int(x_start + direction[0] * length_px)
        y_end = int(y_start + direction[1] * length_px)

        # Clamp end point to image bounds
        x_end = max(0, min(W - 1, x_end))
        y_end = max(0, min(H - 1, y_end))

        # Draw queue line with thickness (same as in scenario generation)
        cv2.line(queue_mask, (x_start, y_start), (x_end, y_end),
                 255, thickness=30)

        # Check overlap between queue mask and walkable mask
        overlap = cv2.bitwise_and(queue_mask, walkable_mask)
        overlap_pixels = np.count_nonzero(overlap)

        # Minimum overlap requirement: at least 30% of queue line must be on walkable
        queue_pixels = np.count_nonzero(queue_mask)
        overlap_ratio = overlap_pixels / queue_pixels if queue_pixels > 0 else 0

        min_overlap_ratio = 0.3  # At least 30% must overlap with walkable

        if overlap_ratio < min_overlap_ratio:
            continue

        # Scan walkable length along queue direction with gap tolerance.
        # Small gaps (lamp posts, seg noise) up to q_gap_tolerance_px are bridged;
        # only a run of consecutive non-walkable pixels exceeding the tolerance stops the scan.
        walkable_length_px = 0
        consecutive_gap    = 0
        for step in range(int(length_px)):
            test_x = int(x_start + direction[0] * step)
            test_y = int(y_start + direction[1] * step)

            if not (0 <= test_y < H and 0 <= test_x < W):
                break
            if walkable_mask[test_y, test_x] > 0:
                walkable_length_px = step + 1  # extend to this walkable pixel
                consecutive_gap    = 0
            else:
                consecutive_gap += 1
                if consecutive_gap > q_gap_tolerance_px:
                    break  # real barrier — stop

        if walkable_length_px < q_min_length_px:
            continue

        # Apply safety margin and clamp, analogous to gathering's adaptive radius
        adaptive_length_px = int(walkable_length_px * q_safety_margin)
        adaptive_length_px = max(int(q_min_length_px), min(adaptive_length_px, int(q_max_length_px)))

        queue['walkable_length_px'] = adaptive_length_px
        validated['queue'].append(queue)
    # ==================== Gathering Validation ====================

    # Load gathering config
    min_radius_meters = gathering_config.get('min_radius_meters', 1.5)
    max_radius_meters = gathering_config.get('max_radius_meters', 4.0)
    safety_margin = gathering_config.get('safety_margin_ratio', 0.7)

    min_radius_px = min_radius_meters * pixel_per_meter
    max_radius_px = max_radius_meters * pixel_per_meter

    for gathering in scenario_metadata.get('gathering', []):
        name = gathering.get('name', 'unknown')
        center = gathering['center']
        cx, cy = int(center[0]), int(center[1])

        # Bounds check
        if not (0 <= cy < H and 0 <= cx < W):
            continue

        # Center walkable check
        if walkable_mask[cy, cx] == 0:
            continue

        # Calculate adaptive radius
        adaptive_radius = calculate_adaptive_radius(
            cx, cy, walkable_mask,
            min_radius_px, max_radius_px, safety_margin
        )

        if adaptive_radius is None:
            continue

        # Update gathering metadata with adaptive radius
        gathering['spawn_radius'] = adaptive_radius

        validated['gathering'].append(gathering)


    return validated


if __name__ == "__main__":
    # Test with dummy data
    print("Testing scenario validator...")

    # Create dummy walkable mask (100x100 with 60x60 plaza in center)
    walkable = np.zeros((100, 100), dtype=np.uint8)
    walkable[20:80, 20:80] = 255

    # Test gathering in center (should get large radius)
    metadata = {
        'queue': [],
        'gathering': [{
            'center': [50, 50],
            'name': 'test_plaza'
        }],
        'crosswalk': []
    }

    config = {
        'min_radius_meters': 1.5,
        'max_radius_meters': 4.0,
        'safety_margin_ratio': 0.7
    }

    validated = validate_scenario_points(metadata, walkable, 10.0, config)

    print(f"\nResult: {validated['gathering']}")
    if len(validated['gathering']) > 0:
        print(f"Adaptive radius: {validated['gathering'][0]['spawn_radius']}px")
