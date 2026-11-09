"""
A* pathfinding with occupancy-aware cost for expert trajectory generation.

Generates clean, efficient paths on navmesh while avoiding predicted group zones.
"""

import numpy as np
import heapq
from scipy.ndimage import distance_transform_edt
from typing import Tuple, List, Optional, Dict


def astar_pathfind(
    navmesh: np.ndarray,
    start_px: Tuple[float, float],
    goal_px: Tuple[float, float],
    cost_map: Optional[np.ndarray] = None,
    diagonal: bool = True,
) -> Optional[np.ndarray]:
    """
    A* pathfinding on navmesh with optional occupancy cost.

    Args:
        navmesh: Binary walkable map (H x W), 1 = walkable, 0 = obstacle
        start_px: Start position (x, y) in pixel coordinates
        goal_px: Goal position (x, y) in pixel coordinates
        cost_map: Optional cost map (H x W), higher = avoid
        diagonal: Allow diagonal movement

    Returns:
        Path as numpy array (N x 2) in pixel coordinates, or None if no path
    """
    h, w = navmesh.shape

    # Convert to grid coordinates
    start = (int(round(start_px[1])), int(round(start_px[0])))  # (row, col)
    goal = (int(round(goal_px[1])), int(round(goal_px[0])))

    # Bounds check
    if not (0 <= start[0] < h and 0 <= start[1] < w):
        return None
    if not (0 <= goal[0] < h and 0 <= goal[1] < w):
        return None
    if navmesh[start] == 0 or navmesh[goal] == 0:
        return None

    # Initialize
    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    f_score = {start: _heuristic(start, goal)}

    # Movement directions (8-connected if diagonal, else 4-connected)
    if diagonal:
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # cardinal
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # diagonal
        ]
        costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]
    else:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        costs = [1.0, 1.0, 1.0, 1.0]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = _reconstruct_path(came_from, current)
            # Convert to pixel coordinates (x, y)
            path_px = np.array([[p[1], p[0]] for p in path], dtype=float)
            return path_px

        for (dr, dc), move_cost in zip(directions, costs):
            neighbor = (current[0] + dr, current[1] + dc)

            # Bounds check
            if not (0 <= neighbor[0] < h and 0 <= neighbor[1] < w):
                continue

            # Walkable check
            if navmesh[neighbor] == 0:
                continue

            # Compute cost
            tentative_g = g_score[current] + move_cost

            # Add occupancy cost if provided
            if cost_map is not None:
                occupancy_cost = float(cost_map[neighbor])
                tentative_g += occupancy_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + _heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    # No path found
    return None


def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Euclidean distance heuristic."""
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def _reconstruct_path(came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Reconstruct path from came_from dict."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def build_occupancy_map(
    navmesh: np.ndarray,
    scenario_metadata: Dict,
    base_cost: float = 100.0,
    gathering_bypass_margin_meters: float = 1.0,
    pixel_per_meter: float = 10.0,
    border_margin_meters: float = 1.0,
    border_cost: float = 50.0,
) -> np.ndarray:
    """
    Build occupancy cost map from scenario metadata.

    Args:
        navmesh: Binary walkable map (H x W)
        scenario_metadata: Dict with 'queue', 'gathering', 'cohesive', 'crosswalk'
        base_cost: Cost to add in occupied zones

    Returns:
        Cost map (H x W), same shape as navmesh
    """
    h, w = navmesh.shape
    cost_map = np.zeros((h, w), dtype=float)

    # Queue zones: rectangular corridor along actual queue line
    for q in scenario_metadata.get('queue', []):
        goal_point = q.get('goal_point')
        direction = q.get('direction')
        if goal_point is None or direction is None:
            continue

        ppm_q = float(q.get('pixel_per_meter', 10.0))
        # Use validated adaptive length if available; fall back to raw scan bound
        length_px = float(q.get('walkable_length_px', q.get('length_px', 5 * ppm_q)))
        width_px  = 1.2 * ppm_q                             # ~person width (1.2m)

        # Queue zone: rectangle starting at goal_point, extending along direction
        # Shift center to midpoint of the line
        dir_arr = np.array(direction, dtype=float)
        norm = np.linalg.norm(dir_arr)
        if norm < 1e-6:
            continue
        dir_arr /= norm
        mid = np.array(goal_point, dtype=float) + dir_arr * (length_px / 2.0)
        _add_queue_zone(cost_map, mid.tolist(), direction, width=width_px, length=length_px, cost=base_cost)

    # Gathering zones: circular
    for g in scenario_metadata.get('gathering', []):
        center = g.get('center')
        if center is None:
            continue

        ppm = float(g.get('pixel_per_meter', 10.0))
        spawn_radius = float(g.get('spawn_radius', 30.0))
        margin_px = gathering_bypass_margin_meters * ppm
        _add_circle_zone(cost_map, center, spawn_radius + margin_px, cost=base_cost)

    # Cohesive zones: skipped — agents are dynamically assigned at runtime,
    # their corridors cannot be predicted at A* planning time.

    # Walkable border cost: penalize cells near walls/curbs to prevent wall-hugging.
    # Uses a linear gradient: full border_cost at wall → 0 at border_margin_meters.
    if border_margin_meters > 0 and border_cost > 0:
        border_px = border_margin_meters * pixel_per_meter
        # distance_transform_edt: for each walkable pixel, distance to nearest non-walkable
        dist_to_wall = distance_transform_edt(navmesh.astype(bool))
        # Linear falloff: 1.0 at wall (dist=0), 0.0 at dist=border_px
        border_weight = np.clip(1.0 - dist_to_wall / border_px, 0.0, 1.0)
        cost_map += border_cost * border_weight * (navmesh > 0)

    return cost_map


def _add_queue_zone(cost_map: np.ndarray, center: List[float], direction: List[float],
                     width: float, length: float, cost: float):
    """Add rectangular queue zone to cost map using numpy vectorization."""
    h, w = cost_map.shape
    cx, cy = center
    dx, dy = direction

    norm = np.sqrt(dx**2 + dy**2)
    if norm < 1e-6:
        return
    dx, dy = dx / norm, dy / norm
    px, py = -dy, dx  # perpendicular

    x_min = int(max(0, cx - length))
    x_max = int(min(w, cx + length))
    y_min = int(max(0, cy - length))
    y_max = int(min(h, cy + length))

    xs = np.arange(x_min, x_max)
    ys = np.arange(y_min, y_max)
    xg, yg = np.meshgrid(xs, ys)  # (rows, cols)

    vec_x = xg - cx
    vec_y = yg - cy
    along = vec_x * dx + vec_y * dy
    perp = np.abs(vec_x * px + vec_y * py)

    mask = (-length / 2 <= along) & (along <= length / 2) & (perp <= width / 2)
    cost_map[y_min:y_max, x_min:x_max][mask] += cost


def _add_circle_zone(cost_map: np.ndarray, center: List[float], radius: float, cost: float):
    """Add circular zone to cost map using numpy vectorization."""
    h, w = cost_map.shape
    cx, cy = center

    x_min = int(max(0, cx - radius - 1))
    x_max = int(min(w, cx + radius + 1))
    y_min = int(max(0, cy - radius - 1))
    y_max = int(min(h, cy + radius + 1))

    xs = np.arange(x_min, x_max)
    ys = np.arange(y_min, y_max)
    xg, yg = np.meshgrid(xs, ys)

    dist_sq = (xg - cx) ** 2 + (yg - cy) ** 2
    mask = dist_sq <= radius ** 2
    cost_map[y_min:y_max, x_min:x_max][mask] += cost


def interpolate_path_at_fps(
    path_px: np.ndarray,
    speed_mps: float,
    pixel_per_meter: float,
    fps: int,
) -> np.ndarray:
    """
    Interpolate A* waypoints to uniform FPS trajectory.

    Args:
        path_px: Waypoints (N x 2) in pixel coordinates
        speed_mps: Walking speed in m/s (e.g., 1.2)
        pixel_per_meter: Pixels per meter (e.g., 10.0)
        fps: Target FPS (e.g., 20)

    Returns:
        Interpolated trajectory (M x 2) at target FPS
    """
    if len(path_px) < 2:
        return path_px

    # Compute cumulative distances
    segments = np.diff(path_px, axis=0)
    seg_lengths = np.linalg.norm(segments, axis=1)
    cum_dist = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_dist = cum_dist[-1]

    # Convert speed to px/frame
    speed_px_per_frame = speed_mps * pixel_per_meter / fps

    # Number of frames needed
    num_frames = max(2, int(np.ceil(total_dist / speed_px_per_frame)))

    # Sample uniformly along path
    sample_distances = np.linspace(0, total_dist, num_frames)
    interpolated = []

    for d in sample_distances:
        # Find segment
        idx = np.searchsorted(cum_dist, d) - 1
        idx = max(0, min(idx, len(path_px) - 2))

        # Local interpolation
        d_start = cum_dist[idx]
        d_end = cum_dist[idx + 1]
        if d_end - d_start < 1e-6:
            t = 0.0
        else:
            t = (d - d_start) / (d_end - d_start)

        pos = path_px[idx] * (1 - t) + path_px[idx + 1] * t
        interpolated.append(pos)

    return np.array(interpolated, dtype=float)
