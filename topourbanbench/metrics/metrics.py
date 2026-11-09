import numpy as np
from typing import Dict, List, Optional, Set, Tuple


class TopologicalMetricEngine:
    """
    Unified topological metrics for TopoUrbanBench (CoRL 2026).

    Implements paper-compliant metrics:
    - QCR (Queue Cutting Rate): number of distinct queues cut at least once per episode
    - GDS (Gathering Disruption Score): seconds spent inside a gathering's O-space
    - CPI (Cohesive Proxemic Intrusion): seconds spent in R-space buffer around cohesive groups

    Usage:
        Training (RL): Use evaluate_frame() to get per-step penalty
        Evaluation (Benchmark): Use get_episode_metrics() for final metrics
    """

    def __init__(self, robot_radius: float = 0.3, intrusion_threshold: float = 0.8,
                 cpi_buffer: float = 1.2):
        """
        Args:
            robot_radius: Robot physical radius in meters (used for RL proximity penalty)
            intrusion_threshold: Distance threshold for edge intrusion (meters)
                               Default: 0.8m = robot_radius(0.3m) + personal_space(0.5m)
            cpi_buffer: Buffer zone distance (D_buffer) for CPI calculation in meters
                       Default: 1.2m (boundary between Hall's personal and social distances)
        """
        self.robot_radius = robot_radius
        self.intrusion_threshold = intrusion_threshold
        self.cpi_buffer = cpi_buffer

        # Reset episode tracking
        self.reset()

    def reset(self, group_graph=None):
        """Reset tracking for new episode.

        Args:
            group_graph: Optional group_graph dict from JSON. If provided,
                         initializes denominators and pre-computes edge caches.
        """
        self.qcr_total_cuts = 0            # number of distinct queues cut (≤ num queues in scene)
        self.gds_total_time = 0.0
        self.cpi_total_time = 0.0
        self.total_frames = 0
        self.intrusion_frames = 0
        self.is_currently_intruding = False
        self.total_intrusion_events = 0

        self.ego_prev = None
        self._qcr_cut_queues: Set[int] = set()  # group_ids already counted; re-cutting same queue ignored

        self._edge_cache = {}  # {group_id: [(agent_id_a, agent_id_b), ...]}

        if group_graph is not None:
            for g in group_graph.get('groups', []):
                gid = g['group_id']
                member_ids = g['member_ids']
                edges = g.get('edges', [])
                self._edge_cache[gid] = [
                    (member_ids[li], member_ids[lj])
                    for li, lj in edges
                    if li < len(member_ids) and lj < len(member_ids)
                ]

    def evaluate_frame(self, ego_loc: List[float], active_agents: Dict[int, List[float]],
                      group_graph: Dict, ego_vel: Optional[List[float]] = None,
                      dt: float = 0.2) -> float:
        """
        Evaluate single timestep (called every step during training/evaluation).

        Args:
            ego_loc: [x, y] robot position in meters
            active_agents: {agent_id: [x, y]} agent positions in meters
            group_graph: {'groups': [{'member_ids': [...], 'group_type': '...', ...}]}
            ego_vel: Optional [vx, vy] robot velocity in m/s (unused, kept for API compatibility)
            dt: Time step in seconds (default: 0.2s = 5Hz)

        Returns:
            step_penalty: float (negative reward for RL, higher = worse)
        """
        self.total_frames += 1
        step_penalty = 0.0
        intruded_this_frame = False
        in_ospace = False  # inside any gathering O-space this frame
        in_rspace = False  # inside any cohesive R-space this frame

        for group in group_graph.get('groups', []):
            formation = group.get('group_type', 'cohesive')
            members = group.get('member_ids', [])

            # Get present members
            present_members = [m for m in members if m in active_agents]
            if len(present_members) < 2:
                continue

            # Evaluate based on group type
            if formation == 'queue':
                penalty = self._evaluate_queue(ego_loc, present_members, active_agents, group, dt)
            elif formation == 'gathering':
                # GDS only: O-space intrusion (inner empty center of ring)
                penalty = self._evaluate_group_space(ego_loc, present_members, active_agents,
                                                     group, formation, dt)
                in_ospace = in_ospace or penalty > 0
            elif formation == 'cohesive':
                # CPI only: R-space buffer around cohesive group
                penalty = self._evaluate_cpi(ego_loc, present_members, active_agents, dt)
                in_rspace = in_rspace or penalty > 0
            else:
                penalty = 0.0

            step_penalty += penalty
            if penalty > 0:
                intruded_this_frame = True

        # GDS/CPI take the max over groups (paper Eq. 2 and 4): a frame inside
        # several overlapping O-spaces / R-spaces is counted once.
        if in_ospace:
            self.gds_total_time += dt
        if in_rspace:
            self.cpi_total_time += dt

        # Track intrusion events
        if intruded_this_frame:
            self.intrusion_frames += 1
            if not self.is_currently_intruding:
                self.total_intrusion_events += 1
            self.is_currently_intruding = True
        else:
            self.is_currently_intruding = False

        self.ego_prev = list(ego_loc)
        return step_penalty

    def _evaluate_queue(self, ego_loc: List[float], member_ids: List[int],
                       positions: Dict[int, List[float]], group: Dict, dt: float = 1.0) -> float:
        """
        Queue Cutting Rate (QCR) evaluation.

        QCR metric: sign-based crossing detection — the queue counts as cut once the
        robot crosses any line segment between consecutive queue members; each queue
        is counted at most once per episode.

        RL penalty: proximity-based continuous signal (dense reward shaping).

        Args:
            ego_loc: Robot position
            member_ids: Queue member IDs (ordered)
            positions: Agent positions
            group: Group metadata

        Returns:
            penalty: float (0 if no intrusion, positive otherwise)
        """
        group_id = group.get('group_id', group.get('queue_idx', 0))
        penalty = 0.0
        ego_curr = np.array(ego_loc[:2])

        cached_edges = self._edge_cache.get(group_id)
        if cached_edges:
            edges_to_check = cached_edges
        else:
            edges_to_check = [
                (member_ids[i], member_ids[i + 1])
                for i in range(len(member_ids) - 1)
            ]

        for edge_idx, (a_id, b_id) in enumerate(edges_to_check):
            if a_id not in positions or b_id not in positions:
                continue

            a = np.array(positions[a_id][:2])
            b = np.array(positions[b_id][:2])

            # --- QCR: sign-based crossing detection ---
            if self.ego_prev is not None:
                ego_p = np.array(self.ego_prev[:2])
                d_prev = self._signed_cross(ego_p, a, b)
                d_curr = self._signed_cross(ego_curr, a, b)

                if np.sign(d_prev) * np.sign(d_curr) < 0:
                    # Sign flipped → verify crossing falls strictly between A and B.
                    alpha = d_prev / (d_prev - d_curr)
                    crossing_pt = ego_p + alpha * (ego_curr - ego_p)
                    ab = b - a
                    ab_len_sq = float(np.dot(ab, ab))
                    if ab_len_sq > 1e-6:
                        t = float(np.dot(crossing_pt - a, ab) / ab_len_sq)
                        if 0.0 < t < 1.0:
                            if group_id not in self._qcr_cut_queues:
                                self.qcr_total_cuts += 1
                                self._qcr_cut_queues.add(group_id)

            # --- RL penalty: proximity-based continuous signal ---
            dist = self._point_to_segment_distance(ego_loc, positions[a_id], positions[b_id])
            if dist < self.robot_radius:
                penalty += (self.robot_radius - dist) * 20.0 * dt

        return penalty

    def _evaluate_group_space(self, ego_loc: List[float], member_ids: List[int],
                             positions: Dict[int, List[float]], group: Dict,
                             formation: str, dt: float) -> float:
        """
        Gathering Disruption Score (GDS) evaluation.

        O-space = disk centered at the member centroid with radius equal to the
        nearest member's distance, i.e., the empty core enclosed by the members.
        GDS time is accumulated in evaluate_frame (once per frame).
        """
        member_positions = np.array([positions[mid][:2] for mid in member_ids])
        center = member_positions.mean(axis=0)
        # O-space = inner empty circle: centroid as center, radius = nearest member.
        # Floor at robot_radius to handle degenerate cases (inner_radius ≈ 0).
        inner_radius = float(max(
            np.min(np.linalg.norm(member_positions - center, axis=1)),
            self.robot_radius,
        ))
        inside = np.linalg.norm(np.array(ego_loc[:2]) - center) < inner_radius

        if inside:
            return 10.0 * dt

        return 0.0

    def get_episode_metrics(self) -> Dict[str, float]:
        """
        Get paper-compliant metrics for benchmark evaluation.

        All time-based metrics (GDS, CPI) are reported in raw seconds so that
        intrusion duration is directly interpretable.

        Returns:
            metrics: {
                'QCR': int - number of distinct queues cut [0, num_queues],
                'GDS': float - seconds inside gathering O-space [s],
                'CPI': float - seconds inside R-space buffer around cohesive groups [s],
                'Total_Intrusion_Frames': int,
                'Total_Intrusion_Events': int,
                'Intrusion_Rate': float [0, 1]
            }
        """
        intrusion_rate = self.intrusion_frames / max(1, self.total_frames)

        return {
            'QCR': self.qcr_total_cuts,
            'GDS': self.gds_total_time,
            'CPI': self.cpi_total_time,
            'Total_Intrusion_Frames': self.intrusion_frames,
            'Total_Intrusion_Events': self.total_intrusion_events,
            'Total_Frames': self.total_frames,
            'Intrusion_Rate': intrusion_rate
        }

    # ========== Helper Functions ==========

    def _signed_cross(self, p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """Z-component of (B-A) × (P-A). Positive = left of A→B, negative = right."""
        ab = b - a
        ap = p - a
        return ab[0] * ap[1] - ab[1] * ap[0]

    def _project_t(self, p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """Scalar projection of P onto segment A-B, clamped to [0, 1]."""
        ab = b - a
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq < 1e-6:
            return 0.0
        return float(np.clip(np.dot(p - a, ab) / ab_len_sq, 0.0, 1.0))

    def _point_to_segment_distance(self, p: List[float], a: List[float], b: List[float]) -> float:
        """
        Calculate minimum distance from point to line segment.

        Args:
            p: Point [x, y]
            a: Segment start [x, y]
            b: Segment end [x, y]

        Returns:
            distance: float (meters)
        """
        p = np.array(p)
        a = np.array(a)
        b = np.array(b)

        ab = b - a
        ap = p - a

        # Handle degenerate segment (two people at same location)
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq < 1e-6:
            return np.linalg.norm(ap)

        # Project point onto line segment (clamped to [0, 1])
        t = np.clip(np.dot(ap, ab) / ab_len_sq, 0.0, 1.0)

        # Closest point on segment
        projection = a + t * ab

        return np.linalg.norm(p - projection)

    def _minimum_enclosing_circle(self, pts: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Minimum enclosing circle (Welzl iterative, O(n) expected).
        Returns (center, radius).  Safe for small N (groups are typically 3-10 members).
        """
        pts = np.asarray(pts, dtype=float)
        n = len(pts)
        if n == 0:
            return np.zeros(2), 0.0
        if n == 1:
            return pts[0].copy(), 0.0
        if n == 2:
            c = (pts[0] + pts[1]) / 2
            return c, float(np.linalg.norm(pts[1] - pts[0])) / 2

        c = (pts[0] + pts[1]) / 2
        r = float(np.linalg.norm(pts[1] - pts[0])) / 2

        for i in range(2, n):
            if np.linalg.norm(pts[i] - c) <= r + 1e-9:
                continue
            c, r = pts[i].copy(), 0.0
            for j in range(i):
                if np.linalg.norm(pts[j] - c) <= r + 1e-9:
                    continue
                c = (pts[i] + pts[j]) / 2
                r = float(np.linalg.norm(pts[j] - pts[i])) / 2
                for k in range(j):
                    if np.linalg.norm(pts[k] - c) <= r + 1e-9:
                        continue
                    c, r = self._circle_from_three(pts[i], pts[j], pts[k])
        return c, r

    def _circle_from_three(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, float]:
        """Circumscribed circle through three 2-D points."""
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        cx, cy = float(c[0]), float(c[1])
        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-9:
            # Collinear — fall back to diameter of the farthest pair
            pairs = [(a, b), (a, c), (b, c)]
            dists = [np.linalg.norm(p[1] - p[0]) for p in pairs]
            p0, p1 = pairs[int(np.argmax(dists))]
            return (p0 + p1) / 2, float(np.max(dists)) / 2
        ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
        uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d
        center = np.array([ux, uy])
        return center, float(np.linalg.norm(a - center))

    def _point_in_convex_hull(self, point: List[float], positions: np.ndarray) -> bool:
        """
        Check if point is inside convex hull of positions.

        Args:
            point: [x, y]
            positions: (N, 2) array of positions

        Returns:
            inside: bool
        """
        if len(positions) < 3:
            return False

        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(positions)
            point = np.array(point)
            # Half-plane test: point is inside iff all inequalities satisfied
            return np.all(hull.equations[:, :2] @ point + hull.equations[:, 2] <= 1e-10)
        except Exception:
            # Fallback for degenerate cases (collinear points etc.)
            min_xy = np.min(positions, axis=0)
            max_xy = np.max(positions, axis=0)
            return (min_xy[0] <= point[0] <= max_xy[0] and
                    min_xy[1] <= point[1] <= max_xy[1])

    def _evaluate_cpi(self, ego_loc: List[float], member_ids: List[int],
                     positions: Dict[int, List[float]], dt: float) -> float:
        """CPI: R-space test (outside hull but within cpi_buffer of it).
        CPI time is accumulated in evaluate_frame (once per frame)."""
        member_positions = np.array([positions[mid] for mid in member_ids])
        in_buffer = self._point_near_convex_hull(ego_loc, member_positions, self.cpi_buffer)
        if in_buffer:
            return 10.0 * dt
        return 0.0

    def _point_near_convex_hull(self, point: List[float], positions: np.ndarray,
                                buffer_dist: float) -> bool:
        """Check if point is in R-space: outside the convex hull but within buffer_dist of it."""
        if len(positions) < 3:
            if len(positions) == 2:
                dist = self._point_to_segment_distance(point, positions[0], positions[1])
                return dist < buffer_dist
            return False
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(positions)
            point_arr = np.array(point)
            signed_dists = hull.equations[:, :2] @ point_arr + hull.equations[:, 2]
            if np.all(signed_dists <= 0):
                return False  # inside P-space (hull interior) — not R-space
            hull_pts = positions[hull.vertices]
            n = len(hull_pts)
            min_dist = min(
                self._point_to_segment_distance(point, hull_pts[i], hull_pts[(i + 1) % n])
                for i in range(n)
            )
            return min_dist < buffer_dist
        except Exception:
            min_xy = np.min(positions, axis=0)
            max_xy = np.max(positions, axis=0)
            p = np.array(point)
            return (min_xy[0] - buffer_dist <= p[0] <= max_xy[0] + buffer_dist and
                    min_xy[1] - buffer_dist <= p[1] <= max_xy[1] + buffer_dist)
