"""
expert_trajectory.py — Expert agent trajectory generator (crowd-prior, least-violating).

Strategy:
  1. Find the earliest sim frame where a cohesive group appears; select the
     non-group agent that spawns closest to that time (active "right after").
  2. Among those candidates, pick the one with most queue/gathering violations.
  3. Correct violations per group type:
       - queue    : replace violated segment with speed-correct bypass going
                    BEHIND the tail; insert extra frames so no freezing.
       - other    : per-member repulsion push (capped per frame).
  4. Apply unicycle kinematic constraints for smooth, physically plausible motion.
  5. Inject into the scenario DataFrame with group_type='expert'.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple



# ── Constants ─────────────────────────────────────────────────────────────────
MEMBER_CLEARANCE_PX     = 18.0   # violation / push radius for non-queue groups (~0.9 m)
QUEUE_CLEARANCE_PX      = 40.0   # queue violation radius — wider than half-spacing (60px/2=30)
                                  # detects expert walking between members or near the line
QUEUE_BYPASS_MARGIN     = 60.0   # extra px beyond tail for bypass waypoint (~6m) - increased from 40
MAX_PUSH_PX             = 8.0    # cap per-frame push magnitude (prevents freezing)
SPAWN_WINDOW_SIM        = 40     # sim frames after cohesive appearance to search
MIN_EXPERT_DISPLACEMENT_PX = 150.0  # start-goal distance of an expert path inside the scene (~3 m), as in agent selection

# O-space and R-space parameters (for gathering and cohesive)
GATHERING_BYPASS_MARGIN_METERS = 0.5   # default bypass clearance (meters); overridden by scenarios.yaml
COHESIVE_CLEARANCE_PX   = 24.0   # R-space clearance for repulsion push (~2.4 m at 10px/m)
EXIT_LOOKAHEAD_FRAMES   = 10     # frames to look ahead for smoother exit reconnection


# ── Cohesive first appearance ─────────────────────────────────────────────────

def _cohesive_first_frame(framework, group_types=('cohesive',)) -> Optional[int]:
    """Earliest sim frame where any member of a group of the given types is active."""
    earliest = None
    for group in framework.group_graph.get('groups', []):
        if group.get('group_type') not in group_types:
            continue
        for agent_id in group.get('member_ids', []):
            traj = framework.agent_trajectory.get(agent_id)
            if traj is None or len(traj) == 0:
                continue
            fg = framework.agent_parameter[agent_id].get('frame_global', 0)
            if earliest is None or fg < earliest:
                earliest = fg
    return earliest


# ── Group frame maps ──────────────────────────────────────────────────────────

def _build_all_groups_frame_map(
    framework,
    include_types: tuple = ('queue', 'cohesive', 'gathering', 'crosswalk'),
) -> Dict[int, List[np.ndarray]]:
    """
    {sim_frame: [member_pos_px, ...]} for ALL group members.
    """
    frame_map: Dict[int, List[np.ndarray]] = {}
    for group in framework.group_graph.get('groups', []):
        if group.get('group_type', '') not in include_types:
            continue
        for agent_id in group.get('member_ids', []):
            traj = framework.agent_trajectory.get(agent_id)
            if traj is None or len(traj) == 0:
                continue
            fg = framework.agent_parameter[agent_id].get('frame_global', 0)
            for t, pos in enumerate(traj):
                frame_map.setdefault(fg + t, []).append(np.asarray(pos, dtype=float))
    return frame_map



def _build_queue_frame_maps(framework) -> List[Dict]:
    """
    Per-queue list of:
      { 'member_ids': [head_id, ..., tail_id],
        'pos_map':    {sim_frame: {agent_id: np.ndarray}} }
    """
    result = []
    for group in framework.group_graph.get('groups', []):
        if group.get('group_type') != 'queue':
            continue
        member_ids = group.get('member_ids', [])
        if len(member_ids) < 2:
            continue
        pos_map: Dict[int, Dict] = {}
        for agent_id in member_ids:
            traj = framework.agent_trajectory.get(agent_id)
            if traj is None or len(traj) == 0:
                continue
            fg = framework.agent_parameter[agent_id].get('frame_global', 0)
            for t, pos in enumerate(traj):
                pos_map.setdefault(fg + t, {})[agent_id] = np.array(pos, dtype=float)
        result.append({'member_ids': member_ids, 'pos_map': pos_map})
    return result


# ── Agent selection ───────────────────────────────────────────────────────────

def _find_top_pedestrian_encountering_agents(
    framework,
    top_n: int = 3,
    clearance_px: float = MEMBER_CLEARANCE_PX,
    min_displacement_px: float = 150.0,
    spawn_window: int = 200,
) -> List[int]:
    """
    Standard-FT agent selection: pick agents with the largest start→goal
    displacement among early-spawning pedestrians.

    Sorting by displacement (not path_length) avoids selecting wandering agents
    whose spawn and goal are close despite a long cumulative path.

    Args:
        framework:           CrowdES framework with agent_trajectory / agent_parameter.
        top_n:               Number of agents to return.
        clearance_px:        Unused (kept for API compatibility).
        min_displacement_px: Ignore agents below this start→goal distance.
        spawn_window:        Only consider agents whose frame_global <= spawn_window.
    """
    candidates = []
    for agent_id, traj in framework.agent_trajectory.items():
        if len(traj) < 40:
            continue
        fg = framework.agent_parameter[agent_id].get('frame_global', 0)
        if fg > spawn_window:
            continue
        traj_arr = np.array(traj, dtype=float)
        displacement = float(np.linalg.norm(traj_arr[-1] - traj_arr[0]))
        if displacement < min_displacement_px:
            continue

        candidates.append((-displacement, fg, agent_id))

    if not candidates:
        return []

    candidates.sort()
    selected = candidates[:top_n]
    print(f'[ExpertAdapter] Standard-FT candidates: {len(candidates)}, '
          f'selecting top {len(selected)}:')
    for rank, c in enumerate(selected):
        print(f'  [{rank+1}] agent_id={c[2]}, frame_global={c[1]}, '
              f'displacement={-c[0]:.1f}px')
    return [c[2] for c in selected]


def _find_top_qg_violating_agents(
    framework,
    qg_frame_map: Dict[int, List[np.ndarray]],
    cohesive_start: int,
    top_n: int = 3,
    clearance_px: float = MEMBER_CLEARANCE_PX,
    min_displacement_px: float = 150.0,
) -> List[int]:
    """
    Return up to top_n non-group agent_ids ranked by qg_violations × displacement
    (combined score), preferring agents whose frame_global is in
    [cohesive_start, cohesive_start + SPAWN_WINDOW_SIM].

    Group members are never selected: the expert replaces the agent, so the
    group graph would list an agent that is no longer in the group. Agents
    outside the spawn window only fill the pool when the window has fewer
    than top_n candidates.

    Scoring: score = qg_violations * displacement
    Tie-breaking: inside the window first → score DESC → frame_global ASC.
    Agents with zero violations are kept if no violating candidates exist.

    Args:
        min_displacement_px: Minimum start-to-goal displacement. Agents with
            shorter paths are excluded to avoid degenerate expert trajectories.
            At 50 px/m this is ~3 m.
    """
    spawn_lo = cohesive_start
    spawn_hi = cohesive_start + SPAWN_WINDOW_SIM
    group_member_ids = {agent_id for group in framework.group_graph.get('groups', [])
                        for agent_id in group.get('member_ids', [])}

    candidates = []

    for agent_id, traj in framework.agent_trajectory.items():
        if len(traj) < 2 or agent_id in group_member_ids:
            continue

        fg = framework.agent_parameter[agent_id].get('frame_global', 0)
        outside_window = not (spawn_lo <= fg <= spawn_hi)

        traj_arr     = np.array(traj, dtype=float)
        displacement = float(np.linalg.norm(traj_arr[-1] - traj_arr[0]))
        if displacement < min_displacement_px:
            continue

        # Count queue/gathering violations
        qg_violations = 0
        clr2 = clearance_px * clearance_px
        for t in range(len(traj_arr)):
            members = qg_frame_map.get(fg + t)
            if not members:
                continue
            diffs = traj_arr[t] - np.array(members, dtype=float)
            if np.einsum('ij,ij->i', diffs, diffs).min() < clr2:
                qg_violations += 1

        score = qg_violations * displacement
        candidates.append((outside_window, -score, fg, agent_id))

    if not candidates:
        return []

    candidates.sort()
    selected = candidates[:top_n]
    print(f'[ExpertAdapter] Candidates: {len(candidates)}, '
          f'selecting top {len(selected)}:')
    for rank, (outside_window, neg_score, fg, agent_id) in enumerate(selected):
        traj_arr = np.array(framework.agent_trajectory[agent_id], dtype=float)
        disp = float(np.linalg.norm(traj_arr[-1] - traj_arr[0]))
        score = -neg_score
        viol = int(score / disp) if disp > 0 else 0
        print(f'  [{rank+1}] agent_id={agent_id}, frame_global={fg}, '
              f'qg_violations={viol}, displacement={disp:.1f}px, score={score:.1f}'
              + (' (outside the spawn window)' if outside_window else ''))
    return [c[3] for c in selected]


# ── Violation segments ────────────────────────────────────────────────────────

def _find_violation_segments(
    traj_px: np.ndarray,
    frame_start: int,
    frame_map: Dict[int, List[np.ndarray]],
    clearance_px: float,
) -> List[Tuple[int, int]]:
    """Return list of (start_idx, end_idx) for contiguous violated frames."""
    in_vio = np.zeros(len(traj_px), dtype=bool)
    clr2 = clearance_px * clearance_px
    for t in range(len(traj_px)):
        members = frame_map.get(frame_start + t)
        if not members:
            continue
        diffs = traj_px[t] - np.array(members, dtype=float)
        if np.einsum('ij,ij->i', diffs, diffs).min() < clr2:
            in_vio[t] = True

    segments, start = [], None
    for t in range(len(in_vio)):
        if in_vio[t] and start is None:
            start = t
        elif not in_vio[t] and start is not None:
            segments.append((start, t - 1))
            start = None
    if start is not None:
        segments.append((start, len(in_vio) - 1))
    return segments


# ── Queue bypass ──────────────────────────────────────────────────────────────

def _pt_to_seg_dist(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Minimum distance from point `pt` to line segment [a, b]."""
    ab   = b - a
    ab2  = float(np.dot(ab, ab))
    if ab2 < 1e-6:
        return float(np.linalg.norm(pt - a))
    t    = float(np.dot(pt - a, ab)) / ab2
    t    = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(pt - proj))


def _find_queue_violation_segments(
    traj_px: np.ndarray,
    frame_start: int,
    q: Dict,
    clearance_px: float = QUEUE_CLEARANCE_PX,
) -> List[Tuple[int, int]]:
    """
    Queue-specific violation detection using LINE SEGMENT distance.

    For each expert trajectory frame, check distance to:
      1. Each individual queue member (point distance)
      2. Each consecutive pair of queue members (segment distance)

    This catches the case where the expert passes *between* two adjacent
    members — each individual distance might be > clearance_px, but the
    perpendicular distance to the line is not.
    """
    member_ids = q['member_ids']
    pos_map    = q['pos_map']

    in_vio = np.zeros(len(traj_px), dtype=bool)
    for t in range(len(traj_px)):
        sim_frame = frame_start + t
        id_pos = pos_map.get(sim_frame)
        if not id_pos:
            # Queue is stationary — fall back to a nearby frame
            for delta in range(-2, 3):
                id_pos = pos_map.get(sim_frame + delta)
                if id_pos:
                    break
            if not id_pos:
                continue

        # Ordered positions: head → tail
        ordered = [id_pos[aid] for aid in member_ids if aid in id_pos]
        if not ordered:
            continue

        pt = traj_px[t]
        clr2 = clearance_px * clearance_px
        arr = np.array(ordered, dtype=float)

        # 1. Point distances to each member (vectorized)
        diffs = pt - arr
        pt_dists2 = np.einsum('ij,ij->i', diffs, diffs)
        if pt_dists2.min() < clr2:
            in_vio[t] = True

        # 2. Segment distances between consecutive members (vectorized)
        if not in_vio[t] and len(arr) > 1:
            pts_a = arr[:-1]
            pts_b = arr[1:]
            ab = pts_b - pts_a
            ab2 = np.einsum('ij,ij->i', ab, ab)
            ap = pt - pts_a
            t_clip = np.clip(
                np.einsum('ij,ij->i', ap, ab) / np.where(ab2 < 1e-6, 1.0, ab2),
                0.0, 1.0
            )
            proj = pts_a + t_clip[:, np.newaxis] * ab
            seg_diffs = pt - proj
            if np.einsum('ij,ij->i', seg_diffs, seg_diffs).min() < clr2:
                in_vio[t] = True

    segments, start = [], None
    for t in range(len(in_vio)):
        if in_vio[t] and start is None:
            start = t
        elif not in_vio[t] and start is not None:
            segments.append((start, t - 1))
            start = None
    if start is not None:
        segments.append((start, len(in_vio) - 1))
    return segments


def _correct_queue_violations(
    traj_px: np.ndarray,
    frame_start: int,
    queue_frame_maps: List[Dict],
    framework,
    clearance_px: float = QUEUE_CLEARANCE_PX,
    bypass_margin: float = QUEUE_BYPASS_MARGIN,
) -> np.ndarray:
    """
    For each violated queue segment, replace with a bypass that goes behind
    the tail.  Frame count is computed from actual path distance at walking
    speed so the agent never freezes.

        entry_pos ──► bypass_waypoint ──► exit_pos

    bypass_waypoint = tail_pos + queue_dir * (clearance + bypass_margin)

    Violation detection uses LINE SEGMENT distance (not just point distance)
    so passing between adjacent members is also caught.
    """
    # Get pixel_per_meter from metadata
    metadata = getattr(framework, 'scenario_metadata', None) or {}
    queues   = metadata.get('queue', [])
    ppm = None
    if queues and 'pixel_per_meter' in queues[0]:
        ppm = float(queues[0]['pixel_per_meter'])
    if ppm is None:
        # Fallback: try to infer from ortho_size (metadata usually has this)
        # ortho_size / 2048 = meters_per_pixel → ppm = 2048 / ortho_size
        ppm = 10.0  # Conservative default for CARLA Town10

    speed_px = 1.2 * ppm / framework.dataset_fps   # ~12 px/step at 20 FPS
    print(f'[ExpertAdapter] Queue: pixel_per_meter={ppm:.1f}, speed_px={speed_px:.2f}')

    corrected = traj_px.copy().astype(float)

    for q in queue_frame_maps:
        member_ids = q['member_ids']
        pos_map    = q['pos_map']
        head_id    = member_ids[0]
        tail_id    = member_ids[-1]

        segments = _find_queue_violation_segments(corrected, frame_start, q, clearance_px)
        if not segments:
            continue

        print(f'[ExpertAdapter] Queue violation: {len(segments)} segment(s) detected')

        # Flat frame_map for fallback push
        q_frame_map: Dict[int, List[np.ndarray]] = {
            f: list(id_pos.values()) for f, id_pos in pos_map.items()
        }

        offset = 0   # accounts for inserted extra frames
        for seg_s_orig, seg_e_orig in segments:
            seg_s = seg_s_orig + offset
            seg_e = seg_e_orig + offset

            mid_frame  = frame_start + (seg_s_orig + seg_e_orig) // 2
            frame_data = pos_map.get(mid_frame, {})
            tail_pos   = frame_data.get(tail_id)
            head_pos   = frame_data.get(head_id)

            # Fall back to nearest available frame if mid_frame not in pos_map
            if tail_pos is None or head_pos is None:
                nearby_frames = sorted(pos_map.keys(), key=lambda f: abs(f - mid_frame))
                for nf in nearby_frames[:3]:
                    fd = pos_map[nf]
                    if tail_id in fd and head_id in fd:
                        tail_pos = fd[tail_id]
                        head_pos = fd[head_id]
                        break

            # If still unavailable, push instead
            if tail_pos is None or head_pos is None:
                for t in range(seg_s, seg_e + 1):
                    members = q_frame_map.get(frame_start + (t - offset), [])
                    pos = corrected[t]
                    net = np.zeros(2)
                    for m_pos in members:
                        vec = pos - m_pos
                        d   = float(np.linalg.norm(vec))
                        if d < clearance_px and d > 1e-6:
                            net += (vec / d) * (clearance_px - d)
                        elif d < 1e-6:
                            net += np.array([clearance_px, 0.0])
                    net_mag = np.linalg.norm(net)
                    if net_mag > 1e-6:
                        corrected[t] = pos + net * min(1.0, MAX_PUSH_PX / net_mag)
                continue

            q_vec = tail_pos - head_pos
            q_len = float(np.linalg.norm(q_vec))
            if q_len < 1e-6:
                continue
            q_dir  = q_vec / q_len

            # Generate bypass waypoint
            bypass_candidate = tail_pos + q_dir * (clearance_px + bypass_margin)

            # Check navmesh validity
            navmesh = getattr(framework, 'navmesh', None)
            walkable = getattr(framework, 'walkable', None)
            def is_walkable(pos):
                # Use walkable map if available, fallback to navmesh
                check_map = walkable if walkable is not None else navmesh
                if check_map is None or isinstance(check_map, dict):
                    return True
                x, y = int(round(pos[0])), int(round(pos[1]))
                h, w = check_map.shape
                if 0 <= y < h and 0 <= x < w:
                    return check_map[y, x] > 0
                return False

            # If bypass is not walkable, try perpendicular offsets
            if not is_walkable(bypass_candidate):
                perp_dir = np.array([-q_dir[1], q_dir[0]])
                for perp_off in [20.0, 40.0, -20.0, -40.0]:
                    alt_bypass = bypass_candidate + perp_dir * perp_off
                    if is_walkable(alt_bypass):
                        bypass = alt_bypass
                        break
                else:
                    # Fallback: use original (even if not walkable)
                    bypass = bypass_candidate
            else:
                bypass = bypass_candidate

            # Use clean frames just OUTSIDE the violation as anchors
            # Look ahead EXIT_LOOKAHEAD_FRAMES for smoother reconnection
            entry_pos = corrected[seg_s - 1].copy() if seg_s > 0 else corrected[seg_s].copy()
            exit_idx = min(seg_e + 1 + EXIT_LOOKAHEAD_FRAMES, len(corrected) - 1)
            exit_pos = corrected[exit_idx].copy()

            # Frame counts based on actual distance at walking speed
            d1 = float(np.linalg.norm(bypass - entry_pos))
            d2 = float(np.linalg.norm(exit_pos - bypass))
            n1 = max(2, int(d1 / speed_px))
            n2 = max(2, int(d2 / speed_px))

            t1   = np.linspace(0., 1., n1, endpoint=False)
            t2   = np.linspace(0., 1., n2, endpoint=True)
            seg1 = np.outer(1. - t1, entry_pos) + np.outer(t1, bypass)
            seg2 = np.outer(1. - t2, bypass)    + np.outer(t2, exit_pos)
            bypass_frames = np.vstack([seg1, seg2])

            # Replace from seg_s to exit_idx (instead of just seg_e+1)
            cut_start = seg_s
            cut_end   = min(exit_idx, len(corrected) - 1)
            orig_n    = cut_end - cut_start + 1
            corrected = np.vstack([
                corrected[:cut_start],
                bypass_frames,
                corrected[cut_end + 1:],
            ])
            offset += len(bypass_frames) - orig_n

    return corrected


# ── Gathering O-space bypass ──────────────────────────────────────────────────

def _correct_gathering_violations(
    traj_px: np.ndarray,
    frame_start: int,
    framework,
    bypass_margin_meters: float = GATHERING_BYPASS_MARGIN_METERS,
) -> np.ndarray:
    """
    Gathering O-space bypass: route expert around the outer radius.

    For each gathering group:
      1. Compute center and outer_radius (mean + std of member distances)
      2. Find frames where expert intrudes O-space (distance < outer_radius)
      3. Generate bypass waypoint tangent to the circle
      4. Speed-correct interpolation
    """
    from scipy.spatial import ConvexHull

    corrected = traj_px.copy().astype(float)

    # Get pixel_per_meter from metadata
    metadata = getattr(framework, 'scenario_metadata', None) or {}
    gatherings = metadata.get('gathering', [])
    ppm = None
    if gatherings and 'pixel_per_meter' in gatherings[0]:
        ppm = float(gatherings[0]['pixel_per_meter'])
    if ppm is None:
        ppm = 10.0  # Conservative default

    speed_px = 1.2 * ppm / framework.dataset_fps
    gathering_radius_margin = bypass_margin_meters * ppm
    print(f'[ExpertAdapter] Gathering: pixel_per_meter={ppm:.1f}, bypass_margin={bypass_margin_meters}m ({gathering_radius_margin:.1f}px)')

    for group in framework.group_graph.get('groups', []):
        if group.get('group_type') != 'gathering':
            continue

        member_ids = group.get('member_ids', [])
        if len(member_ids) < 2:
            continue

        # Build frame map for this gathering
        member_positions_by_frame = {}
        for aid in member_ids:
            traj = framework.agent_trajectory.get(aid)
            if traj is None or len(traj) == 0:
                continue
            fg = framework.agent_parameter[aid].get('frame_global', 0)
            for t, pos in enumerate(traj):
                frame = fg + t
                member_positions_by_frame.setdefault(frame, []).append(np.array(pos, dtype=float))

        # Detect violation segments
        in_vio = np.zeros(len(corrected), dtype=bool)
        centers_by_frame = {}
        radii_by_frame = {}

        for t in range(len(corrected)):
            sim_frame = frame_start + t
            members = member_positions_by_frame.get(sim_frame)
            if not members or len(members) < 2:
                continue

            # Compute gathering center and outer radius
            members_arr = np.array(members)
            center = members_arr.mean(axis=0)
            dists = np.linalg.norm(members_arr - center, axis=1)
            outer_radius = float(dists.mean() + dists.std()) if len(dists) > 1 else float(dists[0]) + 20.0

            centers_by_frame[t] = center
            radii_by_frame[t] = outer_radius

            # Check intrusion using bypass_radius (not outer_radius)
            # This ensures violation detection matches the bypass arc
            bypass_radius = outer_radius + gathering_radius_margin
            expert_pos = corrected[t]
            dist_to_center = float(np.linalg.norm(expert_pos - center))
            if dist_to_center < bypass_radius:
                in_vio[t] = True

        # Find continuous violation segments
        segments, start = [], None
        for t in range(len(in_vio)):
            if in_vio[t] and start is None:
                start = t
            elif not in_vio[t] and start is not None:
                segments.append((start, t - 1))
                start = None
        if start is not None:
            segments.append((start, len(in_vio) - 1))

        if not segments:
            continue

        print(f'[ExpertAdapter] Gathering O-space violation: {len(segments)} segment(s) detected')

        # Correct each segment with bypass
        offset = 0
        for seg_s_orig, seg_e_orig in segments:
            seg_s = seg_s_orig + offset
            seg_e = seg_e_orig + offset

            # Get center and radius at mid-frame
            mid_t = (seg_s_orig + seg_e_orig) // 2
            if mid_t not in centers_by_frame:
                continue

            center = centers_by_frame[mid_t]
            outer_radius = radii_by_frame[mid_t]

            # Use clean frames outside violation as anchors
            # Look ahead EXIT_LOOKAHEAD_FRAMES for smoother reconnection
            entry_pos = corrected[max(0, seg_s - 1)]
            exit_idx = min(seg_e + 1 + EXIT_LOOKAHEAD_FRAMES, len(corrected) - 1)
            exit_pos = corrected[exit_idx]

            # Generate arc bypass around the gathering circle
            # Use entry→center and exit→center to find tangent points on the circle
            entry_to_center = center - entry_pos
            exit_to_center = center - exit_pos

            # Radius for bypass arc (outer_radius + margin)
            bypass_radius = outer_radius + gathering_radius_margin

            # Normalize directions
            entry_dir = entry_to_center / (np.linalg.norm(entry_to_center) + 1e-6)
            exit_dir = exit_to_center / (np.linalg.norm(exit_to_center) + 1e-6)

            # Compute tangent points on the bypass circle
            # Tangent from entry toward center, then perpendicular
            entry_tangent = center - entry_dir * bypass_radius
            exit_tangent = center - exit_dir * bypass_radius

            # Determine which side to go (left or right of entry→exit line)
            # Cross product to determine orientation
            cross = entry_to_center[0] * exit_to_center[1] - entry_to_center[1] * exit_to_center[0]

            # Generate two arc candidates (clockwise and counter-clockwise)
            if abs(cross) < 1e-6:
                # Entry and exit are collinear with center, use perpendicular
                perp_dir = np.array([-entry_dir[1], entry_dir[0]])
                bypass_candidate1 = center + perp_dir * bypass_radius
                bypass_candidate2 = center - perp_dir * bypass_radius
            else:
                # Use tangent points
                bypass_candidate1 = entry_tangent
                bypass_candidate2 = exit_tangent

            # Check navmesh validity for both candidates (with erosion margin)
            navmesh  = getattr(framework, 'navmesh', None)
            walkable = getattr(framework, 'walkable', None)
            _SAFE_MARGIN = 4  # pixels — keeps arc points away from sidewalk edge
            def is_walkable(pos, margin=_SAFE_MARGIN):
                check_map = walkable if walkable is not None else navmesh
                if check_map is None or isinstance(check_map, dict):
                    return True
                x, y = int(round(pos[0])), int(round(pos[1]))
                h, w = check_map.shape
                x0, x1 = max(0, x - margin), min(w, x + margin + 1)
                y0, y1 = max(0, y - margin), min(h, y + margin + 1)
                return bool(check_map[y0:y1, x0:x1].all())

            # Pick the valid candidate closer to midpoint
            mid_point = (entry_pos + exit_pos) / 2.0
            candidates = []
            if is_walkable(bypass_candidate1):
                candidates.append((bypass_candidate1, np.linalg.norm(bypass_candidate1 - mid_point)))
            if is_walkable(bypass_candidate2):
                candidates.append((bypass_candidate2, np.linalg.norm(bypass_candidate2 - mid_point)))

            if not candidates:
                # Fallback: use closer candidate (even if not walkable)
                if np.linalg.norm(bypass_candidate1 - mid_point) < np.linalg.norm(bypass_candidate2 - mid_point):
                    bypass_waypoint = bypass_candidate1
                else:
                    bypass_waypoint = bypass_candidate2
            else:
                # Pick the walkable one closer to mid_point
                bypass_waypoint = min(candidates, key=lambda c: c[1])[0]

            # Generate arc path around the gathering
            entry_angle = np.arctan2(entry_pos[1] - center[1], entry_pos[0] - center[0])
            exit_angle  = np.arctan2(exit_pos[1]  - center[1], exit_pos[0]  - center[0])

            # Determine arc direction via bypass_waypoint (walkability-validated side)
            to_exit_ccw = (exit_angle - entry_angle) % (2 * np.pi)
            if to_exit_ccw <= np.pi:
                angle_diff = to_exit_ccw              # CCW is shorter
            else:
                angle_diff = -(2 * np.pi - to_exit_ccw)  # CW is shorter

            # Number of waypoints along the arc (more for smoother path)
            num_arc_points = max(5, int(abs(angle_diff) / (np.pi / 8)))  # At least 5 points

            # Generate arc waypoints; snap inward-creeping points onto safe walkable region
            arc_waypoints = []
            for i in range(num_arc_points):
                t = i / float(num_arc_points - 1) if num_arc_points > 1 else 0.0
                angle = entry_angle + t * angle_diff
                arc_point = center + bypass_radius * np.array([np.cos(angle), np.sin(angle)])

                # If the nominal arc point is too close to the walkable boundary,
                # nudge it outward (increase radius) until it's safely inside.
                if not is_walkable(arc_point):
                    snapped = arc_point
                    for extra in range(1, 30):
                        candidate = center + (bypass_radius + extra) * np.array([np.cos(angle), np.sin(angle)])
                        if is_walkable(candidate):
                            snapped = candidate
                            break
                    arc_point = snapped

                arc_waypoints.append(arc_point)

            # Compute total arc length
            total_dist = 0.0
            for i in range(len(arc_waypoints) - 1):
                total_dist += np.linalg.norm(arc_waypoints[i + 1] - arc_waypoints[i])

            # Compute required frames for smooth motion
            n_frames = max(len(arc_waypoints), int(np.ceil(total_dist / speed_px)))

            # Interpolate along arc waypoints
            new_traj = []
            for i in range(n_frames):
                alpha = i / float(n_frames - 1) if n_frames > 1 else 0.0
                # Find which arc segment we're in
                segment_idx = int(alpha * (len(arc_waypoints) - 1))
                segment_idx = min(segment_idx, len(arc_waypoints) - 2)

                # Local interpolation within segment
                t_local = (alpha * (len(arc_waypoints) - 1)) - segment_idx
                pos = arc_waypoints[segment_idx] * (1 - t_local) + arc_waypoints[segment_idx + 1] * t_local
                new_traj.append(pos)

            # Replace segment from seg_s to exit_idx
            corrected = np.concatenate([
                corrected[:seg_s],
                np.array(new_traj),
                corrected[exit_idx + 1:]
            ])

            offset += len(new_traj) - (exit_idx - seg_s + 1)

    return corrected


# ── Cohesive R-space bypass ───────────────────────────────────────────────────

def _correct_cohesive_violations(
    traj_px: np.ndarray,
    frame_start: int,
    framework,
    df: 'pd.DataFrame',
) -> np.ndarray:
    """
    Cohesive R-space correction: frame-by-frame repulsion push.

    Since cohesive groups move dynamically, use per-frame repulsion instead of
    static bypass. For each frame where expert intrudes buffered hull:
      1. Compute buffered convex hull at that frame
      2. Push expert away from hull centroid
      3. Cap push magnitude to MAX_PUSH_PX per frame
      4. Verify walkable after push
    """
    from scipy.spatial import ConvexHull
    from shapely.geometry import Point, Polygon

    corrected = traj_px.copy().astype(float)
    n_corrected = 0

    # Walkable check function
    walkable = getattr(framework, 'walkable', None)
    navmesh = getattr(framework, 'navmesh', None)
    def is_walkable(pos):
        check_map = walkable if walkable is not None else navmesh
        if check_map is None or isinstance(check_map, dict):
            return True
        x, y = int(round(pos[0])), int(round(pos[1]))
        h, w = check_map.shape
        if 0 <= y < h and 0 <= x < w:
            return check_map[y, x] > 0
        return False

    for group in framework.group_graph.get('groups', []):
        if group.get('group_type') != 'cohesive':
            continue

        member_ids = group.get('member_ids', [])
        if len(member_ids) < 3:
            continue  # Need at least 3 points for convex hull

        # Build frame map from DataFrame (20 FPS) instead of simulator trajectory
        member_positions_by_frame = {}
        for aid in member_ids:
            mask = df['agent_id'] == aid
            if not mask.any():
                continue
            sub = df.loc[mask].sort_values('frame')
            for frame, x, y in zip(sub['frame'].to_numpy(dtype=int),
                                    sub['x'].to_numpy(dtype=float),
                                    sub['y'].to_numpy(dtype=float)):
                member_positions_by_frame.setdefault(frame, []).append(np.array([x, y]))

        # Frame-by-frame repulsion push
        for t in range(len(corrected)):
            frame_id = frame_start + t
            members = member_positions_by_frame.get(frame_id)
            if not members or len(members) < 3:
                continue

            try:
                # Compute convex hull
                members_arr = np.array(members)
                hull = ConvexHull(members_arr)
                hull_points = members_arr[hull.vertices]

                # Create buffered polygon using shapely
                poly = Polygon(hull_points)
                buffered_poly = poly.buffer(COHESIVE_CLEARANCE_PX)

                # Check intrusion
                expert_pos = corrected[t]
                expert_point = Point(expert_pos)

                if buffered_poly.contains(expert_point):
                    # Push away from centroid
                    centroid = np.array([buffered_poly.centroid.x, buffered_poly.centroid.y])
                    vec = expert_pos - centroid
                    dist = float(np.linalg.norm(vec))

                    if dist > 1e-6:
                        # Desired push: move to edge of buffered zone
                        boundary_coords = np.array(buffered_poly.exterior.coords)
                        max_radius = np.max([np.linalg.norm(pt - centroid) for pt in boundary_coords])
                        target_dist = max_radius + COHESIVE_CLEARANCE_PX
                        push_vec = (vec / dist) * (target_dist - dist)
                    else:
                        # Expert exactly on centroid, push in arbitrary direction
                        push_vec = np.array([COHESIVE_CLEARANCE_PX, 0.0])

                    # Cap push magnitude
                    push_mag = np.linalg.norm(push_vec)
                    if push_mag > MAX_PUSH_PX:
                        push_vec = (push_vec / push_mag) * MAX_PUSH_PX

                    new_pos = expert_pos + push_vec

                    # Verify walkable
                    if is_walkable(new_pos):
                        corrected[t] = new_pos
                        n_corrected += 1
                    else:
                        # Try perpendicular pushes
                        perp_vec = np.array([-push_vec[1], push_vec[0]])
                        for sign in [1.0, -1.0]:
                            alt_pos = expert_pos + perp_vec * sign * np.linalg.norm(push_vec)
                            if is_walkable(alt_pos):
                                corrected[t] = alt_pos
                                n_corrected += 1
                                break

            except Exception:
                continue

    if n_corrected > 0:
        print(f'[ExpertAdapter] Cohesive R-space: pushed {n_corrected} frames')

    return corrected


# ── Standard-FT: pedestrian density cost map ─────────────────────────────────

def _build_pedestrian_density_cost_map(
    navmesh: np.ndarray,
    framework,
    sigma_px: float = 20.0,
    cost_weight: float = 30.0,
) -> np.ndarray:
    """
    Phase 1 — A* planning: time-averaged pedestrian density → additive cost map.
    Pedestrian-heavy areas get higher cost so A* naturally routes around crowds.
    """
    import cv2 as _cv2
    h, w = navmesh.shape
    density = np.zeros((h, w), dtype=np.float32)

    for traj in framework.agent_trajectory.values():
        traj_np = np.array(traj, dtype=float)
        xs = np.round(traj_np[:, 0]).astype(int)
        ys = np.round(traj_np[:, 1]).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        np.add.at(density, (ys[valid], xs[valid]), 1.0)

    if density.max() > 0:
        density /= density.max()
    k = int(sigma_px * 2) | 1  # odd kernel
    density = _cv2.GaussianBlur(density, (k, k), sigma_px)
    return density * cost_weight


# ── Standard-FT: SFM individual repulsion ─────────────────────────────────────

def _correct_individual_violations(
    traj_px: np.ndarray,
    frame_start: int,
    framework,
    ped_frame_map: Dict[int, np.ndarray],
    clearance_px: float = MEMBER_CLEARANCE_PX,
) -> np.ndarray:
    """
    Phase 2 — post-A* SFM repulsion from individual pedestrians.

    For each trajectory frame, accumulates repulsion vectors from all
    non-expert agents within clearance_px (SFM repulsive force component).
    Expert-only: pedestrian trajectories are not modified.
    """
    corrected = traj_px.copy().astype(float)

    walkable = getattr(framework, 'walkable', None)
    navmesh  = getattr(framework, 'navmesh', None)
    def is_walkable(pos):
        check_map = walkable if walkable is not None else navmesh
        if check_map is None or isinstance(check_map, dict):
            return True
        x, y = int(round(pos[0])), int(round(pos[1]))
        hm, wm = check_map.shape
        if 0 <= y < hm and 0 <= x < wm:
            return check_map[y, x] > 0
        return False

    n_corrected = 0
    for t in range(len(corrected)):
        peds = ped_frame_map.get(frame_start + t)
        if peds is None or len(peds) == 0:
            continue

        expert_pos = corrected[t]
        peds_arr = np.array(peds, dtype=float)
        deltas = expert_pos - peds_arr
        dists = np.linalg.norm(deltas, axis=1)
        mask_rep = (dists > 1e-6) & (dists < clearance_px)
        if not mask_rep.any():
            continue
        total_push = ((deltas[mask_rep] / dists[mask_rep, np.newaxis])
                      * (clearance_px - dists[mask_rep, np.newaxis])).sum(axis=0)

        push_mag = float(np.linalg.norm(total_push))
        if push_mag < 1e-6:
            continue

        if push_mag > MAX_PUSH_PX:
            total_push = total_push / push_mag * MAX_PUSH_PX

        new_pos = expert_pos + total_push
        if is_walkable(new_pos):
            corrected[t] = new_pos
            n_corrected  += 1
        else:
            # Try perpendicular directions
            perp = np.array([-total_push[1], total_push[0]])
            perp_mag = np.linalg.norm(perp)
            if perp_mag > 1e-6:
                perp = perp / perp_mag * min(push_mag, MAX_PUSH_PX)
            for sign in [1.0, -1.0]:
                alt_pos = expert_pos + perp * sign
                if is_walkable(alt_pos):
                    corrected[t] = alt_pos
                    n_corrected  += 1
                    break

    if n_corrected > 0:
        print(f'[ExpertAdapter] Individual SFM: pushed {n_corrected} frames')
    return corrected


# ── Smooth reconnection after bypass ──────────────────────────────────────────

def _smooth_reconnection(traj_px: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Apply local smoothing to avoid sudden jumps in trajectory.

    Detects large frame-to-frame distances and applies Gaussian smoothing
    around those points.

    Args:
        traj_px: Trajectory in pixel coordinates (N x 2)
        window: Smoothing window size (frames)

    Returns:
        Smoothed trajectory
    """
    if len(traj_px) < 3:
        return traj_px

    smoothed = traj_px.copy().astype(float)

    # Detect sudden jumps (velocity spikes)
    velocities = np.linalg.norm(np.diff(traj_px, axis=0), axis=1)
    median_vel = np.median(velocities)
    threshold = median_vel * 3.0  # 3x median = likely a jump

    for t in range(1, len(traj_px) - 1):
        if velocities[t - 1] > threshold:
            # Smooth around this point
            start = max(0, t - window // 2)
            end = min(len(traj_px), t + window // 2 + 1)

            # Apply simple moving average
            weights = np.ones(end - start) / (end - start)
            smoothed[t, 0] = np.sum(traj_px[start:end, 0] * weights)
            smoothed[t, 1] = np.sum(traj_px[start:end, 1] * weights)

    return smoothed



# ── Unicycle kinematic correction via Pure Pursuit ────────────────────────────

def _apply_unicycle_kinematics(
    traj_px: np.ndarray,
    speed_px_per_frame: float,
    w_max_rad_per_frame: float = 0.3,
    lookahead_px: float = None,
) -> np.ndarray:
    """
    Pure Pursuit tracking with arc-length progress: satisfies unicycle dynamics.

    Uses cumulative arc length along the reference path so the lookahead target
    is always strictly ahead — prevents spinning when the robot diverges or
    overshoots the path end.

    Unicycle model (dt = 1 frame):
        x_{t+1} = x_t + v * cos(theta_t)
        y_{t+1} = y_t + v * sin(theta_t)
        theta_{t+1} = theta_t + w_t,   |w_t| <= w_max_rad_per_frame

    Args:
        traj_px:              Reference path (N x 2) pixel coords.
        speed_px_per_frame:   Fixed forward speed = v_mps * ppm / fps.
        w_max_rad_per_frame:  Max angular velocity per frame (rad). Default 0.3 ≈ 6 rad/s @ 20 fps.
        lookahead_px:         Pure Pursuit lookahead distance. Default = 8 × speed.

    Returns:
        Kinematically feasible trajectory (N x 2) pixel coords.
    """
    if len(traj_px) < 2:
        return traj_px.copy()

    if lookahead_px is None:
        lookahead_px = speed_px_per_frame * 8.0

    path = traj_px.astype(float)
    N = len(path)

    # Precompute cumulative arc length along reference path
    seg_lens = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_arc = arc_len[-1]

    # Initial heading from first segment
    d0 = path[1] - path[0]
    theta = float(np.arctan2(d0[1], d0[0])) if np.linalg.norm(d0) > 1e-6 else 0.0

    x, y = float(path[0, 0]), float(path[0, 1])
    result = np.empty((N, 2), dtype=float)
    result[0] = [x, y]

    # progress_arc: how far along the reference path we consider the robot to be.
    # Monotonically non-decreasing — prevents backward chasing.
    progress_arc = 0.0
    proj_idx = 0  # index of the closest path point (lower bound for search)

    for t in range(1, N):
        cur = np.array([x, y])

        # --- Update arc progress (monotonic) ---
        # Search forward from proj_idx for the closest path point
        best_dist = np.linalg.norm(path[proj_idx] - cur)
        for i in range(proj_idx + 1, min(proj_idx + 30, N)):
            d = np.linalg.norm(path[i] - cur)
            if d < best_dist:
                best_dist = d
                proj_idx = i
            elif d > best_dist + speed_px_per_frame:
                # Getting farther — stop searching
                break
        progress_arc = max(progress_arc, arc_len[proj_idx])

        # --- Lookahead target ---
        target_arc = progress_arc + lookahead_px

        if target_arc >= total_arc:
            # Near or past the end: steer toward final point then freeze
            target = path[-1]
            if np.linalg.norm(target - cur) < speed_px_per_frame * 1.5:
                result[t:] = cur
                break
        else:
            # Interpolate target position on path at target_arc
            idx = int(np.searchsorted(arc_len, target_arc, side='right')) - 1
            idx = max(0, min(idx, N - 2))
            seg = arc_len[idx + 1] - arc_len[idx]
            frac = (target_arc - arc_len[idx]) / seg if seg > 1e-6 else 0.0
            target = path[idx] + frac * (path[idx + 1] - path[idx])

        # --- Unicycle steering ---
        desired = np.arctan2(target[1] - y, target[0] - x)
        alpha = desired - theta
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi   # wrap to [-π, π]
        w = float(np.clip(alpha, -w_max_rad_per_frame, w_max_rad_per_frame))

        theta = (theta + w + np.pi) % (2 * np.pi) - np.pi
        x += speed_px_per_frame * np.cos(theta)
        y += speed_px_per_frame * np.sin(theta)
        result[t] = [x, y]

    return result


# ── Public API ────────────────────────────────────────────────────────────────

class ExpertAgentAdapter:

    @classmethod
    def inject(
        cls,
        df: pd.DataFrame,
        scene_name: str,
        framework=None,
        target_group_id: Optional[int] = None,  # noqa: ARG002
        num_experts: int = 3,
    ) -> pd.DataFrame:
        if framework is None:
            print('[ExpertAdapter] No framework provided — skipping.')
            return df

        # 1. Find earliest cohesive frame
        cohesive_start = _cohesive_first_frame(framework)

        scenario_metadata = getattr(framework, 'scenario_metadata', {})
        has_scenarios = any(len(v) > 0 for v in scenario_metadata.values())
        standard_ft_mode = (cohesive_start is None and not has_scenarios)

        # Fetch a larger candidate pool so A*-failed agents can be skipped
        # and replaced by the next-best candidate.
        pool_n = num_experts * 3

        queue_fm_list = []
        if standard_ft_mode:
            print('[ExpertAdapter] Standard-FT mode: selecting by pedestrian encounters')
            prior_ids = _find_top_pedestrian_encountering_agents(
                framework, top_n=pool_n
            )
        else:
            anchor = 'cohesive'
            if cohesive_start is None:
                # No cohesive group in this scene: anchor the spawn window on the first queue or gathering instead
                cohesive_start = _cohesive_first_frame(framework, ('queue', 'gathering'))
                anchor = 'queue/gathering'
            if cohesive_start is None:
                print('[ExpertAdapter] No group found — skipping.')
                return df
            print(f'[ExpertAdapter] First {anchor} frame: sim {cohesive_start} '
                  f'(spawn window: {cohesive_start}~{cohesive_start + SPAWN_WINDOW_SIM})')

            # 2. Build frame maps
            queue_fm_list = _build_queue_frame_maps(framework)
            qg_fm = _build_all_groups_frame_map(framework, include_types=('queue', 'gathering'))

            # 3. Select candidate pool — more than num_experts to allow fallback skipping
            prior_ids = _find_top_qg_violating_agents(
                framework, qg_fm, cohesive_start, top_n=pool_n
            )

        if not prior_ids:
            print('[ExpertAdapter] No eligible agents found — skipping.')
            return df

        # Shared resources: walkable map, ppm, cost_map
        from .astar_planner import astar_pathfind, build_occupancy_map, interpolate_path_at_fps

        walkable = getattr(framework, 'walkable', None)
        navmesh  = walkable if isinstance(walkable, np.ndarray) else None

        scenario_metadata = getattr(framework, 'scenario_metadata', {})
        ppm = float(getattr(framework, '_pixel_per_meter', 10.0))
        for key in ['queue', 'gathering', 'cohesive']:
            if key in scenario_metadata and len(scenario_metadata[key]) > 0:
                if 'pixel_per_meter' in scenario_metadata[key][0]:
                    ppm = float(scenario_metadata[key][0]['pixel_per_meter'])
                    break

        # Load bypass parameters from scenarios.yaml
        import yaml as _yaml
        _bypass_margin_meters = GATHERING_BYPASS_MARGIN_METERS
        _queue_clearance_px   = QUEUE_CLEARANCE_PX
        _queue_bypass_margin  = QUEUE_BYPASS_MARGIN
        try:
            with open('configs/scenarios.yaml', 'r') as _f:
                _sc = _yaml.safe_load(_f)
            _bypass_margin_meters = float(_sc.get('gathering', {}).get('bypass_margin_meters', GATHERING_BYPASS_MARGIN_METERS))
            _q_cfg = _sc.get('queue', {})
            if 'bypass_clearance_meters' in _q_cfg:
                _queue_clearance_px = float(_q_cfg['bypass_clearance_meters']) * ppm
            if 'bypass_margin_meters' in _q_cfg:
                _queue_bypass_margin = float(_q_cfg['bypass_margin_meters']) * ppm
        except Exception:
            pass

        if standard_ft_mode:
            # density cost (crowd avoidance) + border cost (wall avoidance)
            if navmesh is not None:
                cost_map = _build_pedestrian_density_cost_map(navmesh, framework)
                # reuse build_occupancy_map border logic: empty metadata = border only
                border_only = build_occupancy_map(
                    navmesh, {}, base_cost=0.0,
                    pixel_per_meter=ppm,
                    border_margin_meters=1.0,
                    border_cost=50.0,
                )
                cost_map = cost_map + border_only
            else:
                cost_map = None
        else:
            cost_map = build_occupancy_map(
                navmesh, scenario_metadata, base_cost=100.0,
                gathering_bypass_margin_meters=_bypass_margin_meters,
                pixel_per_meter=ppm,
                border_margin_meters=1.0,
                border_cost=50.0,
            ) if navmesh is not None else None

        fps = getattr(framework, 'dataset_fps', 20)
        speed_px_per_frame = 1.2 * ppm / fps

        # Phase 2 (Standard-FT): pre-build pedestrian frame map once for all experts
        if standard_ft_mode:
            _ped_df = df[df['group_type'] != 'expert'][['frame', 'x', 'y']]
            ped_frame_map: Dict[int, np.ndarray] = {
                int(f): grp[['x', 'y']].values
                for f, grp in _ped_df.groupby('frame')
            }
        else:
            ped_frame_map = {}

        # 4. Process candidate pool into expert trajectories.
        # A*-failed agents are skipped; next candidate is tried instead.
        ids_to_remove = set()
        expert_dfs    = []

        for rank, prior_id in enumerate(prior_ids):
            if len(expert_dfs) >= num_experts:
                break

            prior_df = df[df['agent_id'] == prior_id].sort_values('frame')
            if len(prior_df) == 0:
                print(f'[ExpertAdapter] [{rank+1}] agent_id={prior_id} not in DataFrame — skipping.')
                continue

            spawn_pos   = prior_df.iloc[0][['x', 'y']].values.astype(float)
            goal_pos    = prior_df.iloc[-1][['x', 'y']].values.astype(float)
            frame_start = int(prior_df.iloc[0]['frame'])
            print(f'[ExpertAdapter] [{rank+1}] agent_id={prior_id} '
                  f'spawn={spawn_pos}, goal={goal_pos}, frame_start={frame_start}')

            # An agent that enters or leaves at the map edge can have only a short stretch inside the scene
            if np.linalg.norm(goal_pos - spawn_pos) < MIN_EXPERT_DISPLACEMENT_PX:
                print(f'[ExpertAdapter] [{rank+1}] Start and goal are too close inside the scene — '
                      f'skipping agent, trying next.')
                continue

            # 4a. A* baseline path — skip agent if A* fails (try next candidate)
            if navmesh is None:
                print(f'[ExpertAdapter] [{rank+1}] No walkable map — skipping agent, trying next.')
                continue
            path_waypoints = astar_pathfind(navmesh, spawn_pos, goal_pos, cost_map=cost_map)
            if path_waypoints is None:
                print(f'[ExpertAdapter] [{rank+1}] A* failed — skipping agent, trying next.')
                continue
            traj_raw = interpolate_path_at_fps(
                path_waypoints, speed_mps=1.2, pixel_per_meter=ppm, fps=fps
            )
            print(f'[ExpertAdapter] [{rank+1}] A* path: '
                  f'{len(path_waypoints)} waypoints → {len(traj_raw)} frames')

            # 4b. Violation corrections
            if standard_ft_mode:
                # Phase 2: SFM repulsion from individual pedestrians
                traj_c = _correct_individual_violations(
                    traj_raw, frame_start, framework, ped_frame_map
                )
            else:
                traj_q = _correct_queue_violations(traj_raw, frame_start, queue_fm_list, framework,
                                                   clearance_px=_queue_clearance_px,
                                                   bypass_margin=_queue_bypass_margin)
                traj_g = _correct_gathering_violations(traj_q, frame_start, framework,
                                                        bypass_margin_meters=_bypass_margin_meters)
                traj_c = _correct_cohesive_violations(traj_g, frame_start, framework, df)

            # 4c. Unicycle kinematic constraints
            traj_final = _apply_unicycle_kinematics(traj_c, speed_px_per_frame)

            # 4d. Frame range filtering
            T              = len(traj_final)
            dataset_frames = list(range(frame_start, frame_start + T))
            max_ds_frame   = int(df['frame'].max())
            valid_pairs    = [(f, xy) for f, xy in zip(dataset_frames, traj_final)
                              if 0 <= f <= max_ds_frame]
            if not valid_pairs:
                print(f'[ExpertAdapter] [{rank+1}] No valid frames — skipping.')
                continue

            valid_frames, valid_pos = zip(*valid_pairs)

            # 4e. Build expert rows (reuse prior_id as expert_id)
            rows = [
                {
                    'scene':      scene_name,
                    'agent_id':   prior_id,
                    'agent_type': 0,
                    'frame':      int(f),
                    'x':          float(xy[0]),
                    'y':          float(xy[1]),
                    'group_id':   -2,
                    'group_type': 'expert',
                }
                for f, xy in zip(valid_frames, valid_pos)
            ]
            expert_dfs.append(pd.DataFrame(rows))
            ids_to_remove.add(prior_id)
            print(f'[ExpertAdapter] [{rank+1}] Expert agent_id={prior_id}: '
                  f'{len(rows)} frames (frame {valid_frames[0]}~{valid_frames[-1]})')

        if not expert_dfs:
            print('[ExpertAdapter] No expert trajectories generated — returning original df.')
            return df

        # 5. Remove all prior agents and inject all experts
        df_cleaned = df[~df['agent_id'].isin(ids_to_remove)].copy()
        combined   = pd.concat([df_cleaned] + expert_dfs, ignore_index=True)
        combined.sort_values(['frame', 'agent_id'], inplace=True)
        combined.reset_index(drop=True, inplace=True)
        print(f'[ExpertAdapter] Injected {len(expert_dfs)} expert agent(s).')
        return combined
