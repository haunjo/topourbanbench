import os
import sys
import math
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from scipy.interpolate import PchipInterpolator
import cv2
from PIL import Image
from diffusers import DDIMScheduler, DDPMScheduler

# Topology utilities (shared module in topourbanbench/utils/)
_TOPO_UTILS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if _TOPO_UTILS_DIR not in sys.path:
    sys.path.insert(0, _TOPO_UTILS_DIR)
from topourbanbench.utils.topology_utils import build_adjacency, topology_to_edges, TOPOLOGY_FOR_GROUP_TYPE

from CrowdES.emitter.emitter_pre_config import CrowdESEmitterPreConfig
from CrowdES.emitter.emitter_pre_model import CrowdESEmitterPreModel
from CrowdES.emitter.emitter_model import CrowdESEmitterModel
from CrowdES.emitter.emitter_pipeline import CrowdESEmitterPipeline
from CrowdES.simulator.simulator_config import CrowdESSimulatorConfig
from CrowdES.simulator.simulator_model import CrowdESSimulatorModel

from utils.utils import reproducibility_settings
from utils.image import sampling_xy_pos
from utils.homography import image2world, world2image
from utils.navmesh import get_control_point, filter_collinear_polygons, PathFinderNew
from utils.trajectory import preprocess_kdtree, batched_nearest_nonzero_idx_kdtree, KalmanModel

from CrowdES.group_graph import GroupGenerator


SAVE_IMAGE_FOR_DEBUG = False  # save image for debug
VERBOSE = True               # print debug information

# Controllability parameters
CONTROL_POPULATION_MULTIPLIER = None  # Population multiplier to control, None: no control, float: multiplier
CONTROL_POPULATION_SIZE = None        # Population size to control, None: no control, int: population size
CONTROL_AGENT_TYPE = None             # Agent type to control, None: no control, 1: pedestrian, 2: bicycle, 3: vehicle
CONTROL_WALKING_PACE = None           # Walking pace to control, None: no control, float: walking pace in m/s


class CrowdESFramework:
    """CrowdES framework for inference."""

    def __init__(self, config):
        self.config = config
        self.dataset_name = config.dataset.dataset_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pretrained model
        CrowdES_emitter_pre_checkpoint_dir = config.crowd_emitter.emitter_pre.checkpoint_dir.format(self.dataset_name)
        self.CrowdES_emitter_pre = CrowdESEmitterPreModel.from_pretrained(CrowdES_emitter_pre_checkpoint_dir)
        self.CrowdES_emitter_pre.to(self.device)
        self.CrowdES_emitter_pre.eval()

        CrowdES_emitter_checkpoint_dir = config.crowd_emitter.emitter.checkpoint_dir.format(self.dataset_name)
        self.CrowdES_emitter = CrowdESEmitterModel.from_pretrained(CrowdES_emitter_checkpoint_dir, use_safetensors=False)
        self.CrowdES_emitter.to(self.device)
        self.CrowdES_emitter.eval()
        CrowdES_emitter_diffusion_steps = config.crowd_emitter.emitter.diffusion_steps
        CrowdES_emitter_noise_scheduler = DDIMScheduler(num_train_timesteps=CrowdES_emitter_diffusion_steps, beta_schedule='linear')
        self.CrowdES_emitter_pipeline = CrowdESEmitterPipeline(unet=self.CrowdES_emitter, scheduler=CrowdES_emitter_noise_scheduler)

        CrowdES_simulator_checkpoint_dir = config.crowd_simulator.simulator.checkpoint_dir.format(self.dataset_name)
        self.CrowdES_simulator = CrowdESSimulatorModel.from_pretrained(CrowdES_simulator_checkpoint_dir)
        self.CrowdES_simulator.to(self.device)
        self.CrowdES_simulator.eval()

        # Set hyperparameters
        self.dataset_fps = self.config.dataset.dataset_fps
        self.simulator_fps = self.config.crowd_simulator.simulator.simulator_fps
        self.window_frame = self.config.crowd_emitter.emitter.window_size * self.simulator_fps
        self.traj_fut_frame = self.config.crowd_simulator.simulator.future_length

        # Prepare ORCA simulator if specified
        if self.config.crowd_simulator.type == 'ORCA':
            navmesh_vertices = [[x, 0, y] for x, y in image2world(np.array(self.navmesh['vertices'])[..., [0, 1]], self.H)]  # (w, h) -> (x, y) -> (x, 0, y)
            navmesh_polygons = self.navmesh['polygons']
            self.pf = PathFinderNew(navmesh_vertices, navmesh_polygons)
            self.sim_agent_id_to_agent_id = {}

        # Initialize group behavior modules (optional)
        self.enable_group_behavior = hasattr(config, 'group_config') and config.group_config is not None
        if self.enable_group_behavior:
            import yaml
            with open(config.group_config, 'r') as f:
                self.group_config = yaml.safe_load(f)
            self.group_generator = GroupGenerator(self.group_config)
            # self.group_force_model = GroupForceModel(self.group_config)
            print(f"[Group Behavior] Enabled with config: {config.group_config}")
        else:
            self.group_generator = None
            self.group_force_model = None
            print("[Group Behavior] Disabled (no group_config provided)")

    def set_randomize_params(self, enabled: bool):
        """Enable per-scene parameter randomization using yaml min/max ranges."""
        self._randomize_params = enabled

    def _is_walkable(self, position, margin_pixels=0):
        """
        Check if a position (x, y) is walkable with circular robot clearance margin.

        Args:
            position: (x, y) in pixels
            margin_pixels: Clearance radius in pixels (for robot navigation)

        Returns:
            bool: True if position and surrounding area are walkable, False otherwise
        """
        x, y = int(position[0]), int(position[1])
        H, W = self.walkable.shape

        # Out of bounds check with margin
        if x - margin_pixels < 0 or x + margin_pixels >= W or y - margin_pixels < 0 or y + margin_pixels >= H:
            return False

        # Check circular area around position
        if margin_pixels == 0:
            # Fast path: just check center
            return self.walkable[y, x] > 0
        else:
            # Check all pixels in circular margin
            margin_int = int(np.ceil(margin_pixels))
            for dy in range(-margin_int, margin_int + 1):
                for dx in range(-margin_int, margin_int + 1):
                    # Circular check
                    if dx*dx + dy*dy <= margin_pixels*margin_pixels:
                        if self.walkable[y + dy, x + dx] == 0:
                            return False
            return True

    def _is_walkable_directional(self, position, direction, margin_pixels):
        """
        Check if a position is walkable along a specific direction with margin.
        Used for queue: only checks walkable space in the queue extension direction.

        Args:
            position: (x, y) in pixels
            direction: (dx, dy) unit vector (queue direction)
            margin_pixels: How far to check along the direction (in pixels)

        Returns:
            bool: True if walkable along the direction, False otherwise
        """
        x, y = int(position[0]), int(position[1])
        H, W = self.walkable.shape

        # Basic bounds & walkable check at center
        if x < 0 or x >= W or y < 0 or y >= H:
            return False
        if self.walkable[y, x] == 0:
            return False

        # Sample points along the direction from position to position + margin * direction
        num_samples = max(2, int(np.ceil(margin_pixels)))
        for t in np.linspace(0, margin_pixels, num_samples):
            sx = int(position[0] + t * direction[0])
            sy = int(position[1] + t * direction[1])
            if sx < 0 or sx >= W or sy < 0 or sy >= H:
                return False
            if self.walkable[sy, sx] == 0:
                return False

        return True

    # ====================================================================================
    # Cohesive Group Cohesion
    # ====================================================================================

    def _load_cohesive_config(self):
        """Load cohesive config once and cache."""
        if not hasattr(self, '_cohesive_config'):
            import yaml
            with open('configs/scenarios.yaml', 'r') as f:
                scenario_config = yaml.safe_load(f)
            cohesive_params = scenario_config.get('cohesive', {})
            # Inherit enabled flag from scenarios.cohesive section
            scenarios_enabled = scenario_config.get('scenarios', {}).get('cohesive', {})
            cohesive_params['enabled'] = scenarios_enabled.get('enabled', True)
            self._cohesive_config = cohesive_params

            # Resolve pixel_per_meter from scene metadata
            queues = self.scenario_metadata.get('queue', [])
            self._pixel_per_meter = queues[0].get('pixel_per_meter', 50.0) if queues else 50.0

        return self._cohesive_config

    def _align_cohesive_goals(self, first_new_group_id: int = 0):
        """
        Set up Leader-Follower formation for cohesive groups.

        Called once after group assignment in process_emitter().

        Leader:    Keeps original diffusion-generated goal (natural path)
        Followers: Goal = leader's goal + lateral offset (side-by-side at destination)

        Also aligns:
          - preferred_speed → group average (same pace)
          - frame_origin    → group minimum (depart together)
          - origin_xy       → side-by-side around origin centroid

        Args:
            first_new_group_id: Only process groups with group_id >= this value.
                                Prevents overwriting frame_global of groups created
                                in previous emission windows.
        """
        config = self._load_cohesive_config()
        spacing_meters = config.get('side_by_side_spacing_meters', config.get('side_by_side_spacing', 0.6))
        spacing_px = spacing_meters * self._pixel_per_meter

        # Apply to cohesive and crosswalk groups (both walk side-by-side)
        WALKING_GROUP_TYPES = {'cohesive', 'crosswalk'}

        for group in self.group_graph['groups']:
            if group['group_id'] < first_new_group_id:
                continue  # Skip groups from previous emission windows
            if group.get('group_type') not in WALKING_GROUP_TYPES:
                continue

            member_ids = group['member_ids']
            if len(member_ids) < 2:
                continue

            # Collect current parameters
            origins = np.array([self.agent_parameter[aid]['origin_xy'] for aid in member_ids])
            goals = np.array([self.agent_parameter[aid]['goal_xy'] for aid in member_ids])
            speeds = np.array([self.agent_parameter[aid]['preferred_speed'] for aid in member_ids])
            frames = np.array([self.agent_parameter[aid]['frame_origin'] for aid in member_ids])

            # ---- Leader selection: agent closest to goal centroid ----
            goal_centroid = goals.mean(axis=0)
            dists_to_centroid = np.linalg.norm(goals - goal_centroid, axis=1)
            leader_local_idx = int(np.argmin(dists_to_centroid))
            leader_id = member_ids[leader_local_idx]
            leader_goal = goals[leader_local_idx].copy()

            # Shared values
            origin_centroid = origins.mean(axis=0)
            shared_speed = float(speeds.mean())
            shared_frame = int(frames.min())

            # Walking direction & perpendicular
            walk_dir = leader_goal - origin_centroid
            walk_dist = np.linalg.norm(walk_dir)
            if walk_dist > 1e-3:
                walk_dir = walk_dir / walk_dist
                perp_dir = np.array([-walk_dir[1], walk_dir[0]])
            else:
                perp_dir = np.array([1.0, 0.0])

            # ---- Assign origins & goals in side-by-side formation ----
            N = len(member_ids)
            formation_offsets = {}  # {agent_id: offset_vector} for formation keeping

            for i, aid in enumerate(member_ids):
                lateral_offset = (i - (N - 1) / 2.0) * spacing_px
                offset_vec = lateral_offset * perp_dir

                # Origin: side-by-side around origin centroid
                new_origin = origin_centroid + offset_vec
                if self._is_walkable(new_origin):
                    self.agent_parameter[aid]['origin_xy'] = new_origin
                    self.agent_trajectory[aid] = np.array([new_origin])

                # Goal: leader's goal + same lateral offset
                new_goal = leader_goal + offset_vec
                self.agent_parameter[aid]['goal_xy'] = new_goal
                # Store signed scalar so cohesion force can rotate with leader heading
                formation_offsets[aid] = float(lateral_offset)

                # Speed & timing sync
                self.agent_parameter[aid]['preferred_speed'] = shared_speed
                self.agent_parameter[aid]['frame_origin'] = shared_frame
                self.agent_parameter[aid]['frame_global'] = self.current_frame + shared_frame

            # Save leader & offsets in group metadata
            group['leader_id'] = leader_id
            group['formation_offsets'] = formation_offsets

    def _apply_cohesion_force(self, traj_fut, active_agent_ids):
        """
        Formation-keeping force: maintain side-by-side offset relative to leader.

        The lateral offset is rotated with the leader's current heading at each
        timestep, so the formation stays side-by-side even when the leader turns.
        formation_offsets stores signed scalar magnitudes (px), not fixed vectors.
        """
        config = self._load_cohesive_config()
        cohesion_strength = config.get('cohesion_strength', 0.7)

        traj_fut = traj_fut.astype(np.float64)

        WALKING_GROUP_TYPES = {'cohesive', 'crosswalk'}

        for group in self.group_graph.get('groups', []):
            if group.get('group_type') not in WALKING_GROUP_TYPES:
                continue

            leader_id = group.get('leader_id')
            offsets = group.get('formation_offsets', {})
            if leader_id is None or leader_id not in active_agent_ids:
                continue

            leader_idx = active_agent_ids.index(leader_id)
            leader_traj = traj_fut[leader_idx]  # (T_fut, 2)

            # Per-timestep perpendicular direction from leader velocity
            T = len(leader_traj)
            vels = np.zeros((T, 2))
            if T > 1:
                vels[:-1] = leader_traj[1:] - leader_traj[:-1]
                vels[-1] = vels[-2]
            vel_norm = np.linalg.norm(vels, axis=1, keepdims=True).clip(1e-6)
            walk_dirs = vels / vel_norm          # (T, 2)
            # perpendicular: rotate 90° CCW
            perp_dirs = np.stack([-walk_dirs[:, 1], walk_dirs[:, 0]], axis=1)  # (T, 2)

            for mid, lateral_offset in offsets.items():
                if mid == leader_id or mid not in active_agent_ids:
                    continue

                follower_idx = active_agent_ids.index(mid)

                # Desired: leader position + laterally rotated offset
                desired = leader_traj + lateral_offset * perp_dirs  # (T, 2)
                correction = desired - traj_fut[follower_idx]
                traj_fut[follower_idx] += correction * cohesion_strength

        return traj_fut

    def _build_goal_pools(self):
        """
        Build goal pools for deterministic queue/gathering agent spawn.

        Queue: Line formation along direction with spacing (walkable-aware)
        Gathering: Annulus (donut) formation around center (walkable-aware)

        Each pool entry: (origin_xy, goal_xy, scenario_type, scenario_idx, group_local_id)
        """
        # Use the already-loaded mode-specific config (set via config.group_config).
        # Previously hardcoded to 'configs/scenarios.yaml', which ignored mode overrides.
        scenario_config = self.group_config

        # Robot clearance margin
        robot_clearance_meters = scenario_config.get('robot_clearance_meters', 2.5)

        queue_config = scenario_config.get('queue', {})
        gathering_config = scenario_config.get('gathering', {})

        max_queue_size = queue_config['max_queue_size']
        # Per-queue size randomization: each queue independently samples from [min, max].
        # When max_queue_size_min == max_queue_size (default), all queues get the same size.
        q_size_min = int(queue_config.get('max_queue_size_min', max_queue_size))
        _sp_min = float(queue_config.get('spacing_meters_min', 0.8))
        _sp_max = float(queue_config.get('spacing_meters_max', 1.8))
        queue_length_meters = float(queue_config.get('queue_length_meters', 10.0))
        min_bypass_meters = queue_config.get('min_bypass_meters', 4.0)

        min_gathering_size = gathering_config['min_gathering_size']
        max_gathering_size = gathering_config['max_gathering_size']
        _g_sp_min = float(gathering_config.get('spacing_meters_min', 0.8))
        _g_sp_max = float(gathering_config.get('spacing_meters_max', 2.0))

        # Domain randomization: per-scene overrides (queue_length, gathering size)
        if getattr(self, '_randomize_params', False):
            ql_min = float(queue_config.get('queue_length_meters_min', queue_length_meters))
            queue_length_meters = float(np.random.uniform(ql_min, queue_length_meters))

            max_gathering_size = int(np.random.randint(min_gathering_size, max_gathering_size + 1))

            print(f'[Randomize] q_size=[{q_size_min},{max_queue_size}] per-queue, '
                  f'queue_length={queue_length_meters:.1f}m, '
                  f'max_gathering_size={max_gathering_size}')

        # Filter gatherings that are too close to any queue line.
        # Uses queue_overlap_clearance_meters from config (runtime re-check since
        # cached metadata may have been generated with a different threshold).
        clearance_meters = gathering_config.get('queue_overlap_clearance_meters', 5.0)
        queue_list = self.scenario_metadata.get('queue', [])
        gathering_list = self.scenario_metadata.get('gathering', [])
        if queue_list and gathering_list:
            ref_ppm = gathering_list[0].get('pixel_per_meter', 50.0)
            clearance_px = clearance_meters * ref_ppm

            def _pt_seg_dist(p, a, b):
                ab = b - a
                ab2 = np.dot(ab, ab)
                if ab2 < 1e-6:
                    return np.linalg.norm(p - a)
                t = np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0)
                return np.linalg.norm(p - (a + t * ab))

            filtered = []
            for g in gathering_list:
                gc = np.array(g['center'], dtype=float)
                too_close = False
                for q in queue_list:
                    qa = np.array(q['goal_point'], dtype=float)
                    qd = np.array(q['direction'], dtype=float)
                    tail_px = float(q.get('walkable_length_px', q.get('length_px', 0)))
                    qb = qa + qd * tail_px
                    if _pt_seg_dist(gc, qa, qb) < clearance_px:
                        too_close = True
                        break
                if not too_close:
                    filtered.append(g)
            if len(filtered) < len(gathering_list):
                print(f'[QueueClearance] Removed {len(gathering_list)-len(filtered)} gathering(s) '
                      f'within {clearance_meters}m of queue')
                self.scenario_metadata['gathering'] = filtered

        # Default spawn radius (will be overridden by metadata's adaptive radius)
        # Note: Gathering metadata already has spawn_radius in pixels from validation
        default_spawn_radius_px = 150.0  # Fallback: ~3.7m at 40.96px/m

        # Initialize pools
        self.deterministic_goal_pool = []  # List of (origin_xy, goal_xy, scenario_type, scenario_idx, group_local_id)

        # ====================================================================================
        # Queue Goal Pool (Line Formation)
        # ====================================================================================
        for queue_idx, queue_info in enumerate(self.scenario_metadata.get('queue', [])):
            goal_point = np.array(queue_info['goal_point'])  # (x, y) in pixels
            direction = np.array(queue_info['direction'])  # Unit vector
            pixel_per_meter = queue_info.get('pixel_per_meter', 50.0)

            # Per-queue independent size sample from [q_size_min, max_queue_size]
            queue_max_size = int(np.random.randint(q_size_min, max_queue_size + 1))

            # Sample spacing per queue from yaml spacing_meters_min/max
            queue_spacing_meters = float(np.random.uniform(_sp_min, _sp_max))
            queue_spacing_px = queue_spacing_meters * pixel_per_meter

            # Robot clearance margin in pixels (checked along queue direction only)
            margin_pixels = robot_clearance_meters * pixel_per_meter

            # Capacity uses actual spacing (consistent with placement below)
            walkable_length_px = queue_info.get('walkable_length_px', None)
            if walkable_length_px is not None:
                capacity = min(queue_max_size, max(0, int(walkable_length_px / queue_spacing_px)))
                if capacity == 0:
                    print(f'[QueuePool] Queue {queue_idx}: skip — '
                          f'walkable {walkable_length_px/pixel_per_meter:.1f}m '
                          f'< spacing {queue_spacing_meters:.1f}m')
                    continue
                print(f'[QueuePool] Queue {queue_idx}: capacity={capacity} '
                      f'(walkable={walkable_length_px/pixel_per_meter:.1f}m, '
                      f'spacing={queue_spacing_meters:.2f}m, max_size={queue_max_size})')
            else:
                capacity = queue_max_size

            # Generate positions along the line, filtering by walkable mask
            valid_positions = []
            attempt = 0
            max_attempts = capacity * 3  # Try 3x to find enough valid positions

            while len(valid_positions) < capacity and attempt < max_attempts:
                # Positions extend from goal_point along direction (0=front of queue)
                position = goal_point + attempt * queue_spacing_px * direction

                if attempt == 0:
                    # First person: stand at goal_point, just check the point itself
                    if self._is_walkable(position):
                        valid_positions.append(position)
                else:
                    # Subsequent people: check walkable along queue direction (backward)
                    if self._is_walkable_directional(position, direction, margin_pixels):
                        valid_positions.append(position)

                attempt += 1

            # Add to pool (origin == goal for stationary agents)
            for i, position in enumerate(valid_positions):
                self.deterministic_goal_pool.append({
                    'origin_xy': position.copy(),
                    'goal_xy': position.copy(),
                    'scenario_type': 'queue',
                    'scenario_idx': queue_idx,
                    'group_local_id': i,  # Position in line (0=front, max-1=back)
                })

            if len(valid_positions) < max_queue_size:
                print(f"[Warning] Queue {queue_idx}: Only {len(valid_positions)}/{max_queue_size} walkable positions found")

        # ====================================================================================
        # Gathering Goal Pool (Circle Formation facing center)
        # ====================================================================================
        for gather_idx, gather_info in enumerate(self.scenario_metadata.get('gathering', [])):
            center = np.array(gather_info['center'], dtype=float)  # (x, y) in pixels

            # Use adaptive spawn_radius from validation (calculated based on local walkable area)
            spawn_radius = float(gather_info.get('spawn_radius', default_spawn_radius_px))
            pixel_per_meter_local = float(gather_info.get('pixel_per_meter', 40.96))

            # Sample spacing per gathering from yaml spacing_meters_min/max
            gathering_spacing_meters = float(np.random.uniform(_g_sp_min, _g_sp_max))
            spacing_px = gathering_spacing_meters * pixel_per_meter_local

            # Count derived from circumference / spacing (radius-dependent, spacing-randomized)
            circumference = 2 * np.pi * spawn_radius
            target_count = int(circumference / spacing_px)
            target_count = int(np.clip(target_count, min_gathering_size, max_gathering_size))
            print(f'[GatheringPool] Gathering {gather_idx}: target={target_count} '
                  f'(radius={spawn_radius/pixel_per_meter_local:.1f}m, '
                  f'spacing={gathering_spacing_meters:.2f}m)')

            # No margin needed (validation already ensured walkable area)
            margin_pixels = 0.0

            # Radial search parameters for fallback
            radial_search_step = 3   # px
            radial_max_expand  = int(spawn_radius * 0.25)  # search ±25% if exact radius not walkable

            # Place agents on the ring with non-uniform gaps (O-space preserved).
            # N people share 2π by splitting it into N gaps via Dirichlet distribution:
            #   gap[i] = min_angle + extra[i],  extra ~ Dirichlet(1,...,1)
            #   sum(gap) = 2π  →  full ring always maintained
            # Natural clustering emerges: some adjacent pairs close, others far apart.
            min_sep_px = 0.5 * pixel_per_meter_local       # 0.5m hard minimum
            min_angle = min_sep_px / spawn_radius           # angle equivalent
            if target_count * min_angle > 2 * np.pi:
                target_count = max(1, int(2 * np.pi / min_angle))
            raw = np.random.exponential(1.0, size=target_count)
            extra_gaps = raw / raw.sum() * (2 * np.pi - target_count * min_angle)
            gaps = extra_gaps + min_angle                   # each >= min_angle, sum = 2π
            start_angle = np.random.uniform(0, 2 * np.pi)
            agent_angles = start_angle + np.concatenate([[0], np.cumsum(gaps[:-1])])

            valid_positions = []
            for angle in agent_angles:
                cos_a, sin_a = np.cos(angle), np.sin(angle)

                placed = False
                # Try spawn_radius first, then search nearby if needed
                for dr in range(0, radial_max_expand + 1, radial_search_step):
                    for r in ([spawn_radius] if dr == 0
                              else [spawn_radius + dr, spawn_radius - dr]):
                        if r < 10:   # minimum 10px from center
                            continue
                        pos = center + np.array([r * cos_a, r * sin_a])
                        if self._is_walkable(pos, margin_pixels):
                            valid_positions.append(pos)
                            placed = True
                            break
                    if placed:
                        break

            print(f"[Gathering {gather_idx}] Placed {len(valid_positions)}/{target_count} agents "
                  f"(center=({int(center[0])},{int(center[1])}), "
                  f"spawn_radius={spawn_radius:.0f}px, spacing={gathering_spacing_meters}m)")

            # Add to pool: origin = standing position, goal = center (agent faces inward)
            for i, position in enumerate(valid_positions):
                self.deterministic_goal_pool.append({
                    'origin_xy': position.copy(),
                    'goal_xy': center.copy(),
                    'scenario_type': 'gathering',
                    'scenario_idx': gather_idx,
                    'group_local_id': i,
                })

            if len(valid_positions) < target_count:
                print(f"[Warning] Gathering {gather_idx}: Only {len(valid_positions)}/{target_count} "
                      f"walkable positions found")

        # ====================================================================================
        # Crosswalk Goal Pool (Moving agents: origin → goal, not stopped)
        # ====================================================================================
        crosswalk_config = scenario_config.get('crosswalk', {})
        num_per_dir = crosswalk_config.get('num_agents_per_direction', 4)
        spawn_margin_meters = crosswalk_config.get('spawn_margin', 0.5)

        for cw_idx, cw_info in enumerate(self.scenario_metadata.get('crosswalk', [])):
            point_a = np.array(cw_info['point_a'])
            point_b = np.array(cw_info['point_b'])
            cw_width = cw_info.get('width', 100)  # pixels, long axis width
            pixel_per_meter = cw_info.get('pixel_per_meter', 50.0)

            # Crossing direction (A→B) and lateral direction (along crosswalk)
            cross_dir = point_b - point_a
            cross_dist = np.linalg.norm(cross_dir)
            if cross_dist < 1e-3:
                continue
            cross_dir = cross_dir / cross_dist
            lateral_dir = np.array([-cross_dir[1], cross_dir[0]])

            spacing_px = queue_spacing_meters * pixel_per_meter
            margin_px = spawn_margin_meters * pixel_per_meter

            # Generate agents for both directions (A→B and B→A)
            for direction_label, (origin_base, goal_base) in enumerate([(point_a, point_b), (point_b, point_a)]):
                # Offset each direction to opposite sides (right-hand traffic)
                side_offset = lateral_dir * (cw_width * 0.15) * (1 if direction_label == 0 else -1)

                for i in range(num_per_dir):
                    lateral_offset = (i - (num_per_dir - 1) / 2.0) * spacing_px
                    pos_offset = lateral_dir * lateral_offset + side_offset

                    origin = origin_base + pos_offset
                    goal = goal_base + pos_offset

                    if self._is_walkable(origin) and self._is_walkable(goal):
                        self.deterministic_goal_pool.append({
                            'origin_xy': origin.copy(),
                            'goal_xy': goal.copy(),
                            'scenario_type': 'crosswalk',
                            'scenario_idx': cw_idx,
                            'group_local_id': direction_label * num_per_dir + i,
                            'flow_direction': direction_label,  # 0=A→B, 1=B→A
                            'is_moving': True,  # Not stopped — walks to goal
                        })

            print(f"[Crosswalk {cw_idx}] Spawning up to {num_per_dir * 2} agents "
                  f"(A→B: {num_per_dir}, B→A: {num_per_dir})")

        num_crosswalk_target = len(self.scenario_metadata.get('crosswalk', [])) * num_per_dir * 2
        print(f"[Goal Pool] Built {len(self.deterministic_goal_pool)} deterministic spawn points "
              f"(Target: Queue={len(self.scenario_metadata.get('queue', []))*max_queue_size}, "
              f"Gathering={len(self.scenario_metadata.get('gathering', []))*max_gathering_size}, "
              f"Crosswalk={num_crosswalk_target})")

    def _point_to_segment_distance(self, point, seg_start, seg_end):
        """
        Calculate minimum distance from point to line segment.

        Args:
            point: (2,) point coordinates
            seg_start: (2,) segment start
            seg_end: (2,) segment end

        Returns:
            distance: float
        """
        # Vector from start to end
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)

        if seg_len_sq < 1e-6:
            # Degenerate segment (point)
            return np.linalg.norm(point - seg_start)

        # Project point onto line (parameterized by t)
        t = np.dot(point - seg_start, seg_vec) / seg_len_sq
        t = np.clip(t, 0, 1)  # Clamp to segment bounds

        # Closest point on segment
        closest = seg_start + t * seg_vec

        # Distance
        return np.linalg.norm(point - closest)

    def initialize_scene(self, img, seg, walkable, navmesh, H, appearance_gt=None, population_gt=None, scenario_gt=None, scenario_metadata=None):
        """Initialize the scene for crowd emitter.

        Params:
            img (np.array): An image.
            seg (np.array): A segmentation map.
            walkable (np.array): A walkable map.
            navmesh (list): A navigation mesh.
            H (np.array): A homography matrix.
            appearance_gt (np.array, optional): A ground truth appearance density map. Defaults to None.
            population_gt (np.array, optional): A ground truth population density map. Defaults to None.
            scenario_gt (np.array, optional): A ground truth scenario density map for goal sampling. Defaults to None.
            scenario_metadata (dict, optional): Unified scenario metadata with keys ['queue', 'cohesive', 'gathering']. Defaults to None.
        """

        self.img = img
        self.seg = seg
        self.walkable = walkable
        self.navmesh = navmesh
        self.H = H
        self.scene_size = self.img.shape[:2]  # (H, W)

        self.appearance_gt = appearance_gt
        self.scenario_gt = scenario_gt
        self.population_gt = population_gt
        self.scenario_metadata = scenario_metadata if scenario_metadata is not None else {'queue': [], 'cohesive': [], 'gathering': []}

        # Build goal pools for deterministic queue/gathering agents
        self._build_goal_pools()

        # Preparation
        self.navmesh['polygons'] = filter_collinear_polygons(navmesh['vertices'], navmesh['polygons'])
        self.env_pixel_meter = self.config.crowd_simulator.simulator.environment_pixel_meter
        self.env_size = tuple(self.config.crowd_simulator.simulator.environment_size)
        env_base_pointer_h = np.linspace(-self.env_size[0] / 2, self.env_size[0] / 2, self.env_size[0])
        env_base_pointer_w = np.linspace(-self.env_size[1] / 2, self.env_size[1] / 2, self.env_size[1])
        self.env_base_pointer = np.stack(np.meshgrid(env_base_pointer_h, env_base_pointer_w), axis=-1) * self.env_pixel_meter  # shape=(64, 64, 2)
        self.kdtree = preprocess_kdtree(self.walkable)

        # Initialize group graph (empty at start)
        self.group_graph = {'groups': [], 'agent_to_group': {}, 'scenario_types': {}}

        # Queue behavior tracking
        self.agent_is_queue = {}  # {agent_id: bool}
        self.agent_is_stopped = {}  # {agent_id: bool}
        self.agent_queue_id = {}  # {agent_id: queue_idx}
        self.agent_is_crosswalk = {}  # {agent_id: bool} — moving crosswalk agents

    def process_emitter_pre(self):
        # Extract the scene layouts from the image
        process_image_size = tuple(self.config.crowd_emitter.emitter_pre.image_size)
        input_types = self.config.crowd_emitter.emitter_pre.input_types
        input = []
        input.append(self.img) if 'scene_img' in input_types else None
        input.append(self.seg) if 'scene_seg' in input_types else None
        input = [(i[:, :, None] if i.ndim == 2 else i) for i in input]
        input = np.concatenate(input, axis=2)
        input = torch.FloatTensor(input).permute(2, 0, 1)
        input = torch.nn.functional.interpolate(input.unsqueeze(0), size=process_image_size, mode='bilinear', align_corners=False)

        input = input.to(self.device)
        outputs = self.CrowdES_emitter_pre(input)

        pred_batch = outputs.logits
        pred_batch = torch.nn.functional.interpolate(pred_batch, size=self.scene_size, mode='bilinear', align_corners=False)
        temperature = self.config.crowd_emitter.emitter_pre.population_density_temperature
        pred_batch = torch.sigmoid(pred_batch / temperature).squeeze(0)  # Sigmoid with steep slope using temperature

        appearance_density_map = pred_batch[0].detach().cpu().numpy()
        population_density_map = pred_batch[1].detach().cpu().numpy()
        appearance_density_map = appearance_density_map / appearance_density_map.max()  # normalize
        population_density_map = population_density_map / population_density_map.max()  # normalize
        self.appearance_density_map = appearance_density_map  # We will limit the appearance points to the walkable area in the emitter.
        self.population_density_map = population_density_map * self.walkable

        # Put the ground truth if it is not None
        if self.appearance_gt is not None:
            assert self.appearance_gt.shape == self.appearance_density_map.shape
            self.appearance_density_map = self.appearance_gt
        if self.population_gt is not None:
            assert self.population_gt.shape == self.population_density_map.shape
            self.population_density_map = self.population_gt

        # Set scenario density map (for goal sampling)
        if self.scenario_gt is not None:
            assert self.scenario_gt.shape == self.appearance_density_map.shape
            self.scenario_density_map = self.scenario_gt
        else:
            self.scenario_density_map = self.appearance_density_map

        # Make the population probability to 0 if the value is too small.
        pred_logit_batch = outputs.logits_unit.squeeze(0).sigmoid().detach().cpu().numpy()
        population_probability_threshold = self.config.crowd_emitter.emitter_pre.population_probability_threshold
        pred_logit_batch[pred_logit_batch < population_probability_threshold] = 0
        self.population_probability = pred_logit_batch / pred_logit_batch.sum()  # normalize

        # Prepare conditions for Emitter
        # IMPORTANT: For spawn-goal decoupling, we remap the channels:
        #   - 'scene_population' slot → appearance_density_map (for spawn sampling)
        #   - 'scene_appearance' slot → scenario_density_map (for goal sampling)
        # This works because emitter_pipeline samples directly from condition[-3] and [-2]
        # The model weights don't need retraining as sampling happens in reprog stage
        input_types_e = self.config.crowd_emitter.emitter.input_types
        environment_e = []
        environment_e.append(self.img) if 'scene_img' in input_types_e else None
        environment_e.append(self.seg) if 'scene_seg' in input_types_e else None
        # Remap: Use appearance for spawn (channel -3 in pipeline)
        environment_e.append(self.appearance_density_map) if 'scene_population' in input_types_e else None
        # Remap: Use scenario for goal (channel -2 in pipeline)
        environment_e.append(self.scenario_density_map) if 'scene_appearance' in input_types_e else None
        environment_e = [(c[:, :, None] if c.ndim == 2 else c) for c in environment_e]
        self.environment_e = np.concatenate(environment_e, axis=2)

        # Prepare conditions for Simulator
        environment_types_s = self.config.crowd_simulator.simulator.environment_types
        environment_s = []
        environment_s.append(self.img) if 'scene_img' in environment_types_s else None
        environment_s.append(self.seg) if 'scene_seg' in environment_types_s else None
        environment_s.append(self.population_density_map) if 'scene_population' in environment_types_s else None
        environment_s.append(self.appearance_density_map) if 'scene_appearance' in environment_types_s else None
        environment_s = [(c[:, :, None] if c.ndim == 2 else c) for c in environment_s]
        self.environment_s = np.concatenate(environment_s, axis=2)
        self.environment_s = torch.FloatTensor(self.environment_s).permute(2, 0, 1)

        return self.appearance_density_map, self.population_density_map, self.population_probability

    def process_emitter(self):
        # Step 1: Plan the number of agents to emit
        max_num_agents_pad = self.config.crowd_emitter.emitter.max_num_agents_pad
        num_agents = np.random.choice(len(self.population_probability), p=self.population_probability)
        num_agents = int(num_agents * self.config.crowd_emitter.emitter_pre.population_multiplier)
        
        # Controllability - Population size
        num_agents = num_agents if CONTROL_POPULATION_MULTIPLIER is None else int(num_agents * CONTROL_POPULATION_MULTIPLIER)
        num_agents = num_agents if CONTROL_POPULATION_SIZE is None else CONTROL_POPULATION_SIZE

        # Count agents in scene (exclude stopped deterministic agents from target)
        # Deterministic agents (queue/gathering) are stationary and don't count toward dynamic population
        num_moving_agents = len([aid for aid in self.agent_ids_in_current_scene
                                 if not self.agent_is_stopped.get(aid, False)])

        num_agents_to_emit = num_agents - num_moving_agents
        num_agents_to_emit = min(num_agents_to_emit, max_num_agents_pad)

        # For initialization, cut down the number of agents in half
        # UNLESS we have deterministic agents to spawn (queue/gathering)
        if self.current_frame < 0:
            if len(self.deterministic_goal_pool) > 0:
                num_agents_to_emit = max(num_agents_to_emit, len(self.deterministic_goal_pool))
            else:
                # Original behavior: cut in half if no deterministic agents
                num_agents_to_emit = math.ceil(num_agents_to_emit / 2)

        if num_agents_to_emit <= 0:
            self.statistics_added = 0
            return self.agent_ids_in_current_scene

        # Step 2: Prepare conditions
        # Make binary map of individual positions in the previous frame.
        process_image_size = tuple(self.config.crowd_emitter.emitter.image_size)
        agent_dist = np.zeros(self.scene_size)
        if len(self.agent_ids_in_current_scene) > 0:
            agent_coord = [self.agent_trajectory[agent_id][-1] if len(self.agent_trajectory[agent_id]) > 0 else np.empty((2,)) for agent_id in self.agent_ids_in_current_scene]
            agent_coord = np.stack(agent_coord, axis=0)
            agent_coord = world2image(agent_coord, self.H)
            agent_coord = agent_coord.astype(int)
            agent_coord = np.clip(agent_coord, 0, np.array(self.scene_size)[[1, 0]] - 1)
            agent_dist[agent_coord[:, 1], agent_coord[:, 0]] = 1
        
        # Make the condition for the crowd emitter
        condition = []
        condition.append(self.environment_e)
        condition.append(agent_dist)
        condition = [(c[:, :, None] if c.ndim == 2 else c) for c in condition]
        condition = np.concatenate(condition, axis=2)
        condition = torch.FloatTensor(condition).permute(2, 0, 1)
        condition = torch.nn.functional.interpolate(condition.unsqueeze(0), size=process_image_size, mode='bilinear', align_corners=False)
        condition = condition.to(self.device)

        # Step 3: Populate the scene with crowd emitter
        # ====================================================================================
        # DETERMINISTIC + SAMPLED AGENT GENERATION
        # ====================================================================================
        # Strategy:
        # - Consume deterministic pool (queue/gathering) as early as possible
        # - Fill remaining quota with sampled agents (cohesive candidates)
        # - Respect num_agents_to_emit limit per frame

        # Consume from deterministic pool if available
        num_deterministic = min(len(self.deterministic_goal_pool), num_agents_to_emit)
        num_sampled = num_agents_to_emit - num_deterministic


        # Pop deterministic agents from goal pool
        deterministic_agents = []
        for _ in range(num_deterministic):
            deterministic_agents.append(self.deterministic_goal_pool.pop(0))  # FIFO

        # Generate sampled agents (cohesive candidates) using diffusion model
        if num_sampled > 0:
            diffusion_steps = self.config.crowd_emitter.emitter.diffusion_steps
            leapfrog_steps = self.config.crowd_emitter.emitter.leapfrog_steps
            min_origin_goal_distance = getattr(self.config.crowd_emitter.emitter, 'min_origin_goal_distance', 0.5)
            seed = np.random.randint(0, 100000)
            gen_crowd_sampled = self.CrowdES_emitter_pipeline(
                batch_size=1,
                condition=condition,
                num_crowd=num_sampled,
                device=self.device,
                seed=seed,
                num_inference_steps=diffusion_steps,
                leapfrog_steps=leapfrog_steps,
                min_origin_goal_distance=min_origin_goal_distance,
            )
            gen_crowd_sampled = gen_crowd_sampled['crowd_emission'].detach().cpu().numpy()[0]

            # Convert normalized coordinates to pixels
            gen_crowd_sampled[:, [3, 5]] *= self.scene_size[1]  # x*w
            gen_crowd_sampled[:, [4, 6]] *= self.scene_size[0]  # y*h

            # Move to nearest walkable points
            origin_xy_sampled = gen_crowd_sampled[:, [3, 4]]
            goal_xy_sampled = gen_crowd_sampled[:, [5, 6]]
            origin_xy_sampled = batched_nearest_nonzero_idx_kdtree(self.kdtree, origin_xy_sampled)
            goal_xy_sampled = batched_nearest_nonzero_idx_kdtree(self.kdtree, goal_xy_sampled)
            gen_crowd_sampled[:, [3, 4]] = origin_xy_sampled
            gen_crowd_sampled[:, [5, 6]] = goal_xy_sampled
            gen_crowd_sampled[:, 1] = np.clip(
                np.round(gen_crowd_sampled[:, 1] * self.window_frame),
                0, self.window_frame - 1,
            )
        else:
            gen_crowd_sampled = np.empty((0, 7))
        # Update the scenario
        # Track newly created agent IDs and metadata for frame-local grouping
        newly_spawned_agent_ids = []
        agent_spawn_metadata = []  # Store (agent_id, scenario_type, scenario_idx, group_local_id)
        # Create deterministic agents (queue/gathering)
        for det_info in deterministic_agents:
            self.last_agent_id += 1
            agent_id = self.last_agent_id
            self.new_agent_ids.append(agent_id)
            newly_spawned_agent_ids.append(agent_id)

            origin_xy = det_info['origin_xy']
            goal_xy = det_info['goal_xy']

            # Default parameters for deterministic agents
            agent_params = {
                'agent_type': 0,  # Pedestrian
                'frame_origin': 0,  # Spawn immediately
                'preferred_speed': 1.0,  # Default walking speed
                'origin_xy': origin_xy,
                'goal_xy': goal_xy,
            }
            agent_params['frame_global'] = self.current_frame + agent_params['frame_origin']
            self.agent_parameter[agent_id] = agent_params
            self.agent_trajectory[agent_id] = np.array([origin_xy])  # Start at spawn position
            self.agent_previous_behavior_states[agent_id] = np.zeros(self.CrowdES_simulator.config.latent_dim)

            # Crosswalk agents move; queue/gathering agents are stationary
            is_moving = det_info.get('is_moving', False)
            self.agent_is_queue[agent_id] = (det_info['scenario_type'] == 'queue')
            self.agent_is_stopped[agent_id] = not is_moving
            self.agent_queue_id[agent_id] = det_info['scenario_idx'] if det_info['scenario_type'] == 'queue' else None
            self.agent_is_crosswalk[agent_id] = (det_info['scenario_type'] == 'crosswalk' and is_moving)

            # Store metadata for group assignment
            # Crosswalk agents: group by (crosswalk, idx, flow_direction)
            flow_dir = det_info.get('flow_direction', None)
            agent_spawn_metadata.append({
                'agent_id': agent_id,
                'scenario_type': det_info['scenario_type'],
                'scenario_idx': det_info['scenario_idx'],  # Always integer
                'flow_direction': flow_dir,
                'group_local_id': det_info['group_local_id'],
            })


        # Create sampled agents (cohesive candidates)
        for j in range(num_sampled):
            self.last_agent_id += 1
            agent_id = self.last_agent_id
            self.new_agent_ids.append(agent_id)
            newly_spawned_agent_ids.append(agent_id)
            
            data = gen_crowd_sampled[j]
            agent_params = {
                'agent_type': int(data[0]),
                'frame_origin': int(data[1]),
                'preferred_speed': data[2],
                'origin_xy': data[3:5],
                'goal_xy': data[5:7],
            }
            agent_params['frame_global'] = self.current_frame + agent_params['frame_origin']
            self.agent_parameter[agent_id] = agent_params
            self.agent_trajectory[agent_id] = np.empty((0, 2))
            self.agent_previous_behavior_states[agent_id] = np.zeros(self.CrowdES_simulator.config.latent_dim)

            # Controllability
            if CONTROL_AGENT_TYPE is not None:
                self.agent_parameter[agent_id]['agent_type'] = CONTROL_AGENT_TYPE
            if CONTROL_WALKING_PACE is not None:
                self.agent_parameter[agent_id]['preferred_speed'] = CONTROL_WALKING_PACE

            # Not part of queue/gathering (moving agents)
            self.agent_is_queue[agent_id] = False
            self.agent_is_stopped[agent_id] = False
            self.agent_queue_id[agent_id] = None
            self.agent_is_crosswalk[agent_id] = False

            # Store metadata (no scenario type for sampled agents)
            agent_spawn_metadata.append({
                'agent_id': agent_id,
                'scenario_type': None,  # Cohesive candidate
                'scenario_idx': None,
                'group_local_id': None,
            })

        self.statistics_added = num_agents_to_emit

        # Frame-local group detection (newly spawned agents only)
        # ====================================================================================
        # METADATA-BASED GROUP ASSIGNMENT (Deterministic) + CLUSTERING (Cohesive)
        # ====================================================================================
        if self.enable_group_behavior and num_agents_to_emit >= 2:
            starting_group_id = len(self.group_graph['groups'])
            next_global_group_id = starting_group_id

            # Step 1: Assign deterministic agents to groups based on metadata
            # Group by (scenario_type, scenario_idx)
            deterministic_groups = {}  # {(scenario_type, scenario_idx): [agent_ids]}
            sampled_agent_ids = []

            for metadata in agent_spawn_metadata:
                agent_id = metadata['agent_id']
                scenario_type = metadata['scenario_type']
                scenario_idx = metadata['scenario_idx']
                flow_dir = metadata.get('flow_direction', None)

                if scenario_type is not None:  # Deterministic agent
                    key = (scenario_type, scenario_idx, flow_dir) if scenario_type == 'crosswalk' else (scenario_type, scenario_idx)
                    if key not in deterministic_groups:
                        deterministic_groups[key] = []
                    deterministic_groups[key].append(agent_id)
                else:  # Sampled agent (cohesive candidate)
                    sampled_agent_ids.append(agent_id)

            # Create groups for deterministic agents
            for key, member_ids in deterministic_groups.items():
                # key is (scenario_type, scenario_idx) or (scenario_type, scenario_idx, flow_dir)
                scenario_type = key[0]
                scenario_idx = key[1]
                flow_dir = key[2] if len(key) == 3 else None

                if len(member_ids) < 2:
                    continue  # Single agent - no group

                # Create adjacency matrix using type-specific topology
                topology = TOPOLOGY_FOR_GROUP_TYPE.get(scenario_type, 'clique')
                adjacency_matrix = build_adjacency(topology, len(member_ids))

                global_group = {
                    'group_id': next_global_group_id,
                    'member_ids': member_ids,
                    'group_type': scenario_type,  # 'queue', 'gathering', or 'crosswalk'
                    'cohesion_strength': 1.0,
                    'adjacency_matrix': adjacency_matrix,
                    'scenario_idx': scenario_idx,
                    'flow_direction': flow_dir,
                }
                self.group_graph['groups'].append(global_group)
                self.group_graph['scenario_types'][next_global_group_id] = scenario_type  # Track scenario type

                for aid in member_ids:
                    self.group_graph['agent_to_group'][aid] = next_global_group_id

                next_global_group_id += 1

            # Step 2: Clustering for sampled agents (cohesive groups)
            cohesive_enabled = self._load_cohesive_config().get('enabled', True)
            if cohesive_enabled and len(sampled_agent_ids) >= 2:
                sampled_profiles = []
                for aid in sampled_agent_ids:
                    params = self.agent_parameter[aid]
                    profile = [
                        params['agent_type'],
                        params['frame_origin'],
                        params['preferred_speed'],
                        params['origin_xy'][0],
                        params['origin_xy'][1],
                        params['goal_xy'][0],
                        params['goal_xy'][1]
                    ]
                    sampled_profiles.append(profile)
                sampled_profiles = np.array(sampled_profiles)

                # Use GroupGenerator for cohesive clustering
                temp_group_graph = self.group_generator.generate(sampled_profiles, sampled_agent_ids)

                # Merge cohesive groups into global graph
                for local_group in temp_group_graph['groups']:
                    global_group = {
                        'group_id': next_global_group_id,
                        'member_ids': local_group['member_ids'],
                        'group_type': 'cohesive',
                        'cohesion_strength': local_group['cohesion_strength'],
                        'adjacency_matrix': local_group['adjacency_matrix'],
                    }
                    self.group_graph['groups'].append(global_group)
                    self.group_graph['scenario_types'][next_global_group_id] = 'cohesive'  # Track scenario type

                    for aid in local_group['member_ids']:
                        self.group_graph['agent_to_group'][aid] = next_global_group_id

                    next_global_group_id += 1

            # Step 3: Align goals for cohesive group members (shared goal)
            # Pass first_new_group_id so we only process groups created in THIS window.
            # Re-processing previous-window groups would overwrite their frame_global
            # to self.current_frame + shared_frame, causing them to spawn very late.
            self._align_cohesive_goals(first_new_group_id=starting_group_id)

        # Visualize the crowd emitter output
        if SAVE_IMAGE_FOR_DEBUG:
            crowd_emitter_result = self.appearance_density_map.copy()
            crowd_emitter_result = (crowd_emitter_result * 255).astype(np.uint8)
            crowd_emitter_result = np.stack([crowd_emitter_result] * 3, axis=-1)

            pred_crowd_color = np.random.rand(1000, 3) * 255

            # Draw deterministic agents (red)
            for det_agent in deterministic_agents:
                origin = det_agent['origin_xy'].astype(np.int32)
                goal = det_agent['goal_xy'].astype(np.int32)
                cv2.circle(crowd_emitter_result, tuple(origin), 8, (0, 0, 255), -1)  # Red filled
                cv2.circle(crowd_emitter_result, tuple(goal), 7, (0, 0, 255), 2)    # Red outline

            # Draw sampled agents (blue)
            for j in range(num_sampled):
                origin = gen_crowd_sampled[j, [3, 4]].astype(np.int32)
                goal = gen_crowd_sampled[j, [5, 6]].astype(np.int32)
                cv2.circle(crowd_emitter_result, tuple(origin), 8, (255, 0, 0), -1)  # Blue filled
                cv2.circle(crowd_emitter_result, tuple(goal), 7, (255, 0, 0), 2)    # Blue outline

            cv2.putText(crowd_emitter_result, f"D:{num_deterministic} S:{num_sampled}",
                       (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            os.makedirs('temp', exist_ok=True)
            crowd_emitter_result = Image.fromarray(crowd_emitter_result)
            crowd_emitter_result.save(f'temp/crowd_emitter_result_{self.current_frame}.png')

    def process_surface_emitter(self):
        # Step 1: Plan the number of agents to emit
        max_num_agents_pad = self.config.crowd_emitter.emitter.max_num_agents_pad
        num_agents = np.random.choice(len(self.population_probability), p=self.population_probability)
        num_agents = int(num_agents * self.config.crowd_emitter.emitter_pre.population_multiplier)
        num_agents_to_emit = num_agents - len(self.agent_ids_in_current_scene)
        num_agents_to_emit = min(num_agents_to_emit, max_num_agents_pad)

        # For initialization, cut down the number of agents in half
        if self.current_frame < 0:
            num_agents_to_emit = math.ceil(num_agents_to_emit / 2)

        # Skip if too few agents to emit (surface emitter doesn't have nested tensor issue, but keep consistent)
        if num_agents_to_emit < 2:
            self.statistics_added = 0
            return self.agent_ids_in_current_scene
        
        # Step 2: Populate the scene with surface emitter
        gen_crowd = np.random.randn(num_agents_to_emit, 7)
        norm_mean = self.CrowdES_emitter.norm_mean.detach().cpu().numpy()
        norm_std = self.CrowdES_emitter.norm_std.detach().cpu().numpy()
        gen_crowd = gen_crowd * norm_std + norm_mean
        
        # Generated agent parameters: [agent_type, frame_origin, preferred_speed, origin_x, origin_y, goal_x, goal_y]       
        gen_crowd[:, 0] = np.round(np.clip(gen_crowd[:, 0], 0, 2))  # round agent_type to 0, 1 or 2
        gen_crowd[:, 1] = np.random.randint(0, self.window_frame, size=(num_agents_to_emit,))
        gen_crowd[:, 2] = np.clip(gen_crowd[:, 2], 0, float('inf'))
        gen_crowd[:, [3, 4]] = sampling_xy_pos(torch.FloatTensor(self.walkable).unsqueeze(dim=0), num_agents_to_emit).squeeze(dim=0).numpy()
        gen_crowd[:, [5, 6]] = sampling_xy_pos(torch.FloatTensor(self.walkable).unsqueeze(dim=0), num_agents_to_emit).squeeze(dim=0).numpy()

        # Update the scenario
        for j in range(num_agents_to_emit):
            self.last_agent_id += 1
            agent_id = self.last_agent_id
            self.new_agent_ids.append(agent_id)
            data = gen_crowd[j]
            agent_params = {'agent_type': int(data[0]), 'frame_origin': int(data[1]), 'preferred_speed': data[2], 'origin_xy': data[3:5], 'goal_xy': data[5:7]}
            agent_params['frame_global'] = self.current_frame + agent_params['frame_origin']
            self.agent_parameter[agent_id] = agent_params
            self.agent_trajectory[agent_id] = np.empty((0, 2))
            self.agent_previous_behavior_states[agent_id] = np.zeros(self.CrowdES_simulator.config.latent_dim)

        self.statistics_added = num_agents_to_emit

    def process_simulator(self):
        # Step 1: Repeat simulation for (self.window_frame / future_length) recurrent times
        T_hist = self.config.crowd_simulator.simulator.history_length
        T_fut = self.traj_fut_frame
        interaction_range = self.config.crowd_simulator.simulator.interaction_range
        interaction_max_num_agents = self.config.crowd_simulator.simulator.interaction_max_num_agents
        control_time_offset = self.config.crowd_simulator.simulator.control_time_offset
        
        before_num_agents = len(self.agent_ids_in_current_scene)

        for recurrent in range(self.window_frame // self.traj_fut_frame):
            # Step 2. Add new agents into current scene
            for agent_id in self.new_agent_ids:
                frame_origin = self.agent_parameter[agent_id]['frame_origin']
                if T_fut * recurrent <= frame_origin < T_fut * (recurrent + 1):
                    self.agent_ids_in_current_scene.append(agent_id)
                    
            if len(self.agent_ids_in_current_scene) == 0:
                continue

            # Step 2.5: Handle stopped queue agents (extend trajectory with last position)
            stopped_agents = []
            for agent_id in self.agent_ids_in_current_scene:
                if self.agent_is_stopped.get(agent_id, False):
                    # Queue agent is stopped - extend trajectory with last position (velocity=0)
                    if len(self.agent_trajectory[agent_id]) > 0:
                        last_position = self.agent_trajectory[agent_id][-1]
                    else:
                        last_position = np.array(self.agent_parameter[agent_id]['origin_xy'])

                    # Add T_fut frames of stationary position
                    for _ in range(T_fut):
                        if len(self.agent_trajectory[agent_id]) == 0:
                            self.agent_trajectory[agent_id] = last_position[None, :]
                        else:
                            self.agent_trajectory[agent_id] = np.vstack([self.agent_trajectory[agent_id], last_position])
                    stopped_agents.append(agent_id)

            # Filter out stopped agents from simulator input
            active_agent_ids = [aid for aid in self.agent_ids_in_current_scene if aid not in stopped_agents]

            if len(active_agent_ids) == 0:
                continue

            # 3. Generate the future trajectory of length future_length
            B = len(active_agent_ids)
            traj_hist = np.zeros((B, T_hist, 2))
            goal = np.zeros((B, 2))
            attr = np.zeros((B, 2))

            for idx, agent_id in enumerate(active_agent_ids):
                agent_trajectory = self.agent_trajectory[agent_id]
                if len(agent_trajectory) < T_hist:
                    origin_xy = self.agent_parameter[agent_id]['origin_xy']
                    traj_hist[idx] = np.array(origin_xy)[None, :]
                    if len(agent_trajectory) > 0:
                        traj_hist[idx, -len(agent_trajectory):] = np.array(agent_trajectory)
                else:
                    traj_hist[idx] = np.array(agent_trajectory[-T_hist:])

                goal[idx] = np.array(self.agent_parameter[agent_id]['goal_xy'])
                attr[idx] = np.array([self.agent_parameter[agent_id]['agent_type'], self.agent_parameter[agent_id]['preferred_speed']])

            # Prepare the control points
            start_points = traj_hist[:, -1]
            finish_points = goal
            distances = attr[:, 1] * control_time_offset
            control_points_meter = get_control_point(self.navmesh['vertices'], self.navmesh['polygons'], start_points, finish_points, self.H, distances)
            control_points_meter = np.array(control_points_meter) # Note: control points are already in meter coordinates

            traj_hist_meter = image2world(traj_hist, self.H)
            goal_meter = image2world(goal, self.H)

            # Aggregate neighborhood trajectories (GROUP-AWARE)
            startpoint_meter = traj_hist_meter[:, -1]
            distance_matrix = np.linalg.norm(startpoint_meter[None, :, :] - startpoint_meter[:, None, :], axis=2)

            neighbor_index = []
            for idx, agent_id in enumerate(active_agent_ids):
                # Priority 1: Same group members (regardless of distance)
                group_neighbors = []
                if self.enable_group_behavior and agent_id in self.group_graph['agent_to_group']:
                    group_id = self.group_graph['agent_to_group'][agent_id]
                    # Find other group members in current scene (only among active agents)
                    for other_idx, other_agent_id in enumerate(active_agent_ids):
                        if other_idx != idx and other_agent_id in self.group_graph['agent_to_group']:
                            if self.group_graph['agent_to_group'][other_agent_id] == group_id:
                                group_neighbors.append(other_idx)

                    # Sort group members by distance and limit to interaction_max_num_agents
                    if len(group_neighbors) > 0:
                        group_distances = distance_matrix[idx, group_neighbors]
                        sorted_indices = np.argsort(group_distances)
                        group_neighbors = [group_neighbors[i] for i in sorted_indices[:interaction_max_num_agents]]

                # Priority 2: Spatial neighbors (distance-based)
                spatial_neighbors = np.argsort(distance_matrix[idx])[1:interaction_max_num_agents+1]
                spatial_mask = distance_matrix[idx, spatial_neighbors] < interaction_range
                spatial_neighbors = spatial_neighbors[spatial_mask].tolist()

                # Merge: Group members first, then spatial neighbors (no duplicates)
                combined_neighbors = group_neighbors.copy()
                for n in spatial_neighbors:
                    if n not in combined_neighbors:
                        combined_neighbors.append(n)
                        if len(combined_neighbors) >= interaction_max_num_agents:
                            break

                if len(combined_neighbors) > interaction_max_num_agents:
                    combined_neighbors = combined_neighbors[:interaction_max_num_agents]

                neighbor_index.append(combined_neighbors)

            # Fill the neighbor matrix with traj_hist_meter
            neighbor_meter = np.zeros((B, interaction_max_num_agents, T_hist, 2))
            group_neighbor_count = 0
            total_neighbor_count = 0
            for idx, neighbor_idx in enumerate(neighbor_index):
                if len(neighbor_idx) > 0:
                    # Safety: Enforce max limit to prevent broadcasting errors
                    if len(neighbor_idx) > interaction_max_num_agents:
                        neighbor_idx = neighbor_idx[:interaction_max_num_agents]
                    neighbor_meter[idx, :len(neighbor_idx)] = traj_hist_meter[neighbor_idx]
                    total_neighbor_count += len(neighbor_idx)

                    # Count how many are group members (use active_agent_ids)
                    agent_id = active_agent_ids[idx]
                    if self.enable_group_behavior and agent_id in self.group_graph['agent_to_group']:
                        group_id = self.group_graph['agent_to_group'][agent_id]
                        for n_idx in neighbor_idx:
                            n_agent_id = active_agent_ids[n_idx]
                            if n_agent_id in self.group_graph['agent_to_group']:
                                if self.group_graph['agent_to_group'][n_agent_id] == group_id:
                                    group_neighbor_count += 1


            # Normalize with startpoint_meter
            traj_hist_meter_norm = traj_hist_meter - startpoint_meter[:, None, :]
            goal_meter_norm = goal_meter - startpoint_meter
            control_points_meter_norm = control_points_meter - startpoint_meter
            neighbor_meter_norm = neighbor_meter - startpoint_meter[:, None, None, :]

            # Build environmental data
            _, h, w = self.environment_s.shape
            env_pointer_meter = self.env_base_pointer[None, :] + startpoint_meter[:, None, None, [1, 0]]  # (B, 64, 64, 2) + (B, None, None, (2=xy -> hw in meter))
            env_pointer_pixel = world2image(env_pointer_meter.transpose(0, 2, 1, 3), self.H).transpose(0, 2, 1, 3)  # hw -> wh
            env_pointer_pixel = env_pointer_pixel.round().astype(int)
            env_pointer_pixel = np.clip(env_pointer_pixel, 0, [h-1, w-1])
            env_data_crop = self.environment_s[:, env_pointer_pixel[..., 0], env_pointer_pixel[..., 1]]
            env_data_crop = env_data_crop.permute(1, 0, 2, 3) if len(env_data_crop.shape) == 4 else env_data_crop
            
            # Aggregate previous behavior states (only for active agents)
            previous_states = np.zeros((B, self.CrowdES_simulator.config.latent_dim))
            for idx, agent_id in enumerate(active_agent_ids):
                previous_states[idx] = self.agent_previous_behavior_states[agent_id]

            # Convert to torch tensors
            traj_hist_meter_norm = torch.FloatTensor(traj_hist_meter_norm).to(self.device)
            goal_meter_norm = torch.FloatTensor(goal_meter_norm).to(self.device)
            attr = torch.FloatTensor(attr).to(self.device)
            control_points_meter_norm = torch.FloatTensor(control_points_meter_norm).to(self.device)
            neighbor_meter_norm = torch.FloatTensor(neighbor_meter_norm).to(self.device)
            env_data_crop = env_data_crop.to(self.device)
            previous_states = torch.FloatTensor(previous_states).to(self.device)

            # Predict the future trajectories of the crowds
            outputs = self.CrowdES_simulator(traj_hist_meter_norm, None, goal_meter_norm, attr, control_points_meter_norm, neighbor_meter_norm, env_data_crop, sampling=True, previous_state=previous_states, tmp=self.config.crowd_simulator.simulator.latent_temperature, alpha=self.config.crowd_simulator.simulator.latent_mixup_alpha)
            behavior_states = outputs.states.detach().cpu().numpy()
            traj_fut_meter_norm = outputs.preds.detach().cpu().numpy()
            traj_fut_meter_norm = traj_fut_meter_norm[:, :T_fut]  # Make sure
            assert traj_fut_meter_norm.shape[1] == T_fut  # Make sure
            
            traj_fut_meter = traj_fut_meter_norm + startpoint_meter[:, None, :]
            traj_fut = world2image(traj_fut_meter, self.H)  # B, T_fut, 2

            # Shift the trajectory coordinates into the walkable area
            traj_fut = batched_nearest_nonzero_idx_kdtree(self.kdtree, traj_fut)

            # Apply cohesion force to keep cohesive group members together
            traj_fut = self._apply_cohesion_force(traj_fut, active_agent_ids)

            # Step 4: Update the scene (only active agents that went through simulator)
            for idx, agent_id in list(enumerate(active_agent_ids)):
                # Update the behavior states
                self.agent_previous_behavior_states[agent_id] = behavior_states[idx]

                if agent_id in self.new_agent_ids:
                    # If the agent is new in the scene, trim the trajectory with frame_origin
                    frame_origin = self.agent_parameter[agent_id]['frame_origin']
                    frame_origin_norm = min(max(0, frame_origin - T_fut * recurrent), T_fut-1)  # Make sure
                    traj_agent = traj_fut[idx, :T_fut-frame_origin_norm]  # from start
                    # traj_agent = traj_fut[idx, frame_origin_norm:]  # from end
                    self.new_agent_ids.remove(agent_id)
                else:
                    traj_agent = np.concatenate([self.agent_trajectory[agent_id], traj_fut[idx]], axis=0)

                # Step 5: Pop the agent if the agent out of the scene or the agent reach the goal
                traj_agent_sceneoutmask = ((traj_fut[idx] <= 0) | (traj_fut[idx] >= np.array(self.scene_size)[[1, 0]] - 1)).any(axis=1)
                traj_agent_goal = np.linalg.norm(traj_fut_meter[idx] - goal_meter[idx][None, :], axis=1)
                traj_agent_goalmask = traj_agent_goal < 0.5  # 0.5 meter threshold to consider the agent reached the goal

                if traj_agent_sceneoutmask.any():
                    cut_pivot = np.where(traj_agent_sceneoutmask)[0][0]
                    cut_pivot_from_end = T_fut - cut_pivot  # It should not be 0 or negative
                    traj_agent = traj_agent[:-cut_pivot_from_end]
                    self.agent_ids_in_current_scene.remove(agent_id)

                elif traj_agent_goalmask.any():
                    # Queue agents: Stop at goal instead of removing
                    if self.agent_is_queue.get(agent_id, False):
                        # Mark as stopped and keep trajectory up to goal
                        if not self.agent_is_stopped[agent_id]:
                            cut_pivot = np.argmin(traj_agent_goal)
                            cut_pivot_from_end = T_fut - cut_pivot + 1
                            traj_agent = traj_agent[:-cut_pivot_from_end] if cut_pivot_from_end > 0 else traj_agent
                            self.agent_is_stopped[agent_id] = True
                            print(f"[Queue Stop] Agent {agent_id} reached queue point at frame {self.current_frame}")
                        # If already stopped, trajectory will be extended with last position in Step 6
                    else:
                        # Regular agents: Remove when reaching goal
                        cut_pivot = np.argmin(traj_agent_goal)
                        cut_pivot_from_end = T_fut - cut_pivot + 1
                        traj_agent = traj_agent[:-cut_pivot_from_end] if cut_pivot_from_end > 0 else traj_agent
                        self.agent_ids_in_current_scene.remove(agent_id)

                # Update the trajectory
                self.agent_trajectory[agent_id] = traj_agent

        after_num_agents = len(self.agent_ids_in_current_scene)
        self.statistics_dropped = self.statistics_added + before_num_agents - after_num_agents

    def process_orca_simulator(self):
        # Step 1: Repeat simulation for window_frame recurrent times
        self.traj_fut_frame = 1  # for ORCA, we will simulate each frame one by one
        before_num_agents = len(self.agent_ids_in_current_scene)

        for recurrent in range(self.window_frame):
            # Step 2. Add new agents into current scene
            for agent_id in self.new_agent_ids.copy():
                frame_origin = self.agent_parameter[agent_id]['frame_origin']
                if recurrent <= frame_origin < recurrent + 1:
                    self.agent_ids_in_current_scene.append(agent_id)
                    self.new_agent_ids.remove(agent_id)

                    # Set agent radius based on agent type (0: pedestrian, 1: bicycle, 2: vehicle)
                    agent_type = self.agent_parameter[agent_id]['agent_type']
                    agent_radius = 0.2 * 2
                    agent_radius = 0.5 * 2 if agent_type == 1 else agent_radius
                    agent_radius = 1.0 * 2 if agent_type == 2 else agent_radius

                    # Add agent to the ORCA simulator
                    origin_xy = image2world(self.agent_parameter[agent_id]['origin_xy'], self.H)
                    goal_xy = image2world(self.agent_parameter[agent_id]['goal_xy'], self.H)
                    preferred_speed = self.agent_parameter[agent_id]['preferred_speed']      
                    sim_agent_id = self.pf.add_agent((origin_xy[0], 0, origin_xy[1]), agent_radius, preferred_speed)

                    if sim_agent_id == -1:
                        self.agent_ids_in_current_scene.remove(agent_id)
                        continue
                    
                    try:
                        self.pf.set_agent_destination(sim_agent_id, (goal_xy[0], 0, goal_xy[1]))
                    except:
                        print(f'ORCA error at frame {self.current_frame}')  # ORCA bug that cannot be fixed
                        break
                    self.sim_agent_id_to_agent_id[sim_agent_id] = agent_id

                    # add agent trajectories
                    origin_xy_image = self.agent_parameter[agent_id]['origin_xy']
                    self.agent_trajectory[agent_id] = np.array(origin_xy_image)[None, :]  # Add start position
            
            # Step 3: Simulate the future coordinates of the crowds
            if len(self.agent_ids_in_current_scene) == 0:
                continue
            
            try:
                self.pf.update(delta_time=1/self.simulator_fps)
            except:
                print(f'ORCA error at frame {self.current_frame}')  # ORCA bug that cannot be fixed
                break
            
            # Step 4: Update the scene
            for sim_agent_id in self.pf._agents_id:
                agent_id = self.sim_agent_id_to_agent_id[sim_agent_id]
                new_pos = self.pf.get_agent_position(sim_agent_id)
                new_pos_image = world2image(np.array(new_pos), self.H)
                before_trajectory = self.agent_trajectory[agent_id]
                self.agent_trajectory[agent_id] = np.concatenate([before_trajectory, new_pos_image[None, :]], axis=0)
            
            # Step 5: Pop the agent if the agent out of the scene or the agent reach the goal
            for sim_agent_id in self.pf._agents_id:
                agent_id = self.sim_agent_id_to_agent_id[sim_agent_id]
                goal_xy = image2world(self.agent_parameter[agent_id]['goal_xy'], self.H)
                agent_pos = self.pf.get_agent_position(sim_agent_id)
                agent_pos_image = world2image(np.array(agent_pos), self.H)
                static_threshold = 10

                # Check if the agent arrived
                if np.linalg.norm(agent_pos - goal_xy) < 0.1:
                    self.agent_ids_in_current_scene.remove(agent_id)
                    self.pf.delete_agent(sim_agent_id)

                # Check if the agent out of the scene
                elif (agent_pos_image < 0).any() or (agent_pos_image >= np.array(self.scene_size)[[1, 0]]).any():
                    self.agent_ids_in_current_scene.remove(agent_id)
                    self.pf.delete_agent(sim_agent_id)

                # Compensate RVO simulation bug by checking if the agent stays more than 10 frames
                elif len(self.agent_trajectory[agent_id]) > static_threshold + 1:
                    traj = self.agent_trajectory[agent_id][-(static_threshold+1):]
                    traj_diff = np.linalg.norm(traj[1:] - traj[:-1], axis=1)
                    if traj_diff.mean() < 0.01:
                        self.agent_ids_in_current_scene.remove(agent_id)
                        self.pf.delete_agent(sim_agent_id)

        after_num_agents = len(self.agent_ids_in_current_scene)
        self.statistics_dropped = self.statistics_added + before_num_agents - after_num_agents

    def postprocess_trajectory(self):
        """Post-process the trajectory by interpolating and smoothing the trajectory."""

        for agent_id in tqdm(self.agent_trajectory.keys(), desc='Post-processing the trajectory'):
            agent_id, agent_trajectory = self.postprocess_trajectory_single(agent_id)
            self.agent_trajectory[agent_id] = agent_trajectory

    def postprocess_trajectory_single(self, agent_id):
        """Smooth the trajectory of a single agent."""
        agent_trajectory = self.agent_trajectory[agent_id]
        if len(agent_trajectory) < 2:
            return agent_id, agent_trajectory

        agent_trajectory_meter = image2world(agent_trajectory, self.H)
        agent_origin_meter = agent_trajectory_meter[0]
        agent_trajectory_meter_norm = agent_trajectory_meter - agent_origin_meter
        
        kf = KalmanModel(dt=1, n_dim=2, n_iter=1)
        smoothed_pos, smoothed_vel = kf.smooth(agent_trajectory_meter_norm)
        agent_trajectory_meter_norm = smoothed_pos

        agent_trajectory_meter = agent_trajectory_meter_norm + agent_origin_meter
        agent_trajectory = world2image(agent_trajectory_meter, self.H)

        return agent_id, agent_trajectory
    
    def densify_scenario(self, interpolation_method='linear'):
        """Densify the scenario by interpolating the trajectory."""

        upsample = self.dataset_fps // self.simulator_fps
        for agent_id in self.agent_parameter.keys():
            agent_trajectory = self.agent_trajectory[agent_id]
            agent_type = self.agent_parameter[agent_id]['agent_type']
            frame_origin = self.agent_parameter[agent_id]['frame_global']

            # Crosswalk agents use 0-based frames so CSV output starts at frame 0
            if self.agent_is_crosswalk.get(agent_id, False):
                frame_origin = 0

            # Get group_id for this agent (-1 if not in any group or group behavior disabled)
            if self.enable_group_behavior:
                group_id = self.group_graph['agent_to_group'].get(agent_id, -1)
            else:
                group_id = -1

            if len(agent_trajectory) < 2:
                continue

            # Interpolate the trajectory
            x = agent_trajectory[:, 0]
            y = agent_trajectory[:, 1]
            frame = np.arange(frame_origin, len(agent_trajectory) + frame_origin) * upsample

            if interpolation_method == 'pchip':
                pchip_x = PchipInterpolator(frame, x)
                pchip_y = PchipInterpolator(frame, y)
                frame_interp = np.arange(frame_origin * upsample, (len(agent_trajectory) + frame_origin) * upsample)
                x_interp = pchip_x(frame_interp)
                y_interp = pchip_y(frame_interp)
            elif interpolation_method == 'linear':
                frame_interp = np.arange(frame_origin * upsample, (len(agent_trajectory) + frame_origin) * upsample)
                x_interp = np.interp(frame_interp, frame, x)
                y_interp = np.interp(frame_interp, frame, y)
            else:
                raise NotImplementedError(f'Interpolation method {interpolation_method} is not implemented.')

            # Trim the trajectory to fit within the scenario length
            for i in range(len(frame_interp)):
                if 0 <= frame_interp[i] < self.scenario_len:
                    self.scenario.append([agent_id, agent_type, frame_interp[i], x_interp[i], y_interp[i], group_id])

    def generate(self, scenario_len, seed=None):
        """Generate crowd trajectories from an image and a segmentation map.

        Params:
            img (np.array): An image.
            seg (np.array): A segmentation map.

        Returns:
            list: A list of trajectories.
        """
        
        # Reproducibility
        reproducibility_settings(seed=seed) if seed is not None else None

        self.scenario = []
        self.scenario_len = scenario_len
        self.last_agent_id = -1
        self.agent_parameter = {}
        self.agent_trajectory = {}
        self.agent_ids_in_current_scene = []
        self.new_agent_ids = []
        self.agent_previous_behavior_states = {}

        max_frame = int(np.ceil(scenario_len / self.dataset_fps * self.simulator_fps))

        # Preprocess the scene layouts for crowd emitter
        scene_layout = self.process_emitter_pre()
        appearance_density_map, population_density_map, population_probability = scene_layout
        
        # Generate crowd trajectories
        start_frames = list(range(-self.window_frame, max_frame, self.window_frame))
        import time as _time
        pbar = tqdm(start_frames, desc='Generating crowd trajectories')
        for frame_idx in start_frames:
            self.current_frame = frame_idx

            # Emitter phase
            pbar.set_description('Crowd Emitter  ')
            _t0 = _time.perf_counter()
            if self.config.crowd_emitter.type == 'CrowdES':
                self.process_emitter()
            elif self.config.crowd_emitter.type == 'surface':
                self.process_surface_emitter()
            else:
                raise NotImplementedError(f'Emitter type {self.config.crowd_emitter.type} is not implemented.')
            _emitter_ms = (_time.perf_counter() - _t0) * 1000

            # Simulator phase
            pbar.set_description('Crowd Simulator')
            _t0 = _time.perf_counter()
            if self.config.crowd_simulator.type == 'CrowdES':
                self.process_simulator()
            elif self.config.crowd_simulator.type == 'ORCA':
                self.process_orca_simulator()
            else:
                raise NotImplementedError(f'Simulator type {self.config.crowd_simulator.type} is not implemented.')
            _sim_ms = (_time.perf_counter() - _t0) * 1000

            pbar.set_postfix(population=len(self.agent_ids_in_current_scene), added=self.statistics_added, dropped=self.statistics_dropped)
            pbar.update()

        pbar.close()

        # Post-process trajectory
        self.postprocess_trajectory()

        # NOTE: Group detection already done in process_emitter() (frame-local)
        # No need to re-detect here, just assign group_id for CSV export

        # Store final group_id in agent_parameter for CSV export
        if self.enable_group_behavior:
            for agent_id in self.agent_parameter.keys():
                group_id = self.group_graph['agent_to_group'].get(agent_id, -1)
                self.agent_parameter[agent_id]['group_id'] = group_id

            num_groups = len(self.group_graph['groups'])
            num_grouped = len(self.group_graph['agent_to_group'])
            print(f"\n[Group Summary] {num_groups} groups, {num_grouped}/{len(self.agent_parameter)} agents grouped")

        # Densify scenario
        self.densify_scenario()

        # Convert to pandas dataframe
        column_names = ['agent_id', 'agent_type', 'frame', 'x', 'y', 'group_id']
        self.scenario = pd.DataFrame(self.scenario, columns=column_names)
        self.scenario = self.scenario.astype({'agent_id': int, 'agent_type': int, 'frame': int, 'x': float, 'y': float, 'group_id': int})

        # Add group_type column from group_graph
        agent_to_formation = {}
        for group in self.group_graph.get('groups', []):
            formation = group.get('group_type', 'cohesive')
            for member_id in group.get('member_ids', []):
                agent_to_formation[member_id] = formation
        self.scenario['group_type'] = self.scenario['agent_id'].map(agent_to_formation).fillna('none')

        # Reorganize agent_id starting from 0
        agent_id2new_id = {agent_id: idx for idx, agent_id in enumerate(self.scenario['agent_id'].unique())}
        self.scenario['agent_id'] = self.scenario['agent_id'].map(agent_id2new_id)

        # Remap group_graph member_ids to match remapped agent IDs so JSON export is consistent with CSV
        if self.enable_group_behavior:
            for group in self.group_graph.get('groups', []):
                group['member_ids'] = [agent_id2new_id[mid] for mid in group['member_ids']
                                       if mid in agent_id2new_id]
            self.group_graph['agent_to_group'] = {
                agent_id2new_id[orig]: gid
                for orig, gid in self.group_graph['agent_to_group'].items()
                if orig in agent_id2new_id
            }

        return self.scenario