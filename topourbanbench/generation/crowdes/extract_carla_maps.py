import sys
import os

# Add the crowdes directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Per-town camera settings. All towns use identical spectator_z for consistent pixel_scale.
# spectator_z : camera height above ground (meters)
# roi         : [x_min, y_min, x_max, y_max] world-space bounding box to filter sidewalks.
#               Optional 5th element z_max: exclude sidewalks above this height (meters).
#               min/max are auto-sorted so order does not matter.
TOWN_CONFIGS = {
    'town10HD_Opt': {
        'spectator_z': 50.0,
        'roi': [-125.0, -78.07, 119.16, 150.0],
    },
    'town03_Opt': {
        'spectator_z': 50.0,
        # z_max=2.0: exclude elevated sidewalks (z > 2.0) present in some Town03 areas
        'roi': [-72.0, -197.93, 237.0, 2.0, 2.0],   # [x_min, y_min, x_max, y_max, z_max]
    },
    'town05_Opt': {
        'spectator_z': 50.0,
        'roi': [-190.0, -88.03, 94.07, 90.0],  # [x_min, y_min, x_max, y_max]
    },
}

import carla
import queue as queue_module  # Avoid name conflict
import cv2
import numpy as np
import math
import argparse
import json
import yaml
import matplotlib.pyplot as plt # for debug plots
from utils.scenario_generator import SemanticPreprocessor
from utils.unified_scenario_generator import UnifiedScenarioGenerator
from utils.scenario_validator import validate_scenario_points
import time

def save_debug_plot(semantic_vis_rgb, walkable_mask, appearance_map, scenario_density,
                    scenario_metadata, scene_id, save_dir):
    """
    2x2 debug plot:
      [0,0] Semantic Map
      [0,1] Walkable Mask
      [1,0] Appearance Density
      [1,1] Scenario Density + Queue/Gathering/Crosswalk markers
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle(f"Scene Debug: {scene_id}", fontsize=20, fontweight='bold')

    # 1. Semantic Map
    axes[0, 0].imshow(semantic_vis_rgb)
    axes[0, 0].set_title("Semantic Map", fontsize=14)
    axes[0, 0].axis('off')

    # 2. Walkable Mask
    axes[0, 1].imshow(walkable_mask, cmap='gray')
    axes[0, 1].set_title("Walkable Mask", fontsize=14)
    axes[0, 1].axis('off')

    # 3. Appearance Density
    axes[1, 0].imshow(appearance_map, cmap='hot')
    axes[1, 0].set_title("Appearance Density", fontsize=14)
    axes[1, 0].axis('off')

    # 4. Scenario Density + scenario markers overlay
    axes[1, 1].imshow(scenario_density, cmap='hot')
    axes[1, 1].set_title("Scenario Density", fontsize=14)
    axes[1, 1].axis('off')

    # Queue markers (blue squares + direction arrows)
    for q in scenario_metadata.get('queue', []):
        pt = q['goal_point']
        axes[1, 1].scatter(pt[0], pt[1], c='blue', s=150, marker='s',
                           edgecolors='white', linewidths=1.5, zorder=5)
        # Draw queue direction arrow
        d = q.get('direction', [0, 0])
        length = q.get('length_px', 80)
        axes[1, 1].annotate('', xy=(pt[0] + d[0]*length, pt[1] + d[1]*length),
                           xytext=(pt[0], pt[1]),
                           arrowprops=dict(arrowstyle='->', color='cyan', lw=2))

    # Gathering markers (green circles)
    for g in scenario_metadata.get('gathering', []):
        ct = g['center']
        axes[1, 1].scatter(ct[0], ct[1], c='lime', s=150, marker='o',
                           edgecolors='white', linewidths=1.5, zorder=5)
        radius = g.get('radius', 50)
        circle = plt.Circle((ct[0], ct[1]), radius, fill=False,
                            edgecolor='lime', linewidth=1.5, linestyle='--')
        axes[1, 1].add_patch(circle)

    # Crosswalk markers (orange diamonds + crossing direction)
    for cw in scenario_metadata.get('crosswalk', []):
        ct = cw['center']
        pa = cw['point_a']
        pb = cw['point_b']
        axes[1, 1].scatter(ct[0], ct[1], c='orange', s=150, marker='D',
                           edgecolors='white', linewidths=1.5, zorder=5)
        # Draw A-B line
        axes[1, 1].plot([pa[0], pb[0]], [pa[1], pb[1]],
                       color='orange', linewidth=2, linestyle='--', alpha=0.8)
        axes[1, 1].scatter([pa[0], pb[0]], [pa[1], pb[1]],
                          c='orange', s=60, marker='x', zorder=5)

    # Legend
    from matplotlib.lines import Line2D
    legend_items = []
    if scenario_metadata.get('queue'):
        legend_items.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='blue',
                                   markersize=10, label='Queue (Bus Stop)'))
    if scenario_metadata.get('gathering'):
        legend_items.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='lime',
                                   markersize=10, label='Gathering'))
    if scenario_metadata.get('crosswalk'):
        legend_items.append(Line2D([0], [0], marker='D', color='w', markerfacecolor='orange',
                                   markersize=10, label='Crosswalk'))
    if legend_items:
        axes[1, 1].legend(handles=legend_items, loc='upper right', fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{scene_id}_debug_plot.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Extract Robust Top-down maps for Group-Aware Navigation Benchmark")
    parser.add_argument('--num_images', type=int, default=12, help='Number of scenes to capture.')
    parser.add_argument('--dataset_root', type=str, default='./datasets/Synthetic', help='Benchmark dataset root')
    parser.add_argument('--town', type=str, default='Town10HD_Opt',
                        help='CARLA town name to load (e.g., Town10HD_Opt, Town03, Town05)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sidewalk sampling.')
    parser.add_argument('--start_scene', type=int, default=0,
                        help='Start from this scene index for crash recovery (0 = from beginning).')
    args = parser.parse_args()

    # Resolve per-town settings, falling back to Town10 defaults for unknown towns.
    town_cfg = TOWN_CONFIGS.get(args.town, TOWN_CONFIGS['town10HD_Opt'])
    spectator_z = town_cfg['spectator_z']

    # Derive a short, filesystem-safe slug from the town name.
    # Examples: 'Town10HD_Opt' -> 'town10', 'Town03' -> 'town03', 'Town05' -> 'town05'
    import re
    _town_slug_match = re.search(r'town(\d+)', args.town, re.IGNORECASE)
    town_slug = 'town' + _town_slug_match.group(1) if _town_slug_match else args.town.lower()

    start_time = time.time()
    # --- 1. Create the directory layout (benchmark format) ---
    dirs = {
        'image_terrain': os.path.join(args.dataset_root, 'image_terrain'),
        'gt': os.path.join(args.dataset_root, 'gt'),
        'homography': os.path.join(args.dataset_root, 'homography'),
        'metadata': os.path.join(args.dataset_root, 'metadata') 
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # --- 2. Connect to CARLA and set synchronous mode ---
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    client.load_world(args.town)  # Load requested CARLA town
    world = client.get_world()
    
    destroyed_count = 0
    for actor in world.get_actors().filter('sensor.camera.*'):
        if actor.is_alive:
            actor.destroy()
            destroyed_count += 1
    if destroyed_count > 0:
        print(f"[Cleanup] Destroyed {destroyed_count} leftover zombie cameras.")
        
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)

    weather = world.get_weather()
    weather.sun_altitude_angle = 90.0 # minimize shadows in the map capture
    world.set_weather(weather)

    world.unload_map_layer(carla.MapLayer.Foliage) 
    # world.unload_map_layer(carla.MapLayer.Buildings) 
    world.unload_map_layer(carla.MapLayer.StreetLights) 

    # --- 3. Spawn the sensors (only once, to save memory) ---
    blueprint_library = world.get_blueprint_library()
    image_resolution = 2048
    


    def setup_camera(bp_name):
        bp = blueprint_library.find(bp_name)
        bp.set_attribute('image_size_x', str(image_resolution))
        bp.set_attribute('image_size_y', str(image_resolution))
        bp.set_attribute('fov', '45')
        return bp

    ortho_size = spectator_z * 2 * math.tan(math.radians(45 / 2))  # Calculate orthographic size based on FOV and height
    ortho_size = round(ortho_size, 3)  # Round to 2 decimal places for cleaner logs
    pixel_scale = ortho_size / image_resolution

    init_transform = carla.Transform(carla.Location(z=spectator_z), carla.Rotation(pitch=-90.0))
    sem_camera = world.spawn_actor(setup_camera('sensor.camera.semantic_segmentation'), init_transform)
    rgb_camera = world.spawn_actor(setup_camera('sensor.camera.rgb'), init_transform)

    sem_queue = queue_module.Queue()
    rgb_queue = queue_module.Queue()
    sem_camera.listen(sem_queue.put)
    rgb_camera.listen(rgb_queue.put)

    sidewalks = world.get_environment_objects(carla.CityObjectLabel.Sidewalks)

    # Filter sidewalks by ROI if defined for this town.
    # ROI format: [x_min, y_min, x_max, y_max] or [x_min, y_min, x_max, y_max, z_max]
    # min/max are auto-sorted so order doesn't matter. z_max excludes elevated sidewalks.
    roi = town_cfg.get('roi')
    if roi is not None:
        x_a, y_a, x_b, y_b = roi[0], roi[1], roi[2], roi[3]
        x_min, x_max = min(x_a, x_b), max(x_a, x_b)
        y_min, y_max = min(y_a, y_b), max(y_a, y_b)
        z_max = roi[4] if len(roi) > 4 else None
        # EnvironmentObject: world position is in bounding_box.location (not transform.location)
        sidewalks = [sw for sw in sidewalks
                     if x_min <= sw.bounding_box.location.x <= x_max
                     and y_min <= sw.bounding_box.location.y <= y_max
                     and (z_max is None or sw.bounding_box.location.z < z_max)]
        print(f"[ROI Filter] {len(sidewalks)} sidewalks within ROI {roi}")

    import random as _random

    def _diversity_sample(pool, n, seed, min_dist_m=60.0, fallback_steps=3):
        """Greedy diversity sampling: enforce min_dist_m between selected sidewalk centers.

        Falls back to progressively smaller min_dist (halved each step) if the pool
        cannot satisfy the requested count at the current threshold.
        """
        if n <= 0 or len(pool) <= n:
            return list(pool)

        rng = _random.Random(seed)
        shuffled = list(pool)
        rng.shuffle(shuffled)

        threshold = min_dist_m
        for _step in range(fallback_steps + 1):
            selected = []
            selected_locs = []
            for sw in shuffled:
                loc = sw.bounding_box.location
                if all(math.sqrt((loc.x - s[0])**2 + (loc.y - s[1])**2) >= threshold
                       for s in selected_locs):
                    selected.append(sw)
                    selected_locs.append((loc.x, loc.y))
                if len(selected) >= n:
                    break
            if len(selected) >= n:
                print(f"[Sidewalk Sampling] seed={seed}  pool={len(pool)}  "
                      f"selected={len(selected)}  min_dist={threshold:.0f}m")
                return selected
            # Relax threshold and retry
            threshold /= 2.0
            print(f"[Sidewalk Sampling] min_dist={threshold*2:.0f}m gave only "
                  f"{len(selected)}/{n} — relaxing to {threshold:.0f}m")

        # Final fallback: plain random sample to hit the requested count
        print(f"[Sidewalk Sampling] diversity exhausted — random fill to {n}")
        rng2 = _random.Random(seed + 1)
        return rng2.sample(pool, n)

    sidewalks_to_process = _diversity_sample(sidewalks, args.num_images, args.seed)

    # Crash recovery: skip already-processed scenes
    if args.start_scene > 0:
        print(f"[Resume] Skipping scenes 0–{args.start_scene - 1}, starting from scene {args.start_scene}")
        sidewalks_to_process = sidewalks_to_process[args.start_scene:]

    # Semantic preprocessor (generates walkable mask and appearance map)
    config_path = os.path.join(os.path.dirname(__file__), "configs/scenarios.yaml")
    preprocessor = SemanticPreprocessor(config_path)

    # Unified scenario generator (generates scenario-specific maps)
    unified_gen = UnifiedScenarioGenerator(config_path)

    # Cache environment objects ONCE (prevents repeated queries that cause memory issues)
    # Dynamic label needed for BP_FoodCart_decal3 (parasol food stalls)
    _static_objs  = world.get_environment_objects(carla.CityObjectLabel.Static)
    _other_objs   = world.get_environment_objects(carla.CityObjectLabel.Other)
    _dynamic_objs = world.get_environment_objects(carla.CityObjectLabel.Dynamic)
    all_bus_stop_candidates = list(_static_objs) + list(_other_objs) + list(_dynamic_objs)

    # Building + wall objects: used for food cart face-border placement
    _bldg_objs = list(world.get_environment_objects(carla.CityObjectLabel.Buildings))
    _wall_objs = list(world.get_environment_objects(carla.CityObjectLabel.Walls))
    all_building_candidates = _bldg_objs + _wall_objs


    # Get map for bus spawning
    carla_map = world.get_map()
    blueprint_library = world.get_blueprint_library()
    bus_blueprint = blueprint_library.find('vehicle.mitsubishi.fusorosa')

    # Track spawned buses/food carts for cleanup
    spawned_buses = []
    spawned_food_carts = []
    spawned_buses_info = []       # per-scene bus transform records for metadata
    spawned_food_carts_info = []  # per-scene dynamic food cart records for metadata

    # --- 4. Capture and formatting loop ---
    try:
        for i, sidewalk in enumerate(sidewalks_to_process, start=args.start_scene):
            spawned_buses_info.clear()       # reset per scene
            spawned_food_carts_info.clear()  # reset per scene
            _scene_start = time.time()
            scene_id = f"carla_{town_slug}_{i:03d}"
            center_loc = sidewalk.bounding_box.location

            target_transform = carla.Transform(
                carla.Location(x=center_loc.x, y=center_loc.y, z=center_loc.z + spectator_z),
                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
            )
            
            sem_camera.set_transform(target_transform)
            rgb_camera.set_transform(target_transform)

            # Aggressively clear queues to prevent memory buildup
            cleared_rgb = 0
            while not rgb_queue.empty():
                try:
                    rgb_queue.get_nowait()
                    cleared_rgb += 1
                except:
                    break
            cleared_sem = 0
            while not sem_queue.empty():
                try:
                    sem_queue.get_nowait()
                    cleared_sem += 1
                except:
                    break

            if cleared_rgb > 5 or cleared_sem > 5:
                print(f"[Queue] Cleared {cleared_rgb} RGB, {cleared_sem} semantic frames from queue")

            # === STEP 1: Spawn buses BEFORE capturing ===
            viewport_bus_stops = []
            viewport_food_carts = []

            # Find bus spawn candidates from roads adjacent to the scene's center sidewalk.
            # Strategy: probe points along the sidewalk bbox perimeter → snap to nearest
            # driving lane via get_waypoint(project_to_road=True) → verify adjacency to sidewalk.
            _SIDEWALK_TYPES = {carla.LaneType.Sidewalk, carla.LaneType.Shoulder,
                               carla.LaneType.Parking, carla.LaneType.Border, carla.LaneType.Biking}
            _MIN_BUS_DIST = 15.0   # minimum distance (m) between two spawned buses
            _PROBE_OFFSET = 8.0    # meters outward from sidewalk edge to probe for road

            # Build probe points: sample along all 4 edges of the sidewalk bbox,
            # offset outward so get_waypoint() snaps onto the adjacent road.
            sw_bbox = sidewalk.bounding_box  # carla.BoundingBox
            sw_cx, sw_cy, sw_cz = center_loc.x, center_loc.y, center_loc.z
            sw_ex = sw_bbox.extent.x  # half-extent in x
            sw_ey = sw_bbox.extent.y  # half-extent in y

            probe_points = []
            _NUM_SAMPLES = 6  # samples per edge
            for t in np.linspace(-1.0, 1.0, _NUM_SAMPLES):
                # North / South edges (vary x, fix y at ±ey)
                probe_points.append(carla.Location(
                    x=sw_cx + t * sw_ex, y=sw_cy - (sw_ey + _PROBE_OFFSET), z=sw_cz))
                probe_points.append(carla.Location(
                    x=sw_cx + t * sw_ex, y=sw_cy + (sw_ey + _PROBE_OFFSET), z=sw_cz))
                # East / West edges (vary y, fix x at ±ex)
                probe_points.append(carla.Location(
                    x=sw_cx - (sw_ex + _PROBE_OFFSET), y=sw_cy + t * sw_ey, z=sw_cz))
                probe_points.append(carla.Location(
                    x=sw_cx + (sw_ex + _PROBE_OFFSET), y=sw_cy + t * sw_ey, z=sw_cz))

            viewport_waypoints = []
            _seen_road_ids = set()  # deduplicate same road segment

            for probe in probe_points:
                # Snap to nearest driving lane
                wp = carla_map.get_waypoint(
                    probe, project_to_road=True, lane_type=carla.LaneType.Driving)
                if wp is None:
                    continue

                # Must be within viewport
                wp_loc = wp.transform.location
                if (abs(wp_loc.x - center_loc.x) >= ortho_size / 2 or
                        abs(wp_loc.y - center_loc.y) >= ortho_size / 2):
                    continue

                # Traverse right lane chain to find rightmost driving lane adjacent to sidewalk
                current = wp
                candidate = None
                while True:
                    right = current.get_right_lane()
                    if right is None:
                        break
                    if right.lane_type == carla.LaneType.Driving:
                        current = right
                    elif right.lane_type in _SIDEWALK_TYPES:
                        candidate = current
                        break
                    else:
                        break
                if candidate is None:
                    continue

                # Deduplicate by (road_id, lane_id, section_id)
                road_key = (candidate.road_id, candidate.lane_id, candidate.section_id)
                if road_key in _seen_road_ids:
                    continue
                _seen_road_ids.add(road_key)
                viewport_waypoints.append(candidate)

            # Try to spawn 1-2 buses per scene, enforcing minimum spacing
            import random
            random.seed(i)  # Deterministic per scene
            num_buses_to_spawn = min(2, len(viewport_waypoints))

            if len(viewport_waypoints) > 0:
                shuffled = list(viewport_waypoints)
                random.shuffle(shuffled)
                sampled_waypoints = []
                for _wp in shuffled:
                    _loc = _wp.transform.location
                    if all(_loc.distance(_s.transform.location) >= _MIN_BUS_DIST
                           for _s in sampled_waypoints):
                        sampled_waypoints.append(_wp)
                    if len(sampled_waypoints) >= num_buses_to_spawn:
                        break

                for wp in sampled_waypoints:
                    # Position bus at curb (slightly offset into driving lane)
                    spawn_transform = wp.transform
                    spawn_transform.location += spawn_transform.get_right_vector() * (wp.lane_width * 0.3)
                    spawn_transform.location.z += 0.5

                    # Try to spawn bus
                    bus_actor = world.try_spawn_actor(bus_blueprint, spawn_transform)

                    if bus_actor:
                        # Disable physics
                        bus_actor.set_simulate_physics(False)
                        spawned_buses.append(bus_actor)

                        # IMPORTANT: Tick to update bus position in CARLA
                        world.tick()
                        world.tick()

                        # Clear camera queues (discard images from ticks)
                        while not rgb_queue.empty():
                            try:
                                rgb_queue.get_nowait()
                            except:
                                break
                        while not sem_queue.empty():
                            try:
                                sem_queue.get_nowait()
                            except:
                                break

                        # Calculate queue anchor (right side of bus)
                        transform = bus_actor.get_transform()
                        bbox = bus_actor.bounding_box
                        bbox_center_world = transform.transform(bbox.location)

                        # Queue starts on right side of bus (sidewalk side = door side)
                        right_vector = transform.get_right_vector()

                        # Anchor: bus door (right side + clearance)
                        anchor_world = bbox_center_world + right_vector * (bbox.extent.y + 1.5)

                        # Transform to image coordinates
                        rel_x = anchor_world.x - center_loc.x
                        rel_y = anchor_world.y - center_loc.y
                        img_x = int(image_resolution / 2 + rel_y / pixel_scale)
                        img_y = int(image_resolution / 2 - rel_x / pixel_scale)

                        # Check if anchor is within viewport
                        if not (0 <= img_x < image_resolution and 0 <= img_y < image_resolution):
                            bus_actor.destroy()
                            spawned_buses.remove(bus_actor)
                            continue

                        # Queue direction: into the sidewalk (perpendicular to road)
                        # Robot walking along the sidewalk must cross this queue → dilemma
                        queue_direction = carla.Vector3D(right_vector.x, right_vector.y, 0.0)
                        queue_dir_img = np.array([queue_direction.y, -queue_direction.x])
                        queue_dir_img = queue_dir_img / (np.linalg.norm(queue_dir_img) + 1e-6)

                        bus_entry = {
                            'type': 'bus',
                            'location': bbox_center_world,
                            'goal_point': [img_x, img_y],
                            'direction': queue_dir_img.tolist(),
                            'length_px': 1000,  # generous scan upper bound (~20m); validator applies safety_margin
                            'pixel_per_meter': 1.0 / pixel_scale,
                            'source': 'bus',
                            'name': f'Bus_{i}_{len(viewport_bus_stops)}'
                        }
                        viewport_bus_stops.append(bus_entry)

                        # Record spawn transform for benchmark re-spawn
                        t = bus_actor.get_transform()
                        spawned_buses_info.append({
                            'blueprint': 'vehicle.mitsubishi.fusorosa',
                            'x': t.location.x,
                            'y': t.location.y,
                            'z': t.location.z,
                            'pitch': t.rotation.pitch,
                            'yaw': t.rotation.yaw,
                            'roll': t.rotation.roll,
                        })

                        pass  # bus spawn logged in scene summary
                    else:
                        pass  # spawn failure is non-critical; bus queue simply won't be generated

            # === STEP 1b: Spawn food carts against building/wall face borders ===
            # Placed against the face of the nearest building/wall that fronts the sidewalk.
            # Spawned before frame capture → footprint excluded from walkable_mask naturally.
            _fc_bp = blueprint_library.find('static.prop.foodcart')
            spawned_food_carts = []
            _MAX_FC = 2
            _MIN_FC_DIST_M = 6.0
            _bus_locs = [carla.Location(x=b['x'], y=b['y'], z=b['z'])
                         for b in spawned_buses_info]

            # Build candidate spawn points from building/wall face borders
            _fc_candidates = []
            for obj in all_building_candidates:
                bbox = obj.bounding_box
                bwx, bwy = bbox.location.x, bbox.location.y

                # Coarse viewport filter (2× ortho_size margin)
                if (abs(bwx - center_loc.x) > ortho_size or
                        abs(bwy - center_loc.y) > ortho_size):
                    continue

                tr  = obj.transform
                fwd = tr.get_forward_vector()
                rgt = tr.get_right_vector()
                ex, ey = bbox.extent.x, bbox.extent.y

                # 4 faces: (outward_normal_xy, half_extent_along_normal, half_face_length)
                _faces = [
                    (( fwd.x,  fwd.y), ex, ey),
                    ((-fwd.x, -fwd.y), ex, ey),
                    (( rgt.x,  rgt.y), ey, ex),
                    ((-rgt.x, -rgt.y), ey, ex),
                ]

                for (nx, ny), half_norm, half_len in _faces:
                    # Face center in world
                    fcx = bwx + nx * half_norm
                    fcy = bwy + ny * half_norm

                    # Only keep faces that point toward the scene center (sidewalk side)
                    dot = nx * (center_loc.x - fcx) + ny * (center_loc.y - fcy)
                    if dot <= 0:
                        continue

                    # Tangent direction (perpendicular to normal, in XY plane)
                    tx, ty = -ny, nx

                    # Sample along face every 4m
                    n_samp = max(1, int(half_len * 2 / 4.0))
                    for k in range(n_samp):
                        t = (float(k) / max(n_samp - 1, 1) - 0.5) * 2.0 * half_len \
                            if n_samp > 1 else 0.0

                        # Spawn point: on face surface + 0.5m outward clearance
                        px = fcx + tx * t + nx * 0.5
                        py = fcy + ty * t + ny * 0.5

                        # Must be within viewport
                        if (abs(px - center_loc.x) >= ortho_size / 2 - 2.0 or
                                abs(py - center_loc.y) >= ortho_size / 2 - 2.0):
                            continue

                        _fc_candidates.append((px, py, nx, ny))

            random.seed(i + 7777)
            random.shuffle(_fc_candidates)

            for _px, _py, _nx, _ny in _fc_candidates:
                if len(spawned_food_carts) >= _MAX_FC:
                    break
                _cloc = carla.Location(x=_px, y=_py, z=center_loc.z)
                _too_close = (
                    any(_cloc.distance(bl) < _MIN_FC_DIST_M for bl in _bus_locs) or
                    any(_cloc.distance(fc.get_location()) < _MIN_FC_DIST_M
                        for fc in spawned_food_carts)
                )
                if _too_close:
                    continue

                # Face outward from building (forward = face normal)
                _yaw = float(np.degrees(np.arctan2(_ny, _nx)))
                _fc_t = carla.Transform(
                    carla.Location(x=_px, y=_py, z=center_loc.z + 0.1),
                    carla.Rotation(yaw=_yaw)
                )
                _fc_actor = world.try_spawn_actor(_fc_bp, _fc_t)
                if _fc_actor:
                    _fc_actor.set_simulate_physics(False)
                    spawned_food_carts.append(_fc_actor)

            if spawned_food_carts:
                world.tick()
                world.tick()
                while not rgb_queue.empty():
                    try: rgb_queue.get_nowait()
                    except: break
                while not sem_queue.empty():
                    try: sem_queue.get_nowait()
                    except: break

            # === STEP 2: Stabilize environment after bus + food cart spawn ===
            if len(spawned_buses) > 0 or len(spawned_food_carts) > 0:
                for _ in range(10):  # 10 ticks to let rendering stabilize
                    world.tick()
                # Clear queues after stabilization ticks
                while not rgb_queue.empty():
                    try:
                        rgb_queue.get_nowait()
                    except:
                        break
                while not sem_queue.empty():
                    try:
                        sem_queue.get_nowait()
                    except:
                        break

            # === STEP 3: Collect bus stops and food carts ===
            # Skip building queue (inaccurate bbox, not good for benchmark)
            # Use cached environment objects (avoid repeated queries)

            # Filter bus stops and food carts within viewport (use cached objects)
            for obj in all_bus_stop_candidates:
                obj_name_lower = obj.name.lower()
                bbox = obj.bounding_box
                dx = bbox.location.x - center_loc.x
                dy = bbox.location.y - center_loc.y

                # Check if within viewport
                if abs(dx) < ortho_size / 2 and abs(dy) < ortho_size / 2:
                    # --- Bus Stop Detection (for queue) ---
                    if 'busstop' in obj_name_lower and obj_name_lower.endswith('sm_0'):
                        # Use check_objects.py logic: forward + right vectors
                        transform = obj.transform
                        forward_vec = transform.get_forward_vector()
                        right_vec = transform.get_right_vector()

                        # Queue goes: forward offset, then extends to right
                        # Start offset: in front of bus stop
                        start_offset = bbox.extent.x + 1.5  # 1.5m clearance

                        # Base anchor location (in CARLA world coords)
                        base_anchor_world = bbox.location + (forward_vec * start_offset)

                        # Queue direction: right vector (flipped)
                        # Flip right to left for queue extension
                        queue_direction = carla.Vector3D(-right_vec.x, -right_vec.y, 0.0)

                        # Transform to image coordinates: CARLA [X,Y] → Image [Y,-X]
                        # Calculate in camera-relative coords
                        rel_x = base_anchor_world.x - center_loc.x
                        rel_y = base_anchor_world.y - center_loc.y

                        # Image pixel coords (origin at top-left)
                        img_x = int(image_resolution / 2 + rel_y / pixel_scale)
                        img_y = int(image_resolution / 2 - rel_x / pixel_scale)

                        # Queue direction in image coords [dx, dy]
                        # Swap and flip: CARLA [X,Y] → Image [Y,-X]
                        queue_dir_img = np.array([queue_direction.y, -queue_direction.x])
                        queue_dir_img = queue_dir_img / (np.linalg.norm(queue_dir_img) + 1e-6)

                        viewport_bus_stops.append({
                            'type': 'bus_stop',
                            'location': bbox.location,
                            'goal_point': [img_x, img_y],  # Queue anchor in image coords
                            'direction': queue_dir_img.tolist(),  # Queue extension direction
                            'length_px': 1000,  # generous scan upper bound (~20m); validator applies safety_margin
                            'pixel_per_meter': 1.0 / pixel_scale,
                            'source': 'bus_stop',
                            'name': obj.name
                        })

                    # --- Food Cart Detection (for queue) ---
                    # SM_Cartel_Add: hot dog cart (Static label)
                    # BP_FoodCart_decal3: parasol food stalls (Dynamic label) — use SM_0 anchor only
                    elif ('cartel_add' in obj_name_lower or 'bp_foodcart' in obj_name_lower) and obj_name_lower.endswith('sm_0'):
                        # Food carts → queue anchor (people queue up in front of food carts)
                        transform = obj.transform
                        forward_vec = transform.get_forward_vector()

                        # Queue anchor: serving window side (SM_Cartel_Add has forward pointing INTO wall)
                        start_offset = bbox.extent.x + 1.0  # 1m clearance
                        neg_forward = carla.Vector3D(-forward_vec.x, -forward_vec.y, 0.0)
                        base_anchor_world = bbox.location + (neg_forward * start_offset)

                        rel_x = base_anchor_world.x - center_loc.x
                        rel_y = base_anchor_world.y - center_loc.y
                        img_x = int(image_resolution / 2 + rel_y / pixel_scale)
                        img_y = int(image_resolution / 2 - rel_x / pixel_scale)

                        # Queue extends backward (opposite of cart forward direction)
                        queue_direction = carla.Vector3D(-forward_vec.x, -forward_vec.y, 0.0)
                        queue_dir_img = np.array([queue_direction.y, -queue_direction.x])
                        queue_dir_img = queue_dir_img / (np.linalg.norm(queue_dir_img) + 1e-6)

                        viewport_bus_stops.append({
                            'type': 'food_cart',
                            'location': bbox.location,
                            'goal_point': [img_x, img_y],
                            'direction': queue_dir_img.tolist(),
                            'length_px': 1000,  # generous scan upper bound (~20m); validator applies safety_margin
                            'pixel_per_meter': 1.0 / pixel_scale,
                            'source': 'food_cart',
                            'name': obj.name
                        })

            n_food_cart_q = sum(1 for b in viewport_bus_stops if b.get('source') == 'food_cart')

            # === STEP 4: Capture frame with buses visible ===
            world.tick()
            world.tick()
            target_frame = world.tick()

            try:
                rgb_raw = None
                sem_raw = None

                # [Strict Sync 3]
                while True:
                    data = rgb_queue.get(timeout=5.0)
                    if data.frame == target_frame:
                        rgb_raw = data
                        break

                while True:
                    data = sem_queue.get(timeout=5.0)
                    if data.frame == target_frame:
                        sem_raw = data
                        break

                if rgb_raw is None or sem_raw is None:
                    print(f"[Warning] Could not sync frames for {scene_id}. Skipping...")
                    continue

                # 1. Save the RGB image
                rgb_array = np.frombuffer(rgb_raw.raw_data, dtype=np.dtype("uint8"))
                rgb_array = np.reshape(rgb_array, (rgb_raw.height, rgb_raw.width, 4))[:, :, :3]
                cv2.imwrite(os.path.join(dirs['image_terrain'], f"{scene_id}_bg.png"), rgb_array)

                # 2. Extract the raw semantic data for processing
                sem_array = np.frombuffer(sem_raw.raw_data, dtype=np.dtype("uint8"))
                sem_array = np.reshape(sem_array, (sem_raw.height, sem_raw.width, 4))
                semantic_raw = sem_array[:, :, 2].copy()

                # Convert to a Cityscapes RGB image for visualization
                sem_raw.convert(carla.ColorConverter.CityScapesPalette)
                sem_color_array = np.frombuffer(sem_raw.raw_data, dtype=np.dtype("uint8"))
                sem_color_array = np.reshape(sem_color_array, (sem_raw.height, sem_raw.width, 4))[:, :, :3]
                semantic_vis_rgb = cv2.cvtColor(sem_color_array, cv2.COLOR_BGR2RGB)

                # 3. Semantic Preprocessing (walkable mask + appearance map)
                cv2.imwrite(os.path.join(dirs['gt'], f"{scene_id}_semantic_raw.png"), semantic_raw)
                # The connected-component filter waits until the crosswalks are filled in
                # (skip_cc_filter=True), so islands that a crosswalk will join are not removed first.
                walkable_mask, appearance_map = preprocessor.generate_base_maps(
                    semantic_raw, pixel_per_meter=1.0 / pixel_scale, skip_cc_filter=True)

                # 3.5 Crosswalk Detection from CARLA Map API
                # Always applied to walkable mask regardless of crosswalk scenario enable flag.
                # Crosswalk scenario agent generation is separately gated by scenarios.yaml.
                viewport_crosswalks = []
                tmap = world.get_map()
                cw_points = tmap.get_crosswalks()

                # get_crosswalks() returns a flat list of carla.Location.
                # Each polygon ends when the current point matches the first point
                # of the current group (closing point). Vertex count varies by town.
                polygons = []
                current_poly = []
                for pt in cw_points:
                    if (current_poly and
                            abs(pt.x - current_poly[0].x) < 0.1 and
                            abs(pt.y - current_poly[0].y) < 0.1):
                        polygons.append(current_poly)  # closing point: end of polygon
                        current_poly = []
                    else:
                        current_poly.append(pt)
                if current_poly:  # flush any incomplete polygon
                    polygons.append(current_poly)

                for poly in polygons:
                    if len(poly) < 3:
                        continue

                    # Include crosswalk if ANY vertex falls within the viewport
                    half = ortho_size / 2
                    if not any(abs(pt.x - center_loc.x) <= half and
                               abs(pt.y - center_loc.y) <= half
                               for pt in poly):
                        continue

                    # Convert all vertices to image coords (clamp to image)
                    img_pts = []
                    for pt in poly:
                        rel_x = pt.x - center_loc.x
                        rel_y = pt.y - center_loc.y
                        img_x = int(image_resolution / 2 + rel_y / pixel_scale)
                        img_y = int(image_resolution / 2 - rel_x / pixel_scale)
                        img_x = max(0, min(image_resolution - 1, img_x))
                        img_y = max(0, min(image_resolution - 1, img_y))
                        img_pts.append([img_x, img_y])

                    if len(img_pts) < 3:
                        continue

                    img_pts = np.array(img_pts, dtype=np.float32)

                    # Fit minimum area rectangle to get center, size, angle
                    rect = cv2.minAreaRect(img_pts)
                    (cx, cy), (w, h), angle = rect
                    if min(w, h) < 1:
                        continue

                    # Determine crossing direction (short axis = direction people walk across)
                    angle_rad = np.radians(angle)
                    if w > h:
                        long_dir = np.array([np.cos(angle_rad), np.sin(angle_rad)])
                        half_long, half_short = w / 2, h / 2
                    else:
                        long_dir = np.array([-np.sin(angle_rad), np.cos(angle_rad)])
                        half_long, half_short = h / 2, w / 2

                    cross_dir = np.array([-long_dir[1], long_dir[0]])

                    point_a = np.array([cx, cy]) - cross_dir * half_short
                    point_b = np.array([cx, cy]) + cross_dir * half_short

                    _zone_center = ((point_a + point_b) / 2).tolist()
                    _ab_dist = float(np.linalg.norm(point_b - point_a))
                    _zone_radius = _ab_dist / 2 + 3.0 * (1.0 / pixel_scale)

                    viewport_crosswalks.append({
                        'type': 'crosswalk',
                        'center': [int(cx), int(cy)],
                        'point_a': point_a.tolist(),
                        'point_b': point_b.tolist(),
                        'width': float(half_long * 2),
                        'crossing_direction': cross_dir.tolist(),
                        'pixel_per_meter': 1.0 / pixel_scale,
                        'polygon_pts': [list(map(int, p)) for p in img_pts],
                        'zone_center': _zone_center,
                        'zone_radius': _zone_radius,
                    })

                _n_crosswalks = len(viewport_crosswalks)

                # Merge crosswalk polygons into walkable mask & appearance map
                for cw in viewport_crosswalks:
                    pts = np.array(cw['polygon_pts'], dtype=np.int32)
                    if len(pts) < 3:
                        continue

                    if np.array_equal(pts[0], pts[-1]):
                        pts = pts[:-1]

                    # Scale polygon 140% along long axis only (anisotropic, PCA-based)
                    centroid = pts.mean(axis=0)
                    centered = (pts - centroid).astype(np.float32)
                    cov = np.cov(centered.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    long_axis = eigenvectors[:, np.argmax(eigenvalues)]
                    short_axis = np.array([-long_axis[1], long_axis[0]])
                    proj_long = centered @ long_axis
                    proj_short = centered @ short_axis
                    scaled_centered = (np.outer(proj_long * 1.4, long_axis)
                                       + np.outer(proj_short, short_axis))
                    scaled_pts = (scaled_centered + centroid).astype(np.int32)
                    scaled_pts[:, 0] = np.clip(scaled_pts[:, 0], 0, image_resolution - 1)
                    scaled_pts[:, 1] = np.clip(scaled_pts[:, 1], 0, image_resolution - 1)

                    cv2.fillPoly(walkable_mask, [scaled_pts], 255)

                # Connected-component filter: runs once, after the crosswalk fill.
                # This is the filter that generate_base_maps() skipped with skip_cc_filter=True.
                # Islands joined by crosswalks now form one component, and the largest one is kept.
                _num_labels, _labels, _stats, _ = cv2.connectedComponentsWithStats(
                    walkable_mask, connectivity=8)
                if _num_labels > 2:
                    _largest_idx = np.argmax(_stats[1:, cv2.CC_STAT_AREA]) + 1
                    _merged = np.zeros_like(walkable_mask)
                    _merged[_labels == _largest_idx] = 255
                    _before = np.sum(walkable_mask > 0)
                    _after = np.sum(_merged > 0)
                    pass  # CC filter applied silently
                    walkable_mask = _merged
                elif _num_labels <= 1:
                    print(f"  [Warning] No walkable areas found after crosswalk fill")
                    walkable_mask = np.zeros_like(walkable_mask)

                # Recompute appearance map with final walkable_mask
                appearance_map = preprocessor._generate_base_appearance(walkable_mask)
                sigma = preprocessor.config.get('probability_smoothing_sigma', 15)
                appearance_map = cv2.GaussianBlur(appearance_map, (sigma, sigma), 0)

                # === STEP 3.5: Build queue anchors from already-spawned food carts ===
                # Food carts were spawned in STEP 1b and are physically present in the
                # semantic capture, so their footprints are already excluded from walkable_mask.
                # Validate anchor against walkable_mask to filter road-straddling outliers.
                for _fc_actor in spawned_food_carts:
                    _fc_t = _fc_actor.get_transform()
                    _fc_loc = _fc_t.location
                    _fc_fwd = _fc_t.get_forward_vector()

                    # Queue anchor: 1.5m in front of the food cart (clears the food cart bbox)
                    _anchor_world = carla.Location(
                        x=_fc_loc.x + _fc_fwd.x * 1.5,
                        y=_fc_loc.y + _fc_fwd.y * 1.5,
                        z=_fc_loc.z,
                    )
                    _rel_x = _anchor_world.x - center_loc.x
                    _rel_y = _anchor_world.y - center_loc.y
                    _img_x = int(image_resolution / 2 + _rel_y / pixel_scale)
                    _img_y = int(image_resolution / 2 - _rel_x / pixel_scale)

                    if not (0 <= _img_x < image_resolution and 0 <= _img_y < image_resolution):
                        print(f'  [FoodCart] skip: anchor out of viewport')
                        continue

                    # Anchor must be on walkable sidewalk — rejects road-straddling outliers
                    if walkable_mask[_img_y, _img_x] == 0:
                        print(f'  [FoodCart] skip: anchor not walkable ({_img_x},{_img_y})')
                        continue

                    # Queue direction: perpendicular to forward (along sidewalk)
                    _qd = np.array([_fc_fwd.y, -_fc_fwd.x])
                    _qd /= max(np.linalg.norm(_qd), 1e-6)

                    viewport_bus_stops.append({
                        'type': 'food_cart_dynamic',
                        'location': _fc_loc,
                        'goal_point': [_img_x, _img_y],
                        'direction': _qd.tolist(),
                        'length_px': 1000,
                        'pixel_per_meter': 1.0 / pixel_scale,
                        'source': 'food_cart_dynamic',
                        'name': f'DynFC_{i}_{len(spawned_food_carts_info)}',
                    })

                    _t = _fc_actor.get_transform()
                    spawned_food_carts_info.append({
                        'blueprint': 'static.prop.foodcart',
                        'x': _t.location.x,
                        'y': _t.location.y,
                        'z': _t.location.z,
                        'pitch': _t.rotation.pitch,
                        'yaw': _t.rotation.yaw,
                        'roll': _t.rotation.roll,
                    })

                if spawned_food_carts_info:
                    print(f"  [FoodCart] {len(spawned_food_carts_info)} dynamic food cart(s) → queue anchors added")

                # 4. Unified Scenario Generation (queue + cohesive + gathering + crosswalk)

                # Generate all enabled scenario maps using unified generator
                scenario_maps, scenario_metadata = unified_gen.generate_from_carla(
                    semantic_raw, walkable_mask, pixel_per_meter=1.0 / pixel_scale,
                    world=world, camera_transform=target_transform,
                    ortho_size=ortho_size, image_resolution=image_resolution,
                    buildings=None,  # Skip buildings (inaccurate bbox)
                    bus_stops=viewport_bus_stops,
                    food_carts=viewport_food_carts,
                    crosswalks=viewport_crosswalks
                )

                # 4.5 Validate scenario points against walkable mask
                with open(config_path, 'r') as f:
                    full_config = yaml.safe_load(f)
                    gathering_config = full_config.get('gathering', {})
                    queue_config = full_config.get('queue', {})

                validated_metadata = validate_scenario_points(
                    scenario_metadata,
                    walkable_mask,
                    pixel_per_meter=1.0 / pixel_scale,
                    gathering_config=gathering_config,
                    queue_config=queue_config
                )

                # Replace with validated metadata
                scenario_metadata = validated_metadata

                # Rebuild scenario density map using validated metadata
                scenario_density = np.zeros((image_resolution, image_resolution), dtype=np.float32)

                # Redraw queue lines
                for queue in validated_metadata['queue']:
                    start = queue['goal_point']
                    direction = np.array(queue['direction'])
                    length = queue.get('length_px', 120)
                    end_x = int(start[0] + direction[0] * length)
                    end_y = int(start[1] + direction[1] * length)
                    cv2.line(scenario_density, (int(start[0]), int(start[1])),
                            (end_x, end_y), 1.0, thickness=30)

                # Redraw gathering circles with adaptive radius
                for gathering in validated_metadata['gathering']:
                    center = gathering['center']
                    radius = gathering.get('spawn_radius', 100)
                    cv2.circle(scenario_density, (int(center[0]), int(center[1])),
                              radius, 1.0, thickness=-1)

                # Normalize final density
                if scenario_density.max() > 0:
                    scenario_density = scenario_density / scenario_density.max()

                pass

                # Save scenario_density map (for emitter goal sampling)
                cv2.imwrite(os.path.join(dirs['gt'], f"{scene_id}_scenario_density.png"),
                           (scenario_density * 255).astype(np.uint8))

                # Collect scenario counts for summary
                _n_q  = len(scenario_metadata.get('queue', []))
                _n_g  = len(scenario_metadata.get('gathering', []))
                _n_cw = len(scenario_metadata.get('crosswalk', []))

                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

                # 5. Save the debug plot (semantic + walkable + appearance + scenario with markers)
                debug_dir = os.path.join(args.dataset_root, 'debug_plots')
                save_debug_plot(semantic_vis_rgb, walkable_mask, appearance_map, scenario_density,
                                scenario_metadata, scene_id, debug_dir)

                # 6. Save the base maps
                cv2.imwrite(os.path.join(dirs['gt'], f"{scene_id}_walkable.png"), walkable_mask)
                cv2.imwrite(os.path.join(dirs['gt'], f"{scene_id}_appearance_density.png"), appearance_map)

                # Population density (use walkable map for uniform spawn)
                cv2.imwrite(os.path.join(dirs['gt'], f"{scene_id}_population_density.png"), walkable_mask)

                # 7. Save the combined metadata (coordinate transform + scenarios in one file)
                with open(os.path.join(dirs['homography'], f"{scene_id}_scale.txt"), 'w') as f:
                    f.write(f"{pixel_scale:.6f}")

                unified_metadata = {
                    "center_x": center_loc.x,
                    "center_y": center_loc.y,
                    "center_z": center_loc.z,
                    "ortho_size": ortho_size,
                    "pixel_per_meter": 1.0 / pixel_scale,
                    "scenarios": scenario_metadata,
                    "static_buses": spawned_buses_info.copy(),
                    "static_food_carts": spawned_food_carts_info.copy(),
                }
                with open(os.path.join(dirs['metadata'], f"{scene_id}_meta.json"), 'w') as f:
                    json.dump(unified_metadata, f, indent=4, default=convert_numpy)

                _walkable_px = np.sum(walkable_mask > 0)
                _elapsed = time.time() - _scene_start
                _total = len(sidewalks_to_process)
                _idx = i - args.start_scene + 1
                print(f"[{_idx:>3}/{_total}] {scene_id}  │  Q:{_n_q}  G:{_n_g}  CW:{_n_cw}  │  {_walkable_px/1e6:.2f}M px  │  {_elapsed:.1f}s ✓")

                # Explicitly delete large objects to free memory
                del rgb_array, sem_array, semantic_raw, sem_color_array, semantic_vis_rgb
                del walkable_mask, appearance_map, scenario_maps, scenario_density
                del viewport_bus_stops, viewport_food_carts

                # Clean up buses and food carts spawned for this scene
                for bus in spawned_buses:
                    try:
                        if bus.is_alive:
                            bus.destroy()
                    except:
                        pass
                spawned_buses.clear()
                for fc in spawned_food_carts:
                    try:
                        if fc.is_alive:
                            fc.destroy()
                    except:
                        pass
                spawned_food_carts.clear()

            except Exception as e:
                import traceback
                print(f"[Error] Failed to process {scene_id}: {e}")
                traceback.print_exc()
                # Clean up buses and food carts even on error
                for bus in spawned_buses:
                    try:
                        if bus.is_alive:
                            bus.destroy()
                    except:
                        pass
                spawned_buses.clear()
                for fc in spawned_food_carts:
                    try:
                        if fc.is_alive:
                            fc.destroy()
                    except:
                        pass
                spawned_food_carts.clear()

    finally:
        # === Release resources (always runs) ===
        # 0. Clean up any remaining buses and food carts
        for bus in spawned_buses:
            try:
                if bus.is_alive:
                    bus.destroy()
            except:
                pass
        for fc in spawned_food_carts:
            try:
                if fc.is_alive:
                    fc.destroy()
            except:
                pass

        # 1. Stop the listeners
        try:
            if 'sem_camera' in locals() and sem_camera is not None and sem_camera.is_listening:
                sem_camera.stop()
        except Exception as e:
            print(f"[Cleanup] Error stopping semantic listener: {e}")

        try:
            if 'rgb_camera' in locals() and rgb_camera is not None and rgb_camera.is_listening:
                rgb_camera.stop()
        except Exception as e:
            print(f"[Cleanup] Error stopping RGB listener: {e}")

        # 2. Give CARLA time to stop listeners
        try:
            world.tick()
            world.tick()
        except:
            pass

        # 3. Empty the queues
        if 'sem_queue' in locals():
            while not sem_queue.empty():
                try:
                    sem_queue.get_nowait()
                except:
                    break
        if 'rgb_queue' in locals():
            while not rgb_queue.empty():
                try:
                    rgb_queue.get_nowait()
                except:
                    break

        # 4. Destroy the cameras
        if 'sem_camera' in locals() and sem_camera is not None:
            try:
                if sem_camera.is_alive:
                    sem_camera.destroy()
            except Exception as e:
                print(f"[Cleanup] Error destroying semantic camera: {e}")

        if 'rgb_camera' in locals() and rgb_camera is not None:
            try:
                if rgb_camera.is_alive:
                    rgb_camera.destroy()
            except Exception as e:
                print(f"[Cleanup] Error destroying RGB camera: {e}")

        # 5. Give CARLA time to destroy actors
        try:
            world.tick()
            world.tick()
        except:
            pass

        # 6. Turn off synchronous mode
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        except Exception as e:
            print(f"[Cleanup] Error disabling sync mode: {e}")

        print(f"[Done] Elapsed: {time.time() - start_time:.1f}s")
if __name__ == '__main__':
    main()