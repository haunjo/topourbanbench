"""
generate_foodcart_queues.py — Food cart virtual placement for scenes missing queue scenarios.

For scenes where scenarios.queue is empty, detect walkable placement candidates via
FFT convolution (same method as gathering detection), then compute the best queue
anchor direction per candidate using a 16-direction corridor scan.

No CARLA connection required — operates solely on saved walkable_mask PNGs and _meta.json.

Run from topourbanbench/generation/crowdes/:
    python generate_foodcart_queues.py --town town03 --subset topo_ft
    python generate_foodcart_queues.py --dry_run --town town05
"""

import argparse
import json
import math
import os
import sys
import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, List, Tuple

# ---------------------------------------------------------------------------
# Path setup: identical to generate_diversity_groups.py
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

_TOPO_UTILS_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..'))
if _TOPO_UTILS_DIR not in sys.path:
    sys.path.insert(0, _TOPO_UTILS_DIR)

from CrowdES.inference_model import CrowdESFramework
from utils.dataloader.synthetic_dataloader import SyntheticDataset
from utils.utils import reproducibility_settings
from utils.visualization import save_expert_trajectory_png
from utils.expert_trajectory import ExpertAgentAdapter
from utils.scenario_validator import validate_scenario_points
from utils.config import get_config

from CrowdES.evaluate_synthetic_dataset import (
    load_scenario_metadata,
    save_group_graph_json,
    discover_scenes_from_dataset,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCENARIO_LENGTH = 20 * 60 * 2   # frames (20 fps × 60 s × 2 min)
POSTFIX = 'crowdes-synthetic'
MODEL_CONFIG = './configs/model/CrowdES_zara2.yaml'
DATASET_ROOT = './datasets/Synthetic'
OUTPUT_BASE = '../../../output/generated/synthetic'
FC_SUFFIX = 99                   # food-cart generation suffix
IMAGE_RESOLUTION = 2048          # standard ortho resolution


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _load_meta(scene: str, dataset_root: str) -> dict:
    """Load raw _meta.json (center_x/y/z, ortho_size, pixel_per_meter, scenarios)."""
    meta_path = Path(dataset_root) / 'metadata' / f'{scene}_meta.json'
    with open(meta_path, 'r') as f:
        return json.load(f)


def _load_walkable(scene: str, dataset_root: str) -> np.ndarray:
    """Load walkable PNG as uint8 binary mask (>127 → 1)."""
    path = Path(dataset_root) / 'gt' / f'{scene}_walkable.png'
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'Walkable mask not found: {path}')
    return (img > 127).astype(np.uint8)


def _pixel_to_world(img_x: int, img_y: int, meta: dict) -> tuple:
    """
    Convert image pixel (img_x, img_y) to CARLA world coordinates.

    Follows extract_carla_maps.py convention:
        pixel_scale = ortho_size / IMAGE_RESOLUTION
        world_x = center_x + (IMAGE_RESOLUTION/2 - img_y) * pixel_scale
        world_y = center_y + (img_x - IMAGE_RESOLUTION/2) * pixel_scale
    """
    ortho_size = float(meta['ortho_size'])
    center_x   = float(meta['center_x'])
    center_y   = float(meta['center_y'])
    center_z   = float(meta.get('center_z', 0.0))
    pixel_scale = ortho_size / IMAGE_RESOLUTION

    world_x = center_x + (IMAGE_RESOLUTION / 2 - img_y) * pixel_scale
    world_y = center_y + (img_x - IMAGE_RESOLUTION / 2) * pixel_scale
    return world_x, world_y, center_z


def _save_meta_fc(scene: str, dataset_root: str, new_scenarios: dict,
                  static_food_carts: list) -> str:
    """Copy _meta.json, inject new scenarios + static_food_carts, save as _meta_fc.json."""
    meta = _load_meta(scene, dataset_root)
    meta['scenarios'] = new_scenarios
    meta['static_food_carts'] = static_food_carts

    out_path = Path(dataset_root) / 'metadata' / f'{scene}_meta_fc.json'
    with open(out_path, 'w') as f:
        json.dump(meta, f, indent=4)
    return str(out_path)


# ---------------------------------------------------------------------------
# Placement candidate detection (FFT convolution, same as gathering)
# ---------------------------------------------------------------------------

def detect_placement_candidates(walkable_mask: np.ndarray,
                                 pixel_per_meter: float,
                                 search_radius_m: float = 4.0,
                                 min_coverage_ratio: float = 0.5,
                                 max_candidates: int = 8,
                                 min_distance_m: float = 6.0) -> list:
    """
    Detect candidate food-cart placement positions via circular FFT convolution.

    Uses the same algorithm as _generate_gathering_map / _detect_gatherings:
    1. Convolve walkable mask with circular kernel of search_radius_m.
    2. Find local maxima with NMS distance = min_distance_m.
    3. Return up to max_candidates positions sorted by coverage (best first).

    Returns:
        List of (img_x, img_y) tuples in image pixel coordinates.
    """
    from scipy.ndimage import maximum_filter
    from scipy.signal import fftconvolve

    H, W = walkable_mask.shape

    search_radius_px = int(search_radius_m * pixel_per_meter)
    min_distance_px  = int(min_distance_m * pixel_per_meter)

    # Circular kernel
    y_grid, x_grid = np.ogrid[
        -search_radius_px:search_radius_px + 1,
        -search_radius_px:search_radius_px + 1,
    ]
    kernel = (x_grid ** 2 + y_grid ** 2 <= search_radius_px ** 2).astype(float)
    kernel_area = kernel.sum()

    walkable_float = (walkable_mask > 0).astype(float)
    coverage_map   = fftconvolve(walkable_float, kernel, mode='same')
    coverage_ratio = coverage_map / kernel_area

    valid_mask   = coverage_ratio >= min_coverage_ratio
    local_max_map = maximum_filter(coverage_map, size=min_distance_px)
    is_local_max  = (coverage_map == local_max_map) & valid_mask

    candidates_y, candidates_x = np.where(is_local_max)
    candidates_cov = coverage_ratio[candidates_y, candidates_x]

    sorted_idx = np.argsort(candidates_cov)[::-1]
    candidates_y   = candidates_y[sorted_idx]
    candidates_x   = candidates_x[sorted_idx]
    candidates_cov = candidates_cov[sorted_idx]

    selected = []
    for i in range(len(candidates_y)):
        if len(selected) >= max_candidates:
            break
        cy, cx = int(candidates_y[i]), int(candidates_x[i])

        too_close = any(
            math.sqrt((cy - sy) ** 2 + (cx - sx) ** 2) < min_distance_px
            for sx, sy, _ in selected
        )
        if not too_close:
            selected.append((cx, cy, float(candidates_cov[i])))

    # Return as (img_x, img_y) only
    return [(cx, cy) for cx, cy, _ in selected]


# ---------------------------------------------------------------------------
# Best queue anchor computation
# ---------------------------------------------------------------------------

def _scan_walkable_length(start_x: float, start_y: float,
                           direction: np.ndarray,
                           walkable_mask: np.ndarray,
                           max_scan_px: int,
                           gap_tolerance_px: int) -> int:
    """
    Scan along direction from (start_x, start_y) and return walkable extent.

    A run of consecutive non-walkable pixels <= gap_tolerance_px is bridged
    (lamp posts, segmentation noise). The scan stops when the gap exceeds
    the tolerance or bounds are exceeded.

    Returns walkable_length in pixels.
    """
    H, W = walkable_mask.shape
    walkable_length = 0
    consecutive_gap = 0

    for step in range(max_scan_px):
        tx = int(round(start_x + direction[0] * step))
        ty = int(round(start_y + direction[1] * step))

        if not (0 <= ty < H and 0 <= tx < W):
            break

        if walkable_mask[ty, tx] > 0:
            walkable_length = step + 1
            consecutive_gap = 0
        else:
            consecutive_gap += 1
            if consecutive_gap > gap_tolerance_px:
                break

    return walkable_length


def compute_best_queue_anchor(img_x: int, img_y: int,
                               walkable_mask: np.ndarray,
                               pixel_per_meter: float,
                               min_length_m: float = 1.5,
                               offset_m: float = 1.5,
                               max_scan_m: float = 20.0,
                               gap_tolerance_m: float = 0.4,
                               queue_id: str = 'fc_0') -> Optional[dict]:
    """
    Try 16 directions around (img_x, img_y) and pick the one with the longest
    walkable corridor.

    Strategy per direction:
      - The food cart sits at (img_x, img_y).
      - The queue anchor (goal_point) is placed at offset_m perpendicular to the
        food-cart facing, so people stand beside the cart rather than on top of it.
      - From the anchor, scan in the queue direction (away from cart) for walkable
        corridor length.

    Returns a queue metadata dict compatible with validate_scenario_points, or
    None if no direction yields walkable_length >= min_length_m.
    """
    H, W = walkable_mask.shape

    offset_px       = int(offset_m * pixel_per_meter)
    max_scan_px     = int(max_scan_m * pixel_per_meter)
    gap_tolerance_px = int(gap_tolerance_m * pixel_per_meter)
    min_length_px   = min_length_m * pixel_per_meter

    # 16 equally-spaced direction angles (food cart forward vector in image space)
    n_dirs = 16
    angles_deg = [i * (360.0 / n_dirs) for i in range(n_dirs)]

    best_length = -1
    best_result = None

    for angle_deg in angles_deg:
        angle_rad = math.radians(angle_deg)
        # Image-space forward direction of food cart
        fwd_x = math.cos(angle_rad)
        fwd_y = math.sin(angle_rad)

        # Queue stands perpendicular to cart forward: use 90° rotation of fwd
        # perp direction = (-fwd_y, fwd_x)  [left-hand side in image coords]
        perp_x = -fwd_y
        perp_y =  fwd_x

        # Anchor position: candidate + perp * offset
        anchor_x = img_x + perp_x * offset_px
        anchor_y = img_y + perp_y * offset_px

        anchor_ix = int(round(anchor_x))
        anchor_iy = int(round(anchor_y))

        # Anchor must be within image bounds
        if not (0 <= anchor_iy < H and 0 <= anchor_ix < W):
            continue

        # Anchor must be walkable
        if walkable_mask[anchor_iy, anchor_ix] == 0:
            continue

        # Queue direction = perp direction (people queue away from cart)
        queue_dir = np.array([perp_x, perp_y], dtype=float)

        wl = _scan_walkable_length(
            anchor_x, anchor_y, queue_dir,
            walkable_mask, max_scan_px, gap_tolerance_px
        )

        if wl > best_length:
            best_length = wl
            best_result = {
                'angle_deg': angle_deg,
                'anchor_ix': anchor_ix,
                'anchor_iy': anchor_iy,
                'queue_dir': queue_dir,
                'walkable_length_px': wl,
                'fwd_x': fwd_x,
                'fwd_y': fwd_y,
            }

    if best_result is None or best_length < min_length_px:
        return None

    r = best_result
    return {
        'type': 'queue',
        'source': 'food_cart',
        'goal_point': [r['anchor_ix'], r['anchor_iy']],
        'direction': r['queue_dir'].tolist(),
        'length_px': r['walkable_length_px'],
        'walkable_length_px': r['walkable_length_px'],
        'queue_id': queue_id,
        'pixel_per_meter': float(pixel_per_meter),
        # food cart position in image coords (for world coord conversion)
        '_cart_img_x': img_x,
        '_cart_img_y': img_y,
        '_cart_fwd_x': r['fwd_x'],
        '_cart_fwd_y': r['fwd_y'],
        '_angle_deg': r['angle_deg'],
    }


# ---------------------------------------------------------------------------
# CrowdES runner (mirrors generate_diversity_groups._run_crowdes exactly)
# ---------------------------------------------------------------------------

def _run_crowdes(scene: str, dataset: SyntheticDataset, framework: CrowdESFramework,
                 scene_idx: int, scenario_metadata: dict, unified_config_path: str,
                 out_dir: str, suffix: int) -> None:
    """Run CrowdES inference for one scene and write outputs."""
    data = dataset[scene_idx]

    with open(unified_config_path, 'r') as f:
        unified_config = yaml.safe_load(f)

    enabled_scenarios = [k for k, v in unified_config['scenarios'].items() if v['enabled']]
    filtered_metadata = {
        k: (v if k in enabled_scenarios else [])
        for k, v in scenario_metadata.items()
    }

    meta_summary = {k: len(v) for k, v in filtered_metadata.items() if v}
    if meta_summary:
        print(f'  [Metadata] {meta_summary}')

    framework.initialize_scene(
        data['img'], data['seg'], data['walkable'], data['navmesh'], data['H'],
        appearance_gt=data['appearance'], population_gt=data['population'],
        scenario_gt=data['scenario'],
        scenario_metadata=filtered_metadata,
    )
    generated = framework.generate(SCENARIO_LENGTH, seed=suffix)
    generated['scene'] = scene
    num_agents = len(generated['agent_id'].unique())
    print(f'  Generated: {num_agents} agents')

    generated = ExpertAgentAdapter.inject(generated, scene, framework=framework)

    columns = ['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y', 'group_id', 'group_type']
    export_df = generated[columns].copy()

    os.makedirs(out_dir, exist_ok=True)
    csv_path = f'{out_dir}/{scene}-{POSTFIX}-{suffix}.csv'
    export_df.to_csv(csv_path, index=False)

    json_path = csv_path.replace('.csv', '_group_graph.json')
    save_group_graph_json(framework.group_graph, filtered_metadata, json_path)

    png_path = csv_path.replace('.csv', '_last_frame.png')
    save_expert_trajectory_png(png_path, data['bg'], export_df)

    grouped = export_df[export_df['group_id'] != -1]
    num_groups = grouped['group_id'].nunique()
    if num_groups > 0:
        print(f'  [Groups] {num_groups} groups, {grouped["agent_id"].nunique()}/{num_agents} grouped')


# ---------------------------------------------------------------------------
# Queue crossing filter
# ---------------------------------------------------------------------------

def _segments_intersect(p1, d1, l1, p2, d2, l2) -> bool:
    """Return True if two directed line segments intersect (2D)."""
    # Segment A: p1 → p1 + d1*l1,  Segment B: p2 → p2 + d2*l2
    ax1, ay1 = p1
    ax2, ay2 = p1[0] + d1[0] * l1, p1[1] + d1[1] * l1
    bx1, by1 = p2
    bx2, by2 = p2[0] + d2[0] * l2, p2[1] + d2[1] * l2

    def cross2d(ox, oy, ax, ay, bx, by):
        return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)

    d = cross2d(ax1, ay1, ax2, ay2, bx1, by1) * cross2d(ax1, ay1, ax2, ay2, bx2, by2)
    e = cross2d(bx1, by1, bx2, by2, ax1, ay1) * cross2d(bx1, by1, bx2, by2, ax2, ay2)
    return d < 0 and e < 0


def filter_crossing_queues(queues: list) -> list:
    """
    Remove crossing queue lines, keeping the longer one of each crossing pair.

    Iterates greedily: add each queue to the accepted set only if it doesn't
    cross any already-accepted queue. Queues sorted by walkable_length_px
    descending so the longest lines are kept.
    """
    if len(queues) <= 1:
        return queues

    sorted_q = sorted(queues, key=lambda q: q.get('walkable_length_px', 0), reverse=True)
    accepted = []

    for q in sorted_q:
        gp  = q['goal_point']
        d   = q['direction']
        l   = q.get('walkable_length_px', 0)
        crosses = False
        for aq in accepted:
            agp = aq['goal_point']
            ad  = aq['direction']
            al  = aq.get('walkable_length_px', 0)
            if _segments_intersect(gp, d, l, agp, ad, al):
                crosses = True
                break
        if not crosses:
            accepted.append(q)

    return accepted


# ---------------------------------------------------------------------------
# Per-scene processing
# ---------------------------------------------------------------------------

def process_scene(scene: str, scene_idx: int,
                  dataset: Optional[SyntheticDataset],
                  framework: Optional[CrowdESFramework],
                  dataset_root: str,
                  out_dir: str,
                  unified_config_path: str,
                  args: argparse.Namespace) -> bool:
    """
    Process one scene: detect food cart placements, validate queues, run CrowdES.

    Returns True on success (at least one valid queue generated).
    """
    csv_out = f'{out_dir}/{scene}-{POSTFIX}-{FC_SUFFIX}.csv'
    if os.path.exists(csv_out) and not args.force:
        print(f'[{scene}] Skip (output exists): {csv_out}')
        return True

    # Load walkable mask
    try:
        walkable_mask = _load_walkable(scene, dataset_root)
    except FileNotFoundError as e:
        print(f'[{scene}] Skip: {e}')
        return False

    meta = _load_meta(scene, dataset_root)
    pixel_per_meter = float(meta['pixel_per_meter'])

    # Detect placement candidates
    candidates = detect_placement_candidates(
        walkable_mask,
        pixel_per_meter,
        search_radius_m=args.search_radius_m,
        max_candidates=args.max_food_carts,
    )
    print(f'[{scene}] {len(candidates)} placement candidates detected')

    if not candidates:
        print(f'[{scene}] No candidates; skipping.')
        return False

    # Compute queue anchor for each candidate
    raw_queues = []
    gap_tolerance_m = 0.4
    for cart_idx, (cx, cy) in enumerate(candidates):
        anchor = compute_best_queue_anchor(
            cx, cy,
            walkable_mask,
            pixel_per_meter,
            min_length_m=args.min_queue_length_m,
            gap_tolerance_m=gap_tolerance_m,
            queue_id=f'fc_{cart_idx}',
        )
        if anchor is None:
            print(f'[{scene}] anchor {cart_idx}: no walkable corridor >= {args.min_queue_length_m}m')
            continue

        wl_m = anchor['walkable_length_px'] / pixel_per_meter
        angle = anchor['_angle_deg']
        print(f'[{scene}] anchor {cart_idx}: dir={angle:.1f}deg, walkable={wl_m:.1f}m -> VALID')
        raw_queues.append(anchor)

    if not raw_queues:
        print(f'[{scene}] No valid anchors found; skipping.')
        return False

    # Remove crossing queue lines (keep longer ones)
    raw_queues = filter_crossing_queues(raw_queues)
    print(f'[{scene}] {len(raw_queues)} queues after crossing filter')

    # Validate via scenario_validator
    with open(unified_config_path, 'r') as f:
        unified_config = yaml.safe_load(f)

    # Use relaxed queue_config as specified in the task
    relaxed_queue_config = dict(unified_config.get('queue', {}))
    relaxed_queue_config['safety_margin_ratio']     = args.safety_margin
    relaxed_queue_config['min_length_meters']       = args.min_queue_length_m
    relaxed_queue_config['scan_gap_tolerance_meters'] = gap_tolerance_m

    candidate_metadata = {
        'queue':    raw_queues,
        'gathering': [],
        'crosswalk': [],
    }

    validated = validate_scenario_points(
        candidate_metadata,
        walkable_mask,
        pixel_per_meter,
        unified_config.get('gathering', {}),
        relaxed_queue_config,
    )

    valid_queues = validated.get('queue', [])
    print(f'[{scene}] {len(valid_queues)} valid queues after validation', end='')

    if not valid_queues:
        print(' -> skipping.')
        return False

    print(' -> running CrowdES' if not args.dry_run else ' -> dry_run, skipping CrowdES.')

    # Build food cart world coordinates from validated anchors
    static_food_carts = []
    for q in valid_queues:
        cart_ix = q.get('_cart_img_x')
        cart_iy = q.get('_cart_img_y')
        fwd_x   = q.get('_cart_fwd_x', 0.0)
        fwd_y   = q.get('_cart_fwd_y', 1.0)

        if cart_ix is None or cart_iy is None:
            continue

        wx, wy, wz = _pixel_to_world(int(cart_ix), int(cart_iy), meta)

        # CARLA forward vector from image-space direction:
        #   image dir [fwd_x, fwd_y] -> CARLA fwd = (-fwd_y, fwd_x)
        carla_fwd_x = -fwd_y
        carla_fwd_y =  fwd_x
        yaw_deg = math.degrees(math.atan2(carla_fwd_y, carla_fwd_x))

        static_food_carts.append({
            'queue_id': q.get('queue_id', 'fc_unknown'),
            'world_x': wx,
            'world_y': wy,
            'world_z': wz,
            'yaw_deg': yaw_deg,
            'img_x':   int(cart_ix),
            'img_y':   int(cart_iy),
        })

    # Strip internal underscore keys before saving meta
    clean_queues = []
    for q in valid_queues:
        clean_q = {k: v for k, v in q.items() if not k.startswith('_')}
        clean_queues.append(clean_q)

    # Keep existing gathering/crosswalk metadata from original scene
    existing_meta = load_scenario_metadata(scene, dataset_root)
    new_scenarios = {
        'queue':     clean_queues,
        'gathering': existing_meta.get('gathering', []),
        'crosswalk': existing_meta.get('crosswalk', []),
        'cohesive':  [],
    }

    meta_fc_path = _save_meta_fc(scene, dataset_root, new_scenarios, static_food_carts)
    print(f'[{scene}] Saved meta -> {meta_fc_path}')

    if args.dry_run:
        return True

    # Run CrowdES
    scenario_metadata_for_crowdes = dict(new_scenarios)
    _run_crowdes(scene, dataset, framework, scene_idx,
                 scenario_metadata_for_crowdes, unified_config_path,
                 out_dir, FC_SUFFIX)

    csv_path = f'{out_dir}/{scene}-{POSTFIX}-{FC_SUFFIX}.csv'
    print(f'[{scene}] Saved -> {csv_path}')
    return True


# ---------------------------------------------------------------------------
# Target scene selection
# ---------------------------------------------------------------------------

def _has_empty_queue(scene: str, dataset_root: str) -> bool:
    """Return True if this scene has no queue scenarios in its _meta.json."""
    meta = load_scenario_metadata(scene, dataset_root)
    return len(meta.get('queue', [])) == 0


def gather_target_scenes(args: argparse.Namespace, dataset_root: str) -> list:
    """Collect scenes that need food-cart queue injection."""
    if args.scenes:
        scenes = list(args.scenes)
        print(f'[generate_foodcart_queues] Manual scene list: {len(scenes)} scenes')
        return scenes

    # Auto-discover by town filter
    town_filter = None if args.town == 'all' else args.town

    # Restrict to town03 / town05 by default unless overridden
    if town_filter is not None:
        all_scenes = discover_scenes_from_dataset(dataset_root, town_filter)
    else:
        scenes_03 = discover_scenes_from_dataset(dataset_root, 'town03')
        scenes_05 = discover_scenes_from_dataset(dataset_root, 'town05')
        all_scenes = scenes_03 + scenes_05

    # Keep only scenes with empty queue
    target = [s for s in all_scenes if _has_empty_queue(s, dataset_root)]

    # Summary by town
    counts = {}
    for s in target:
        for part in ['town03', 'town05', 'town10']:
            if part in s:
                counts[part] = counts.get(part, 0) + 1
                break

    counts_str = ', '.join(f'{t}: {c}' for t, c in sorted(counts.items()))
    print(f'[generate_foodcart_queues] Targeting {len(target)} scenes ({counts_str})')
    return target


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Food cart virtual placement for scenes missing queue scenarios'
    )
    parser.add_argument('--town', default='all',
                        help='Town filter: all / town03 / town05')
    parser.add_argument('--scenes', nargs='*', default=None,
                        help='Specific scene names (overrides --town)')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory (default: output/generated/synthetic/<subset>_fc)')
    parser.add_argument('--subset', default='topo_ft',
                        choices=['topo_ft', 'standard_ft'],
                        help='Which subset is used to determine target scenes')
    parser.add_argument('--config', default='configs/scenarios_long_queue.yaml',
                        help='Unified scenarios config path (default: long_queue → gathering disabled)')
    parser.add_argument('--min_queue_length_m', type=float, default=1.5,
                        help='Minimum queue walkable length (meters)')
    parser.add_argument('--safety_margin', type=float, default=0.70,
                        help='Queue safety margin ratio for validator')
    parser.add_argument('--search_radius_m', type=float, default=4.0,
                        help='Circular kernel radius for convolution candidate detection (meters)')
    parser.add_argument('--max_food_carts', type=int, default=4,
                        help='Maximum food cart placements per scene')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing output files')
    parser.add_argument('--dry_run', action='store_true',
                        help='Detect candidates and validate but do not run CrowdES')
    args = parser.parse_args()

    reproducibility_settings(seed=42)

    dataset_root = os.path.normpath(os.path.join(_SCRIPT_DIR, DATASET_ROOT))

    if args.output_dir is None:
        out_dir = os.path.normpath(
            os.path.join(_SCRIPT_DIR, OUTPUT_BASE, f'{args.subset}_fc')
        )
    else:
        out_dir = os.path.normpath(args.output_dir)

    unified_config_path = args.config
    if not os.path.isabs(unified_config_path):
        unified_config_path = os.path.normpath(
            os.path.join(_SCRIPT_DIR, unified_config_path)
        )

    print(f'[generate_foodcart_queues] Dataset: {dataset_root}')
    print(f'[generate_foodcart_queues] Output:  {out_dir}')
    print(f'[generate_foodcart_queues] Config:  {unified_config_path}')
    print(f'[generate_foodcart_queues] Suffix:  {FC_SUFFIX}')

    # Collect target scenes
    target_scenes = gather_target_scenes(args, dataset_root)
    if not target_scenes:
        print('[generate_foodcart_queues] No target scenes found; exiting.')
        sys.exit(0)

    # Load CrowdES model (skip in dry_run to avoid GPU overhead)
    if not args.dry_run:
        config = get_config(MODEL_CONFIG)
        config.group_config = unified_config_path
        dataset_obj = SyntheticDataset(config, target_scenes, dataset_path=dataset_root)
        framework   = CrowdESFramework(config)
    else:
        print('[generate_foodcart_queues] Dry-run: CrowdES model NOT loaded.')
        dataset_obj = None
        framework   = None

    # Process each scene
    n_success = 0
    for scene_idx, scene in enumerate(target_scenes):
        try:
            ok = process_scene(
                scene, scene_idx,
                dataset_obj, framework,
                dataset_root, out_dir,
                unified_config_path, args,
            )
            if ok:
                n_success += 1
        except Exception as exc:
            print(f'[{scene}] ERROR: {exc}')
            import traceback
            traceback.print_exc()

    print(f'\n[Summary] {n_success}/{len(target_scenes)} scenes succeeded')


if __name__ == '__main__':
    main()
