"""
generate_diversity_groups.py — Extra scene variants with another seed or group preset.

Modes (each uses its own config in configs/):
  standard        — scenarios_eval_seed0.yaml: the archetypes of scenarios.yaml with small changes to
                    queue spacing and gathering detection, so the layout differs from the main run.
  long_queue      — scenarios_long_queue.yaml: longer queues, gatherings off. Add --fc to also use the
                    food-cart anchors in <scene>_meta_fc.json (see generate_foodcart_queues.py).
  large_gathering — scenarios_large_gathering.yaml: larger gatherings found with a wider search radius,
                    queues off.
  dense_mixed     — scenarios_dense_mixed.yaml: small gatherings, mid-size queues, larger cohesive groups.

Each mode writes <scene>-crowdes-synthetic-N.csv with its own default N (standard 1, long_queue 2,
large_gathering 3, dense_mixed 4), so the modes never overwrite each other or the main run (N = 0).
--suffix sets another N. N is also the CrowdES seed. The scenario anchors of each run are saved next to
its CSV as <scene>-crowdes-synthetic-N_meta.json.

Run from topourbanbench/generation/crowdes/ (without --scenes or --town, every scene in datasets/Synthetic/):
    python generate_diversity_groups.py --mode standard
    python generate_diversity_groups.py --mode long_queue
    python generate_diversity_groups.py --mode large_gathering --town town10
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np
import yaml
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: allow imports identical to evaluate_synthetic_dataset.py
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
DEFAULT_SUFFIX = {'standard': 1, 'long_queue': 2, 'large_gathering': 3, 'dense_mixed': 4}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_walkable(scene: str, dataset_root: str) -> np.ndarray:
    """Load walkable PNG as uint8 binary mask (>127 → 1)."""
    path = Path(dataset_root) / 'gt' / f'{scene}_walkable.png'
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'Walkable mask not found: {path}')
    return (img > 127).astype(np.uint8)


def _walkable_ratio(walkable_mask: np.ndarray) -> float:
    return float(walkable_mask.sum()) / walkable_mask.size


def _load_pixel_per_meter(scene: str, dataset_root: str) -> float:
    meta_path = Path(dataset_root) / 'metadata' / f'{scene}_meta.json'
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return float(meta['pixel_per_meter'])


def _merge_fc_queues(scenario_metadata: dict, scene: str, dataset_root: str) -> dict:
    """Merge food-cart queue anchors from _meta_fc.json into scenario_metadata."""
    fc_path = Path(dataset_root) / 'metadata' / f'{scene}_meta_fc.json'
    if not fc_path.exists():
        return scenario_metadata
    with open(fc_path) as f:
        fc_data = json.load(f)
    fc_scenarios = fc_data.get('scenarios', fc_data)
    fc_queues = fc_scenarios.get('queue', [])
    if fc_queues:
        merged = dict(scenario_metadata)
        merged['queue'] = scenario_metadata.get('queue', []) + fc_queues
        print(f'  +{len(fc_queues)} food-cart queue(s) from _meta_fc.json '
              f'(total: {len(merged["queue"])})')
        return merged
    return scenario_metadata


def _save_scene_meta(scene: str, dataset_root: str, new_scenarios: dict, csv_out: str) -> str:
    """Copy _meta.json with the scenarios of this run, saved next to the output CSV as <csv stem>_meta.json."""
    meta_path = Path(dataset_root) / 'metadata' / f'{scene}_meta.json'
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    # Keep all top-level fields; replace scenarios
    meta['scenarios'] = new_scenarios

    out_path = csv_out.replace('.csv', '_meta.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(meta, f, indent=4)
    return out_path


def _detect_gatherings(walkable_mask: np.ndarray, pixel_per_meter: float,
                        config: dict) -> list:
    """
    Run the same 2D-convolution gathering detection as UnifiedScenarioGenerator,
    using parameters from config['gathering'].
    Returns list of gathering dicts (without spawn_radius; call validate afterward).
    """
    from scipy.ndimage import maximum_filter
    from scipy.signal import fftconvolve

    H, W = walkable_mask.shape
    g_cfg = config['gathering']

    search_radius_m = g_cfg.get('search_radius_meters', 2.5)
    min_coverage = g_cfg.get('min_coverage_ratio', 0.6)
    max_gatherings = g_cfg.get('max_gathering_points', 5)
    min_distance_m = g_cfg.get('min_distance_meters', 6.0)

    search_radius_px = int(search_radius_m * pixel_per_meter)
    min_distance_px = int(min_distance_m * pixel_per_meter)

    y_grid, x_grid = np.ogrid[
        -search_radius_px:search_radius_px + 1,
        -search_radius_px:search_radius_px + 1,
    ]
    kernel = (x_grid ** 2 + y_grid ** 2 <= search_radius_px ** 2).astype(float)
    kernel_area = kernel.sum()

    walkable_float = (walkable_mask > 0).astype(float)
    coverage_map = fftconvolve(walkable_float, kernel, mode='same')
    coverage_ratio = coverage_map / kernel_area

    valid_mask = coverage_ratio >= min_coverage
    local_max_map = maximum_filter(coverage_map, size=min_distance_px)
    is_local_max = (coverage_map == local_max_map) & valid_mask

    candidates_y, candidates_x = np.where(is_local_max)
    candidates_coverage = coverage_ratio[candidates_y, candidates_x]
    sorted_idx = np.argsort(candidates_coverage)[::-1]
    candidates_y = candidates_y[sorted_idx]
    candidates_x = candidates_x[sorted_idx]
    candidates_coverage = candidates_coverage[sorted_idx]

    selected = []
    for i in range(len(candidates_y)):
        if len(selected) >= max_gatherings:
            break
        cy, cx = int(candidates_y[i]), int(candidates_x[i])
        cov = float(candidates_coverage[i])

        too_close = any(
            np.sqrt((cy - py) ** 2 + (cx - px) ** 2) < min_distance_px
            for py, px, _ in selected
        )
        if not too_close:
            selected.append((cy, cx, cov))

    metadata = []
    for cy, cx, cov in selected:
        metadata.append({
            'type': 'gathering',
            'source': 'convolution',
            'center': [cx, cy],
            'pixel_per_meter': pixel_per_meter,
            'coverage_ratio': cov,
        })
    return metadata


def _run_crowdes(scene: str, dataset: SyntheticDataset, framework: CrowdESFramework,
                 scene_idx: int, scenario_metadata: dict, unified_config_path: str,
                 out_dir: str, suffix: int, no_expert: bool = False) -> None:
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

    if not no_expert:
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
# Mode: long_queue
# ---------------------------------------------------------------------------

def run_long_queue(args, scenes: list, dataset_root: str, out_dir: str,
                   framework: CrowdESFramework, dataset: SyntheticDataset,
                   unified_config_path: str) -> None:
    for scene_idx, scene in enumerate(scenes):
        csv_out = f'{out_dir}/{scene}-{POSTFIX}-{args.suffix}.csv'
        if os.path.exists(csv_out) and not args.force:
            print(f'[{scene_idx+1}/{len(scenes)}] {scene} — skip (exists)')
            continue
        print(f'[{scene_idx+1}/{len(scenes)}] {scene}')

        scenario_metadata = load_scenario_metadata(scene, dataset_root)

        if args.fc:
            scenario_metadata = _merge_fc_queues(scenario_metadata, scene, dataset_root)

        # Save the scenarios of this run (queue positions unchanged)
        new_scenarios = {
            'queue': scenario_metadata.get('queue', []),
            'gathering': [],
            'crosswalk': scenario_metadata.get('crosswalk', []),
            'cohesive': [],
        }
        scene_meta_path = _save_scene_meta(scene, dataset_root, new_scenarios, csv_out)
        print(f'  Saved {scene_meta_path}')

        _run_crowdes(scene, dataset, framework, scene_idx, scenario_metadata,
                     unified_config_path, out_dir, args.suffix,
                     no_expert=args.no_expert)


# ---------------------------------------------------------------------------
# Mode: large_gathering
# ---------------------------------------------------------------------------

def run_large_gathering(args, scenes: list, dataset_root: str, out_dir: str,
                        framework: CrowdESFramework, dataset: SyntheticDataset,
                        unified_config_path: str) -> None:
    with open(unified_config_path, 'r') as f:
        unified_config = yaml.safe_load(f)

    for scene_idx, scene in enumerate(scenes):
        csv_out = f'{out_dir}/{scene}-{POSTFIX}-{args.suffix}.csv'
        if os.path.exists(csv_out) and not args.force:
            print(f'[{scene_idx+1}/{len(scenes)}] {scene} — skip (exists)')
            continue
        print(f'[{scene_idx+1}/{len(scenes)}] {scene}')

        try:
            walkable_mask = _load_walkable(scene, dataset_root)
        except FileNotFoundError as e:
            print(f'  Skip: {e}')
            continue

        ratio = _walkable_ratio(walkable_mask)
        if ratio < args.walkable_min:
            print(f'  Skip: walkable ratio {ratio:.3f} < {args.walkable_min}')
            continue
        print(f'  Walkable ratio: {ratio:.3f}')

        pixel_per_meter = _load_pixel_per_meter(scene, dataset_root)

        # Detect gatherings with large_gathering config parameters
        raw_gatherings = _detect_gatherings(walkable_mask, pixel_per_meter, unified_config)
        print(f'  Raw gathering candidates: {len(raw_gatherings)}')

        # Load existing queue metadata (queue positions kept as-is)
        existing_meta = load_scenario_metadata(scene, dataset_root)
        candidate_metadata = {
            'queue': existing_meta.get('queue', []),
            'gathering': raw_gatherings,
            'crosswalk': existing_meta.get('crosswalk', []),
        }

        # Validate: compute adaptive spawn_radius for each gathering
        validated = validate_scenario_points(
            candidate_metadata,
            walkable_mask,
            pixel_per_meter,
            unified_config.get('gathering', {}),
            unified_config.get('queue', {}),
        )
        print(f'  Validated gatherings: {len(validated["gathering"])}')

        if len(validated['gathering']) == 0:
            print('  No valid gatherings; skipping scene.')
            continue

        # Save the scenarios of this run
        new_scenarios = {
            'queue': validated['queue'],
            'gathering': validated['gathering'],
            'crosswalk': validated.get('crosswalk', []),
            'cohesive': [],
        }
        scene_meta_path = _save_scene_meta(scene, dataset_root, new_scenarios, csv_out)
        print(f'  Saved {scene_meta_path}')

        _run_crowdes(scene, dataset, framework, scene_idx, validated,
                     unified_config_path, out_dir, args.suffix,
                     no_expert=args.no_expert)


# ---------------------------------------------------------------------------
# Mode: dense_mixed
# ---------------------------------------------------------------------------

def run_dense_mixed(args, scenes: list, dataset_root: str, out_dir: str,
                    framework: CrowdESFramework, dataset: SyntheticDataset,
                    unified_config_path: str) -> None:
    """Mixed mode: small gatherings (3~4) + mid queues (4~6) + larger cohesive (3~4).

    Re-detects gatherings with smaller search radius (tight spots) and keeps
    existing queue positions — both types enabled simultaneously.
    """
    with open(unified_config_path, 'r') as f:
        unified_config = yaml.safe_load(f)

    for scene_idx, scene in enumerate(scenes):
        csv_out = f'{out_dir}/{scene}-{POSTFIX}-{args.suffix}.csv'
        if os.path.exists(csv_out) and not args.force:
            print(f'[{scene_idx+1}/{len(scenes)}] {scene} — skip (exists)')
            continue
        print(f'[{scene_idx+1}/{len(scenes)}] {scene}')

        try:
            walkable_mask = _load_walkable(scene, dataset_root)
        except FileNotFoundError as e:
            print(f'  Skip: {e}')
            continue

        ratio = _walkable_ratio(walkable_mask)
        if ratio < args.walkable_min:
            print(f'  Skip: walkable ratio {ratio:.3f} < {args.walkable_min}')
            continue
        print(f'  Walkable ratio: {ratio:.3f}')

        pixel_per_meter = _load_pixel_per_meter(scene, dataset_root)

        # Detect small gatherings using dense_mixed config parameters
        raw_gatherings = _detect_gatherings(walkable_mask, pixel_per_meter, unified_config)
        print(f'  Raw gathering candidates: {len(raw_gatherings)}')

        # Keep existing queue positions
        existing_meta = load_scenario_metadata(scene, dataset_root)
        candidate_metadata = {
            'queue': existing_meta.get('queue', []),
            'gathering': raw_gatherings,
            'crosswalk': existing_meta.get('crosswalk', []),
        }

        validated = validate_scenario_points(
            candidate_metadata,
            walkable_mask,
            pixel_per_meter,
            unified_config.get('gathering', {}),
            unified_config.get('queue', {}),
        )
        print(f'  Validated gatherings: {len(validated["gathering"])}  '
              f'queues: {len(validated["queue"])}')

        new_scenarios = {
            'queue': validated['queue'],
            'gathering': validated['gathering'],
            'crosswalk': validated.get('crosswalk', []),
            'cohesive': [],
        }
        scene_meta_path = _save_scene_meta(scene, dataset_root, new_scenarios, csv_out)
        print(f'  Saved {scene_meta_path}')

        _run_crowdes(scene, dataset, framework, scene_idx, validated,
                     unified_config_path, out_dir, args.suffix,
                     no_expert=args.no_expert)


# ---------------------------------------------------------------------------
# Mode: standard
# ---------------------------------------------------------------------------

def run_standard(args, scenes: list, dataset_root: str, out_dir: str,
                 framework: CrowdESFramework, dataset: SyntheticDataset,
                 unified_config_path: str) -> None:
    """Re-run CrowdES with all _meta.json positions + optional FC queues + different seed."""
    for scene_idx, scene in enumerate(scenes):
        csv_out = f'{out_dir}/{scene}-{POSTFIX}-{args.suffix}.csv'
        if os.path.exists(csv_out) and not args.force:
            print(f'[{scene_idx+1}/{len(scenes)}] {scene} — skip (exists)')
            continue
        print(f'[{scene_idx+1}/{len(scenes)}] {scene}')

        scenario_metadata = load_scenario_metadata(scene, dataset_root)

        if args.fc:
            scenario_metadata = _merge_fc_queues(scenario_metadata, scene, dataset_root)

        _run_crowdes(scene, dataset, framework, scene_idx, scenario_metadata,
                     unified_config_path, out_dir, args.suffix,
                     no_expert=args.no_expert)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Diversity-augmented CrowdES trajectory generation')
    parser.add_argument('--mode', required=True,
                        choices=['standard', 'long_queue', 'large_gathering', 'dense_mixed'])
    parser.add_argument('--suffix', type=int, default=None,
                        help='N of the output <scene>-crowdes-synthetic-N.csv, also the CrowdES seed '
                             '(default: standard 1, long_queue 2, large_gathering 3, dense_mixed 4)')
    parser.add_argument('--subset', default='topo_ft', choices=['topo_ft', 'standard_ft'])
    parser.add_argument('--config', default='configs/scenarios.yaml',
                        help='Base scenarios.yaml (overridden by mode-specific config if found)')
    parser.add_argument('--scenes', nargs='*', default=None)
    parser.add_argument('--town', default=None)
    parser.add_argument('--walkable_min', type=float, default=0.4,
                        help='large_gathering: skip scenes with walkable ratio below this')
    parser.add_argument('--fc', action='store_true',
                        help='Merge food-cart queue anchors from _meta_fc.json into metadata')
    parser.add_argument('--no_expert', action='store_true',
                        help='Do not add expert paths to the scene')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.suffix is None:
        args.suffix = DEFAULT_SUFFIX[args.mode]

    # Resolve mode-specific config
    mode_config_map = {
        'standard':        'configs/scenarios_eval_seed0.yaml',
        'long_queue':      'configs/scenarios_long_queue.yaml',
        'large_gathering': 'configs/scenarios_large_gathering.yaml',
        'dense_mixed':     'configs/scenarios_dense_mixed.yaml',
    }
    unified_config_path = mode_config_map[args.mode]
    if not os.path.exists(unified_config_path):
        print(f'[Warning] Mode config not found: {unified_config_path}, falling back to {args.config}')
        unified_config_path = args.config

    reproducibility_settings(seed=42)

    dataset_root = os.path.normpath(os.path.join(_SCRIPT_DIR, DATASET_ROOT))
    out_dir = os.path.normpath(os.path.join(_SCRIPT_DIR, OUTPUT_BASE, args.subset))

    # Discover / filter scenes
    if args.scenes:
        scenes = args.scenes
        print(f'[Manual] {len(scenes)} scenes specified')
    else:
        scenes = discover_scenes_from_dataset(dataset_root, args.town)
        if not scenes:
            print(f'[Error] No scenes found in {dataset_root} (town filter={args.town})')
            sys.exit(1)
        print(f'[Auto] {len(scenes)} scenes discovered (town={args.town})')

    print(f'Mode: {args.mode} | Suffix: {args.suffix} | Subset: {args.subset}')
    print(f'Config: {unified_config_path}')
    print(f'Output: {out_dir}')

    # Load CrowdES model
    config = get_config(MODEL_CONFIG)
    config.group_config = unified_config_path

    dataset = SyntheticDataset(config, scenes, dataset_path=dataset_root)
    framework = CrowdESFramework(config)

    if args.mode == 'standard':
        run_standard(args, scenes, dataset_root, out_dir, framework, dataset, unified_config_path)
    elif args.mode == 'long_queue':
        run_long_queue(args, scenes, dataset_root, out_dir, framework, dataset, unified_config_path)
    elif args.mode == 'large_gathering':
        run_large_gathering(args, scenes, dataset_root, out_dir, framework, dataset, unified_config_path)
    else:
        run_dense_mixed(args, scenes, dataset_root, out_dir, framework, dataset, unified_config_path)

    print('\nDone.')


if __name__ == '__main__':
    main()
