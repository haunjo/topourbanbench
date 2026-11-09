import os
import sys
import json
import zlib
import yaml
from pathlib import Path

# Topology utilities (shared module)
_TOPO_UTILS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if _TOPO_UTILS_DIR not in sys.path:
    sys.path.insert(0, _TOPO_UTILS_DIR)
from topourbanbench.utils.topology_utils import topology_to_edges

from CrowdES.inference_model import CrowdESFramework
from utils.dataloader.synthetic_dataloader import SyntheticDataset
from utils.utils import reproducibility_settings
from utils.visualization import save_expert_trajectory_png
from utils.expert_trajectory import ExpertAgentAdapter


def save_group_graph_json(group_graph, scenario_metadata, output_path):
    """Export group_graph to JSON following the topology schema.

    Args:
        group_graph: dict with keys 'groups', 'agent_to_group', 'scenario_types'
        scenario_metadata: dict with keys 'queue', 'gathering', 'crosswalk', etc.
        output_path: str path ending with _group_graph.json
    """
    groups_json = []

    for group in group_graph.get('groups', []):
        group_id = group['group_id']
        group_type = group.get('group_type', 'cohesive')
        scenario_idx = group.get('scenario_idx')
        member_ids = group.get('member_ids', [])

        # Resolve topology string from group_type
        from topourbanbench.utils.topology_utils import TOPOLOGY_FOR_GROUP_TYPE
        topology = TOPOLOGY_FOR_GROUP_TYPE.get(group_type, 'clique')

        # Compute edge list using local member indices
        edges = topology_to_edges(topology, len(member_ids))

        # Build type-specific metadata and trigger
        type_metadata = {}
        trigger = None

        if group_type == 'queue':
            queue_list = scenario_metadata.get('queue', [])
            if scenario_idx is not None and scenario_idx < len(queue_list):
                q_info = queue_list[scenario_idx]
                type_metadata['head_agent_id'] = member_ids[0] if member_ids else None
                type_metadata['tail_agent_id'] = member_ids[-1] if member_ids else None
                type_metadata['queue_direction'] = q_info.get('direction', None)

        elif group_type == 'gathering':
            gather_list = scenario_metadata.get('gathering', [])
            if scenario_idx is not None and scenario_idx < len(gather_list):
                g_info = gather_list[scenario_idx]
                type_metadata['center'] = g_info.get('center', None)

        elif group_type == 'crosswalk':
            cw_list = scenario_metadata.get('crosswalk', [])
            if scenario_idx is not None and scenario_idx < len(cw_list):
                cw_info = cw_list[scenario_idx]
                type_metadata['flow_direction_vec'] = cw_info.get('crossing_direction', None)
                zone_center = cw_info.get('zone_center', cw_info.get('center', None))
                zone_radius = cw_info.get('zone_radius', None)
                if zone_center is not None:
                    trigger = {
                        'type': 'proximity',
                        'zone_center': zone_center,
                        'zone_radius': zone_radius,
                    }

        group_entry = {
            'group_id': group_id,
            'group_type': group_type,
            'topology': topology,
            'member_ids': member_ids,
            'edges': edges,
            'trigger': trigger,
            'metadata': type_metadata,
        }
        groups_json.append(group_entry)

    # Summary counts
    type_counts = {}
    for g in groups_json:
        t = g['group_type']
        type_counts[t] = type_counts.get(t, 0) + 1

    output_doc = {
        'summary': {
            'num_queues': type_counts.get('queue', 0),
            'num_gatherings': type_counts.get('gathering', 0),
            'num_crosswalks': type_counts.get('crosswalk', 0),
            'num_cohesive': type_counts.get('cohesive', 0),
            'total_groups': len(groups_json),
        },
        'groups': groups_json,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_doc, f, indent=2)
    print(f'[GroupGraph] Saved {len(groups_json)} groups to {output_path}')


# ============================================================
# Global Configuration
# (the trainval.py --synthetic options in parentheses override these defaults)
# ============================================================
TRIALS = 1                      # Number of trials for each scene
SCENARIO_LENGTH = 20 * 60 * 2   # Scenario length in frames (20fps * 60s = 2 minutes)
POSTFIX = 'crowdes-synthetic'   # Postfix for generated files
EXPORT_VIDEO = False            # Video disabled; saves expert trajectory PNG instead
EXPERT_ENABLED = True           # Inject expert agent for IL dataset
RANDOMIZE_PARAMS = False        # Randomize group parameters per scene (--randomize)
RANDOMIZE_SEED = 42             # Base seed for parameter randomization
FORCE_REGEN = False             # Regenerate scenes whose CSV already exists (--force)

# Unified config path
UNIFIED_CONFIG = 'configs/scenarios.yaml'

# Dataset and output paths (--dataset_root, --out_subset)
DATASET_ROOT = './datasets/Synthetic'
OUT_SUBSET = None               # None: detected from configs/scenarios.yaml

# ============================================================
# Scene List Configuration
# ============================================================
# Set to None for auto-discovery, or specify list of scenes (--scene)
SCENE_LIST = None

# Town filter, only used when SCENE_LIST is None (--town)
TOWN_FILTER = None


def load_scenario_metadata(scene_name: str, dataset_root: str = './datasets/Synthetic'):
    """
    Load scenario metadata from unified _meta.json.

    Returns:
        Dict with keys ['queue', 'cohesive', 'gathering', 'crosswalk']
        Each value is a list of location dicts, or empty list if not found.
    """
    default = {'queue': [], 'cohesive': [], 'gathering': [], 'crosswalk': []}

    meta_path = Path(dataset_root) / 'metadata' / f'{scene_name}_meta.json'
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            loaded = json.load(f)
        # Unified format: scenarios nested under 'scenarios' key
        if 'scenarios' in loaded:
            scenarios = loaded['scenarios']
            default.update(scenarios)
            return default
        # Legacy format: scenarios at top level
        default.update(loaded)
        return default

    return default


def discover_scenes_from_dataset(dataset_root: str, town_filter=None):
    """
    Auto-discover scenes from dataset directory.

    Args:
        dataset_root: Path to dataset (e.g., './datasets/Synthetic')
        town_filter: Filter by town name (e.g., 'town10'), or None for all

    Returns:
        List of scene names
    """
    metadata_dir = Path(dataset_root) / 'metadata'
    if not metadata_dir.exists():
        print(f"[Warning] Metadata directory not found: {metadata_dir}")
        return []

    scenes = []
    for meta_file in sorted(metadata_dir.glob('*_meta.json')):
        scene_name = meta_file.stem.replace('_meta', '')

        # Apply town filter
        if town_filter and town_filter.lower() not in scene_name.lower():
            continue

        # Check required files exist
        bg_file = Path(dataset_root) / 'image_terrain' / f'{scene_name}_bg.png'
        walkable_file = Path(dataset_root) / 'gt' / f'{scene_name}_walkable.png'

        if bg_file.exists() and walkable_file.exists():
            scenes.append(scene_name)
        else:
            print(f"[Warning] Incomplete scene: {scene_name}")

    return scenes


def main(config, seed=0):
    # Reproducibility
    reproducibility_settings(seed=seed)

    # Determine scene list
    # Resolve dataset root: DATASET_ROOT global (relative → abs relative to crowdes dir)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isabs(DATASET_ROOT):
        resolved_dataset_root = DATASET_ROOT
    else:
        resolved_dataset_root = os.path.normpath(os.path.join(base_dir, DATASET_ROOT))

    if SCENE_LIST is None:
        # Auto-discover scenes
        dataset_root = resolved_dataset_root
        discovered_scenes = discover_scenes_from_dataset(dataset_root, TOWN_FILTER)

        if len(discovered_scenes) == 0:
            print(f"[Error] No scenes found in {dataset_root}")
            print(f"Town filter: {TOWN_FILTER}")
            return

        scene_list_to_use = discovered_scenes
        print(f"\n{'='*80}")
        print(f"[Auto-Discovery] Found {len(discovered_scenes)} scenes")
        if TOWN_FILTER:
            print(f"[Filter] Town: {TOWN_FILTER}")
        print(f"{'='*80}\n")
    else:
        # Use manually specified list
        dataset_root = resolved_dataset_root
        scene_list_to_use = SCENE_LIST
        print(f"\n{'='*80}")
        print(f"[Manual Mode] Using {len(SCENE_LIST)} specified scenes")
        print(f"{'='*80}\n")

    # Load dataset and framework
    dataset_test = SyntheticDataset(config, scene_list_to_use, dataset_path=dataset_root)
    # Enable group behavior (queue/gathering/cohesive) by pointing to unified scenario config
    config.group_config = UNIFIED_CONFIG
    framework = CrowdESFramework(config)

    # Start inference
    scene_list = dataset_test.scene_list

    # Pre-detect subset for skip check (same logic used inside the loop)
    with open(UNIFIED_CONFIG, 'r') as _f:
        _pre_cfg = yaml.safe_load(_f)
    _pre_enabled = [k for k, v in _pre_cfg['scenarios'].items() if v['enabled']]
    _pre_subset = OUT_SUBSET if OUT_SUBSET else ('topo_ft' if _pre_enabled else 'standard_ft')

    for scene_idx, scene in enumerate(scene_list):
        for trial in range(TRIALS):
            # Skip if output CSV already exists (resume support), unless --force
            _csv_check = f'../../../output/generated/synthetic/{_pre_subset}/{scene}-{POSTFIX}-{trial}.csv'
            if os.path.exists(_csv_check) and not FORCE_REGEN:
                print(f'Scene {scene_idx + 1}/{len(scene_list)}: {scene} — already exists, skipping.')
                continue

            print(f'Scene {scene_idx + 1}/{len(scene_list)}: {scene}, Trial {trial + 1}/{TRIALS}')

            # Load data
            data = dataset_test[scene_idx]
            scene_img = data['img']
            scene_seg = data['seg']
            scene_H = data['H']
            scene_appearance = data['appearance']
            scene_population = data['population']
            scene_scenario = data['scenario']
            scene_walkable = data['walkable']
            scene_navmesh = data['navmesh']

            # Load unified config
            with open(UNIFIED_CONFIG, 'r') as f:
                unified_config = yaml.safe_load(f)

            # Load scenario metadata (all types)
            dataset_root = dataset_test.dataset_path
            scenario_metadata = load_scenario_metadata(scene, dataset_root)

            # Filter scenario_metadata to only include enabled scenarios
            enabled_scenarios = [k for k, v in unified_config['scenarios'].items() if v['enabled']]
            subset = 'topo_ft' if enabled_scenarios else 'standard_ft'
            print(f'[Scenarios] Enabled: {enabled_scenarios}  →  subset={subset}')
            scenario_metadata = {k: (v if k in enabled_scenarios else [])
                                 for k, v in scenario_metadata.items()}

            # Log loaded metadata summary
            meta_summary = {k: len(v) for k, v in scenario_metadata.items() if len(v) > 0}
            if meta_summary:
                print(f'  [Metadata] {meta_summary}')

            # Enable per-scene parameter randomization for domain diversity (topo_ft)
            if RANDOMIZE_PARAMS:
                framework.set_randomize_params(True)

            # Inference
            scenario_length = SCENARIO_LENGTH
            # Group parameters are sampled in initialize_scene(). Seeding by scene name makes a scene come out the
            # same whether it is generated alone or together with other scenes.
            reproducibility_settings(seed=(RANDOMIZE_SEED + zlib.crc32(f'{scene}-{trial}'.encode())) % 2**32)
            framework.initialize_scene(scene_img, scene_seg, scene_walkable, scene_navmesh, scene_H,
                                     appearance_gt=scene_appearance, population_gt=scene_population,
                                     scenario_gt=scene_scenario,
                                     scenario_metadata=scenario_metadata)
            generated_scenario = framework.generate(scenario_length, seed=trial)
            generated_scenario['scene'] = scene
            num_agents = len(generated_scenario["agent_id"].unique())
            print(f'Generated scenario: {num_agents} total agents')

            # NOTE: Group detection already done in inference_model.py (frame-local)
            # generated_scenario already contains group_id column from framework.generate()

            # Inject expert agent for IL dataset
            if EXPERT_ENABLED:
                generated_scenario = ExpertAgentAdapter.inject(
                    generated_scenario, scene, framework=framework
                )

            # Save generated scenario (includes group_id and group_type from inference)
            columns = ['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y', 'group_id', 'group_type']
            generated_scenario_export = generated_scenario[columns].copy()
            effective_subset = OUT_SUBSET if OUT_SUBSET else subset
            out_dir = f'../../../output/generated/synthetic/{effective_subset}'
            os.makedirs(out_dir, exist_ok=True)
            csv_path = f'{out_dir}/{scene}-{POSTFIX}-{trial}.csv'
            generated_scenario_export.to_csv(csv_path, index=False)

            # Export group graph JSON alongside CSV
            json_path = csv_path.replace('.csv', '_group_graph.json')
            save_group_graph_json(framework.group_graph, scenario_metadata, json_path)

            # Print group statistics from generated scenario
            grouped_df = generated_scenario_export[generated_scenario_export['group_id'] != -1]
            num_groups = grouped_df['group_id'].nunique()
            if num_groups > 0:
                num_grouped = grouped_df['agent_id'].nunique()
                print(f'[Groups] {num_groups} groups detected, {num_grouped}/{num_agents} agents grouped')

            # Export expert trajectory PNG
            scene_bg = data['bg']
            png_path = f'{out_dir}/{scene}-{POSTFIX}-{trial}_last_frame.png'
            save_expert_trajectory_png(png_path, scene_bg, generated_scenario_export)

    return 
