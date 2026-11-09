"""
TopoUrbanBench scene generation.

CrowdES-based crowd generation with ground-truth social groups.

Entry points (shell scripts in the repository's scripts/ directory):
    - generate_carla_scenarios.sh: extract walkable maps, navmeshes, and semantic
      anchors from CARLA (runs crowdes/extract_carla_maps.py; requires a CARLA server)
    - generate_crowdes_trajectories.sh: generate crowds, group graphs, and expert
      trajectories from extracted maps (runs crowdes/trainval.py --synthetic)

Archetype parameters (queues, gatherings, cohesive groups) are set in
crowdes/configs/scenarios.yaml.

Internal:
    - crowdes/: CrowdES core implementation (models, utils, configs)
"""

__all__ = []
