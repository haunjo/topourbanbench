"""
CrowdES Core Implementation

This package contains the original CrowdES codebase for:
- Diffusion-based crowd emitter
- State-switching trajectory simulator
- Group behavior generation

Structure:
    - CrowdES/: Model implementations (emitter, simulator, group_graph)
    - utils/: Data loading, preprocessing, metrics
    - configs/: Model and training configurations
    - trainval.py: Training and evaluation script
    - extract_carla_maps.py: CARLA scenario generation

Note:
    Run the pipeline through scripts/generate_carla_scenarios.sh and
    scripts/generate_crowdes_trajectories.sh.
"""

__all__ = ['CrowdES', 'utils', 'configs']
