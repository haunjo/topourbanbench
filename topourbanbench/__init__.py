"""
TopoUrbanBench: Group-Structural Compliance Evaluation for Urban Robot Navigation

Pedestrian social groups (queues, gatherings, cohesive walking groups) are generated
as intention-driven graphs, and three archetype-specific metrics measure how a robot
trajectory intersects them.

Modules:
    - generation: CARLA map extraction and CrowdES-based crowd generation with
      social groups and topology-corrected expert trajectories
    - metrics: group-structural compliance metrics (QCR, GDS, CPI)
    - utils: group-graph topology helpers
    - envs: closed-loop CARLA environments (released with the evaluation benchmark)

Usage:
    from topourbanbench.metrics import TopologicalMetricEngine

References:
    CrowdES: Continuous Locomotive Crowd Behavior Generation, CVPR 2025
"""

__version__ = '0.1.0'

# Lazy imports for faster package loading
def _lazy_import():
    pass

__all__ = ['envs', 'metrics', 'generation']
