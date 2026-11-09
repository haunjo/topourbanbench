"""
TopoUrbanBench Metrics Module

Group-structural compliance metrics for pedestrian navigation.

Metrics:
    - TopologicalMetricEngine: the metrics reported in the TopoUrbanBench paper (CoRL 2026)
        - QCR (Queue Cutting Rate): number of distinct queues cut at least once per episode
        - GDS (Gathering Disruption Score): seconds inside a gathering's O-space
        - CPI (Cohesive Proxemic Intrusion): seconds inside the R-space buffer around cohesive groups

Usage:
    from topourbanbench.metrics import TopologicalMetricEngine

    # Initialize metrics
    metrics = TopologicalMetricEngine(robot_radius=0.3, intrusion_threshold=0.8)

    # Training (RL)
    penalty = metrics.evaluate_frame(ego_loc, active_agents, group_graph)
    reward = -penalty

    # Evaluation (Benchmark)
    episode_metrics = metrics.get_episode_metrics()
    print(f"QCR: {episode_metrics['QCR']:.3f}")
    print(f"GDS: {episode_metrics['GDS']:.2f}s")
    print(f"CPI: {episode_metrics['CPI']:.2f}s")
"""
# metrics/__init__.py
from .metrics import TopologicalMetricEngine
