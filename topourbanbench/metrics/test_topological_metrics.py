"""
Test script for TopologicalMetricEngine

Validates QCR, GDS, CPI on synthetic scenarios that follow the paper's
definitions (Sec. 3, Algorithms 1-3):
  - QCR: number of queues whose member-to-member segments the robot crosses
         (each queue counted at most once per episode)
  - GDS: seconds inside any gathering O-space
         (disk at the member centroid, radius = nearest member distance)
  - CPI: seconds inside any cohesive R-space
         (1.2 m buffer outside the members' convex hull)
GDS and CPI count each frame at most once, even when groups overlap.

Run: python -m topourbanbench.metrics.test_topological_metrics
"""

import numpy as np
from topourbanbench.metrics.metrics import TopologicalMetricEngine

DT = 1.0  # 1 Hz robot trajectory, as in the paper


def _queue(group_id, member_ids):
    return {
        'group_id': group_id,
        'group_type': 'queue',
        'topology': 'directed_chain',
        'member_ids': member_ids,
        'edges': [[i, i + 1] for i in range(len(member_ids) - 1)],
    }


def _group(group_id, group_type, member_ids):
    return {'group_id': group_id, 'group_type': group_type, 'member_ids': member_ids}


def _ring(center_x, member_ids, radius=2.0):
    """Members evenly spaced on a circle centered at (center_x, 0)."""
    angles = np.linspace(0, 2 * np.pi, len(member_ids), endpoint=False)
    return {mid: [center_x + radius * np.cos(a), radius * np.sin(a)]
            for mid, a in zip(member_ids, angles)}


def _run(trajectory, active_agents, group_graph, dt=DT):
    metrics = TopologicalMetricEngine(robot_radius=0.3, intrusion_threshold=0.8, cpi_buffer=1.2)
    metrics.reset(group_graph)
    for ego_loc in trajectory:
        metrics.evaluate_frame(ego_loc, active_agents, group_graph, dt=dt)
    return metrics.get_episode_metrics()


def _check(name, value, expected):
    print(f"  {name}: {value:g} (expected: {expected:g})")
    assert abs(value - expected) < 1e-6, f"{name}: got {value}, expected {expected}"


def test_edge_proximity():
    """Test point-to-segment distance calculation"""
    print("\n" + "="*60)
    print("Test 1: Edge Proximity Calculation")
    print("="*60)

    metrics = TopologicalMetricEngine()

    test_cases = [
        # (point, segment_a, segment_b, expected_dist)
        ([0, 0], [1, 0], [2, 0], 1.0),  # Beyond segment start
        ([1.5, 1], [1, 0], [2, 0], 1.0),  # Perpendicular to middle
        ([0, 0], [0, 0], [2, 0], 0.0),  # On segment start
        ([1, 0], [0, 0], [2, 0], 0.0),  # On segment middle
    ]

    for point, a, b, expected in test_cases:
        _check(f"dist({point}, [{a}, {b}])", metrics._point_to_segment_distance(point, a, b), expected)

    print("  ✓ Test passed")


def test_queue_cutting():
    """QCR: a queue counts once it is cut, no matter how often (Algorithm 1)"""
    print("\n" + "="*60)
    print("Test 2: Queue Cutting Rate (QCR)")
    print("="*60)

    # Queue of 3 people at y=5, x = 0, 1, 2
    active_agents = {1: [0.0, 5.0], 2: [1.0, 5.0], 3: [2.0, 5.0]}
    group_graph = {'groups': [_queue(0, [1, 2, 3])]}

    cases = [
        ("passes 3 m away", [[0.5, 8.0], [1.5, 8.0]], 0),
        ("crosses between members 1-2", [[0.5, 4.0], [0.5, 6.0]], 1),
        ("crosses the same queue twice", [[0.5, 4.0], [0.5, 6.0], [1.5, 6.0], [1.5, 4.0]], 1),
        ("crosses the queue line beyond its tail", [[3.0, 4.0], [3.0, 6.0]], 0),
    ]
    for name, trajectory, expected in cases:
        _check(name, _run(trajectory, active_agents, group_graph)['QCR'], expected)

    print("  ✓ Test passed")


def test_multiple_queues():
    """QCR is a count over queues (0..K), not a fraction"""
    print("\n" + "="*60)
    print("Test 3: Multiple Queues")
    print("="*60)

    active_agents = {
        1: [0.0, 5.0], 2: [1.0, 5.0], 3: [2.0, 5.0],     # Queue 0 at y=5
        4: [0.0, 10.0], 5: [1.0, 10.0], 6: [2.0, 10.0],  # Queue 1 at y=10
    }
    group_graph = {'groups': [_queue(0, [1, 2, 3]), _queue(1, [4, 5, 6])]}

    _check("cuts queue 0 only",
           _run([[0.5, 4.0], [0.5, 6.0], [0.5, 8.0]], active_agents, group_graph)['QCR'], 1)
    _check("cuts both queues",
           _run([[0.5, 4.0], [0.5, 6.0], [0.5, 8.0], [0.5, 11.0]], active_agents, group_graph)['QCR'], 2)

    print("  ✓ Test passed")


def test_gathering_disruption():
    """GDS: dwell time in the O-space = disk at centroid, radius = nearest member (Algorithm 2)"""
    print("\n" + "="*60)
    print("Test 4: Gathering Disruption Score (GDS)")
    print("="*60)

    # Centroid (0, 0); nearest members are 1 m away, so the O-space radius is 1 m.
    active_agents = {1: [2.0, 0.0], 2: [-2.0, 0.0], 3: [0.0, 1.0], 4: [0.0, -1.0]}
    group_graph = {'groups': [_group(0, 'gathering', [1, 2, 3, 4])]}

    cases = [
        ("passes outside the members", [[4.0, 0.0], [5.0, 0.0]], 0.0),
        ("1.5 m from centroid: between members, outside O-space", [[1.5, 0.0], [1.5, 0.0]], 0.0),
        ("3 frames inside the O-space", [[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]], 3 * DT),
    ]
    for name, trajectory, expected in cases:
        results = _run(trajectory, active_agents, group_graph)
        _check(name, results['GDS'], expected)
        assert results['CPI'] == 0.0, "a gathering must not add CPI"

    print("  ✓ Test passed")


def test_overlapping_gatherings():
    """GDS takes the max over gatherings: overlapping O-spaces count a frame once"""
    print("\n" + "="*60)
    print("Test 5: Overlapping Gatherings")
    print("="*60)

    # Two rings (radius 2 m) centered at x=0 and x=1; (0.5, 0) is inside both O-spaces.
    active_agents = {**_ring(0.0, list(range(1, 7))), **_ring(1.0, list(range(7, 13)))}
    group_graph = {'groups': [_group(0, 'gathering', list(range(1, 7))),
                              _group(1, 'gathering', list(range(7, 13)))]}

    _check("3 frames inside both O-spaces",
           _run([[0.5, 0.0]] * 3, active_agents, group_graph)['GDS'], 3 * DT)

    print("  ✓ Test passed")


def test_cpi_buffer_zone():
    """CPI: dwell time in the R-space = 1.2 m buffer outside the convex hull (Algorithm 3)"""
    print("\n" + "="*60)
    print("Test 6: Cohesive Proxemic Intrusion (CPI)")
    print("="*60)

    # Cohesive group: 4 agents on the square [0, 2] x [0, 2]
    active_agents = {10: [0.0, 0.0], 11: [2.0, 0.0], 12: [2.0, 2.0], 13: [0.0, 2.0]}
    group_graph = {'groups': [_group(0, 'cohesive', [10, 11, 12, 13])]}

    cases = [
        ("5 m from the group", [[7.0, 1.0]] * 3, 0.0),
        ("inside the hull (P-space, not R-space)", [[1.0, 1.0]] * 3, 0.0),
        ("0.5 m outside the hull edge", [[2.5, 1.0]] * 3, 3 * DT),
        ("1.5 m outside the hull edge", [[3.5, 1.0]] * 3, 0.0),
    ]
    for name, trajectory, expected in cases:
        results = _run(trajectory, active_agents, group_graph)
        _check(name, results['CPI'], expected)
        assert results['GDS'] == 0.0, "a cohesive group must not add GDS"

    print("  ✓ Test passed")


def test_overlapping_cohesive_groups():
    """CPI takes the max over cohesive groups: overlapping R-spaces count a frame once"""
    print("\n" + "="*60)
    print("Test 7: Overlapping Cohesive Groups")
    print("="*60)

    # Two walking pairs 2 m apart; (1, 1) is 1 m from both, inside both R-spaces.
    active_agents = {1: [0.0, 0.0], 2: [0.0, 2.0], 3: [2.0, 0.0], 4: [2.0, 2.0]}
    group_graph = {'groups': [_group(0, 'cohesive', [1, 2]), _group(1, 'cohesive', [3, 4])]}

    _check("3 frames inside both R-spaces",
           _run([[1.0, 1.0]] * 3, active_agents, group_graph)['CPI'], 3 * DT)

    print("  ✓ Test passed")


def test_reset_with_group_graph():
    """reset(group_graph) caches queue edges and starts a clean episode"""
    print("\n" + "="*60)
    print("Test 8: reset() with group_graph")
    print("="*60)

    active_agents = {1: [0.0, 5.0], 2: [1.0, 5.0], 3: [2.0, 5.0], 4: [0.0, 10.0], 5: [1.0, 10.0]}
    group_graph = {'groups': [_queue(0, [1, 2, 3]), _queue(1, [4, 5])]}

    metrics = TopologicalMetricEngine()
    metrics.reset(group_graph)
    assert metrics._edge_cache[0] == [(1, 2), (2, 3)], f"Unexpected edge cache: {metrics._edge_cache[0]}"
    assert metrics._edge_cache[1] == [(4, 5)], f"Unexpected edge cache: {metrics._edge_cache[1]}"
    print("  Edge cache verified")

    for ego_loc in [[0.5, 4.0], [0.5, 6.0]]:
        metrics.evaluate_frame(ego_loc, active_agents, group_graph, dt=DT)
    _check("QCR before reset", metrics.get_episode_metrics()['QCR'], 1)

    # A new episode must not pair its first frame with the last frame of the previous one.
    metrics.reset(group_graph)
    metrics.evaluate_frame([0.5, 4.0], active_agents, group_graph, dt=DT)
    results = metrics.get_episode_metrics()
    _check("QCR after reset", results['QCR'], 0)
    _check("Total_Frames after reset", results['Total_Frames'], 1)

    print("  ✓ Test passed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TopologicalMetricEngine Test Suite")
    print("="*60)

    try:
        test_edge_proximity()
        test_queue_cutting()
        test_multiple_queues()
        test_gathering_disruption()
        test_overlapping_gatherings()
        test_cpi_buffer_zone()
        test_overlapping_cohesive_groups()
        test_reset_with_group_graph()

        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60 + "\n")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}\n")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}\n")
        raise
