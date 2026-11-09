"""
Shared topology utilities for TopoUrbanBench.

Single source of truth for adjacency matrix construction and edge list derivation.
Used by generation (inference_model.py), environment (GroupGraphRuntime),
and metrics (TopologicalMetricEngine).
"""

import numpy as np
from typing import List, Tuple


TOPOLOGY_FOR_GROUP_TYPE = {
    'queue': 'directed_chain',
    'gathering': 'ring',
    'cohesive': 'clique',
    'crosswalk': 'clique',
}


def build_adjacency(topology: str, M: int) -> np.ndarray:
    """Build (M, M) adjacency matrix for given topology.

    Args:
        topology: 'directed_chain', 'ring', or 'clique'
        M: number of group members

    Returns:
        W: (M, M) float32 adjacency matrix, 0 on diagonal
    """
    if M < 1:
        return np.zeros((0, 0), dtype=np.float32)

    if topology == 'directed_chain':
        W = np.zeros((M, M), dtype=np.float32)
        for i in range(1, M):
            W[i, i - 1] = 1.0  # agent i follows agent i-1
        return W

    elif topology == 'ring':
        W = np.zeros((M, M), dtype=np.float32)
        for i in range(M):
            W[i, (i + 1) % M] = 1.0
            W[(i + 1) % M, i] = 1.0
        return W

    elif topology == 'clique':
        W = np.ones((M, M), dtype=np.float32)
        np.fill_diagonal(W, 0.0)
        return W

    else:
        raise ValueError(f"Unknown topology: {topology}")


def topology_to_edges(topology: str, M: int) -> List[Tuple[int, int]]:
    """Derive edge list (local indices) from topology type.

    Args:
        topology: 'directed_chain', 'ring', or 'clique'
        M: number of group members

    Returns:
        List of (local_idx_i, local_idx_j) tuples
    """
    if M < 2:
        return []

    if topology == 'directed_chain':
        return [(i, i + 1) for i in range(M - 1)]

    elif topology == 'ring':
        return [(i, (i + 1) % M) for i in range(M)]

    elif topology == 'clique':
        return [(i, j) for i in range(M) for j in range(i + 1, M)]

    else:
        raise ValueError(f"Unknown topology: {topology}")
