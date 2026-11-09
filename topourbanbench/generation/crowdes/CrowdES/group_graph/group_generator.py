"""
Group Graph Generator based on Emitter Prior

Analyzes agent profiles from CrowdES Emitter and clusters them into groups
based on spatial, temporal, and kinematic similarity.
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple


class GroupGenerator:
    def __init__(self, config: Dict):
        self.config = config

        # New unified scenarios.yaml structure (Fallbacks applied)
        cohesive_config = config.get('cohesive', {})
        self.grouping_rules = {
            'proximity_threshold': cohesive_config.get('proximity_threshold', 80.0),
            'goal_similarity_threshold': cohesive_config.get('goal_similarity_threshold', 0.7),
            'temporal_threshold': cohesive_config.get('temporal_threshold', 2.0),
            'speed_variance_threshold': cohesive_config.get('speed_variance_threshold', 8.0),
            'min_group_size': cohesive_config.get('min_group_size', 2),
            'max_group_size': cohesive_config.get('max_group_size', 5),
        }

    def generate(self, agent_profiles: np.ndarray, agent_ids: List[int],
                 queue_metadata: List[Dict] = None,
                 existing_queue_assignments: Dict[int, List[int]] = None) -> Dict:
        """
        Generate group graph from agent profiles (ALL GROUP TYPES).

        Args:
            agent_profiles: [agent_type, frame_origin, speed, origin_x, origin_y, goal_x, goal_y]
            agent_ids: List of agent IDs
            queue_metadata: Optional queue location metadata from UnifiedScenarioGenerator
            existing_queue_assignments: Optional persistent queue assignments {queue_idx: [agent_ids]}

        Returns:
            Unified group graph with queue + cohesive groups
        """
        N = len(agent_ids)
        if N == 0:
            return {'groups': [], 'agent_to_group': {}}

        # Step 1: Check if queue scenario is enabled
        if queue_metadata is not None and len(queue_metadata) > 0:
            # Assign agents to queue groups (preserving existing assignments)
            queue_groups = self._assign_queue_groups(
                agent_profiles, agent_ids, queue_metadata, existing_queue_assignments
            )
        else:
            queue_groups = {'groups': [], 'agent_to_group': {}}

        # Step 2: For non-queue agents, detect cohesive groups
        non_queue_agent_ids = [aid for aid in agent_ids if aid not in queue_groups['agent_to_group']]


        if len(non_queue_agent_ids) >= self.grouping_rules['min_group_size']:
            # Get indices of non-queue agents
            non_queue_indices = [agent_ids.index(aid) for aid in non_queue_agent_ids]
            non_queue_profiles = agent_profiles[non_queue_indices]

            # Extract features
            features = {
                'origins': non_queue_profiles[:, 3:5],
                'goals': non_queue_profiles[:, 5:7],
                'speeds': non_queue_profiles[:, 2],
                'times': non_queue_profiles[:, 1],
            }

            # Compute compatibility matrix
            compatibility_matrix = self._compute_compatibility(features)

            # Find connected components (Clusters)
            clusters = self._cluster_agents(compatibility_matrix, non_queue_agent_ids,
                                            origins=features['origins'])


            # Build group graph for cohesive groups
            cohesive_groups = []
            cohesive_agent_to_group = {}
            formation_type = self.config.get('scenario_type', 'family')

            # Get cohesion strength from config
            cohesive_config = self.config.get('cohesive', {})
            cohesion_strength = cohesive_config.get('cohesion_strength', 10.0) / 10.0  # Normalize to [0, 1]

            for group_id, member_ids in enumerate(clusters):
                M = len(member_ids)
                # Build adjacency matrix (clique topology for cohesive groups)
                adjacency_matrix = self._build_adjacency_matrix(M, 'clique')

                cohesive_groups.append({
                    'group_id': group_id,
                    'member_ids': member_ids,
                    'formation_type': formation_type,
                    'cohesion_strength': cohesion_strength,
                    'adjacency_matrix': adjacency_matrix,
                })
                for agent_id in member_ids:
                    cohesive_agent_to_group[agent_id] = group_id


            non_queue_groups = {
                'groups': cohesive_groups,
                'agent_to_group': cohesive_agent_to_group
            }
        else:
            non_queue_groups = {'groups': [], 'agent_to_group': {}}

        # Step 3: Merge queue and cohesive groups
        group_graph = self._merge_group_graphs(queue_groups, non_queue_groups)

        return group_graph

    def _compute_compatibility(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        N = len(features['origins'])
        compat = np.ones((N, N), dtype=bool)

        # Rule 1: Spatial proximity
        dist_origins = cdist(features['origins'], features['origins'], metric='euclidean')
        compat &= (dist_origins <= self.grouping_rules['proximity_threshold'])

        # Rule 2: Goal similarity (Cosine similarity)
        goal_vectors = features['goals'] - features['origins']
        norms = np.linalg.norm(goal_vectors, axis=1, keepdims=True)
        norms = np.where(norms > 1e-6, norms, 1.0)
        goal_vectors_norm = goal_vectors / norms
        
        cosine_sim = goal_vectors_norm @ goal_vectors_norm.T
        compat &= (cosine_sim >= self.grouping_rules['goal_similarity_threshold'])

        # Rule 3: Temporal alignment
        time_diff = np.abs(features['times'][:, None] - features['times'][None, :])
        time_diff_sec = time_diff * 0.2 # assume 5 FPS
        compat &= (time_diff_sec <= self.grouping_rules['temporal_threshold'])

        # Rule 4: Speed compatibility
        speed_diff = np.abs(features['speeds'][:, None] - features['speeds'][None, :])
        compat &= (speed_diff <= self.grouping_rules['speed_variance_threshold'])

        np.fill_diagonal(compat, False)
        return compat.astype(float)

    def _cluster_agents(self, compatibility_matrix: np.ndarray, agent_ids: List[int],
                        origins: np.ndarray = None) -> List[List[int]]:
        # BFS connected components with diameter constraint.
        # Allows side-by-side chains (A-B-C each within threshold of neighbors)
        # while rejecting DFS-chained strangers (A-C >> threshold apart).
        N = len(agent_ids)
        threshold = self.grouping_rules['proximity_threshold']
        max_group_size = self.grouping_rules['max_group_size']
        # Maximum span allowed: (max_members - 1) adjacent spacings
        max_diameter = threshold * (max_group_size - 1)

        adj = [[] for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if compatibility_matrix[i, j] > 0:
                    adj[i].append(j)
                    adj[j].append(i)

        assigned = [False] * N
        clusters = []

        for start in range(N):
            if assigned[start]:
                continue

            cluster = [start]
            in_cluster = {start}
            queue = list(adj[start])

            while queue:
                node = queue.pop(0)
                if assigned[node] or node in in_cluster:
                    continue
                if len(cluster) >= max_group_size:
                    break
                # Reject if adding this node would exceed diameter limit
                if origins is not None:
                    pts = origins[cluster + [node]]
                    if np.linalg.norm(pts[:, None] - pts[None, :], axis=2).max() > max_diameter:
                        continue
                cluster.append(node)
                in_cluster.add(node)
                for nb in adj[node]:
                    if nb not in in_cluster and not assigned[nb]:
                        queue.append(nb)

            for idx in cluster:
                assigned[idx] = True
            if self.grouping_rules['min_group_size'] <= len(cluster) <= max_group_size:
                clusters.append([agent_ids[idx] for idx in cluster])

        return clusters

    def _build_adjacency_matrix(self, M: int, topology: str) -> np.ndarray:
        """
        Build adjacency matrix for given topology type.

        Args:
            M: Number of members in group
            topology: 'clique', 'chain', 'directed_chain', 'ring', or 'star'

        Returns:
            W: (M, M) adjacency matrix
        """
        if topology == 'clique':
            # Fully connected graph (all members attend to each other equally)
            W = np.ones((M, M), dtype=float)
            np.fill_diagonal(W, 0)  # No self-attention

        elif topology == 'chain':
            # Bidirectional linear chain
            W = np.zeros((M, M), dtype=float)
            for i in range(M - 1):
                W[i, i+1] = 1.0
                W[i+1, i] = 1.0

        elif topology == 'directed_chain':
            # Unidirectional chain for queue: each follower attends to leader ahead
            # W[i, j] = 1 means agent i follows agent j
            # Agent 0 is leader, agent 1 follows 0, agent 2 follows 1, etc.
            W = np.zeros((M, M), dtype=float)
            for i in range(1, M):
                W[i, i-1] = 1.0  # Agent i follows agent i-1

        elif topology == 'ring':
            # Ring topology (chain with wraparound)
            W = np.zeros((M, M), dtype=float)
            for i in range(M):
                W[i, (i+1) % M] = 1.0
                W[(i+1) % M, i] = 1.0

        elif topology == 'star':
            # Star topology (all members attend to first member, who attends to all)
            W = np.zeros((M, M), dtype=float)
            W[0, 1:] = 1.0
            W[1:, 0] = 1.0

        else:
            raise ValueError(f"Unknown topology: {topology}")

        return W

    def _point_to_segment_distance(self, point, seg_start, seg_end):
        """
        Calculate minimum distance from point to line segment.

        Args:
            point: (2,) or (N, 2) point coordinates
            seg_start: (2,) segment start
            seg_end: (2,) segment end

        Returns:
            distance: float or (N,) array
        """
        # Handle both single point and batch
        single_point = (point.ndim == 1)
        if single_point:
            point = point[None, :]  # (1, 2)

        # Vector from start to end
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)

        if seg_len_sq < 1e-6:
            # Degenerate segment (point)
            distances = np.linalg.norm(point - seg_start[None, :], axis=1)
        else:
            # Project point onto line (parameterized by t)
            t = np.dot(point - seg_start[None, :], seg_vec) / seg_len_sq
            t = np.clip(t, 0, 1)  # Clamp to segment bounds

            # Closest point on segment
            closest = seg_start[None, :] + t[:, None] * seg_vec[None, :]

            # Distance
            distances = np.linalg.norm(point - closest, axis=1)

        return distances[0] if single_point else distances

    def _assign_queue_groups(self, agent_profiles: np.ndarray,
                            agent_ids: List[int],
                            queue_metadata: List[Dict],
                            existing_queue_assignments: Dict[int, List[int]] = None) -> Dict:
        """
        Assign agents to queue groups based on goal proximity to queue lines.

        Args:
            agent_profiles: (N, 7) agent profiles
            agent_ids: List of N agent IDs
            queue_metadata: List of queue locations from UnifiedScenarioGenerator
            existing_queue_assignments: Persistent queue assignments {queue_idx: [agent_ids]}

        Returns:
            group_graph: Dictionary with queue groups
        """
        # Get queue detection parameters from config
        queue_config = self.config.get('queue', {})
        goal_threshold = queue_config.get('goal_proximity_threshold', 500.0)
        min_queue_size = queue_config.get('min_queue_size', 2)
        max_queue_size = queue_config.get('max_queue_size', 10)

        # Extract goal positions and times from agent profiles
        goals = agent_profiles[:, 5:7]  # goal_x, goal_y
        times = agent_profiles[:, 1]    # frame_origin

        # Start with existing assignments (persistent queue membership)
        queue_assignments = existing_queue_assignments.copy() if existing_queue_assignments else {}

        # Get already assigned agent IDs
        already_assigned = set()
        for assigned_ids in queue_assignments.values():
            already_assigned.update(assigned_ids)

        for queue_idx, queue_info in enumerate(queue_metadata):
            # Skip if this queue already has assignments (persistent)
            if queue_idx in queue_assignments and len(queue_assignments[queue_idx]) > 0:
                queue_name = queue_info.get('queue_id', queue_info.get('building_name', f'Queue_{queue_idx}'))
                print(f"[Queue Group] {queue_name}: {len(queue_assignments[queue_idx])} agents (persisted)")
                continue

            queue_point = np.array(queue_info['goal_point'])

            # Use point-to-line-segment distance (consistent with inference_model.py)
            queue_direction = np.array(queue_info.get('direction', [0, 1]))
            queue_length_pixels = queue_info.get('length_px', 20)  # Use actual generated length

            # Line segment: queue_point to queue_point + direction * length
            line_end = queue_point + queue_direction * queue_length_pixels

            # Compute distances from all agent goals to this queue line segment
            distances = self._point_to_segment_distance(goals, queue_point, line_end)

            # Find agents within threshold (excluding already assigned agents)
            nearby_agent_indices = []
            for idx in np.where(distances <= goal_threshold)[0]:
                if agent_ids[idx] not in already_assigned:
                    nearby_agent_indices.append(idx)

            if len(nearby_agent_indices) >= min_queue_size:
                # Limit to max queue size (no sorting - natural order)
                selected_indices = nearby_agent_indices[:max_queue_size]

                # Store agent IDs (goals remain as sampled from scenario_density)
                new_assignments = [agent_ids[i] for i in selected_indices]
                queue_assignments[queue_idx] = new_assignments
                already_assigned.update(new_assignments)

                queue_name = queue_info.get('queue_id', queue_info.get('building_name', f'Queue_{queue_idx}'))
                print(f"[Queue Group] {queue_name}: {len(selected_indices)} agents assigned (new)")

        # Build group graph
        groups = []
        agent_to_group = {}
        formation_type = self.config.get('scenario_type', 'queue_line')

        # Get cohesion strength from queue config
        cohesion_strength = queue_config.get('leader_follower_strength', 15.0) / 15.0  # Normalize to [0, 1]

        for queue_idx, member_ids in queue_assignments.items():
            M = len(member_ids)
            # Build adjacency matrix (directed_chain topology for queue)
            adjacency_matrix = self._build_adjacency_matrix(M, 'directed_chain')

            # Get queue name from metadata
            queue_info = queue_metadata[queue_idx]
            queue_name = queue_info.get('queue_id', queue_info.get('building_name', f'Queue_{queue_idx}'))

            group_entry = {
                'group_id': len(groups),
                'member_ids': member_ids,
                'formation_type': formation_type,
                'cohesion_strength': cohesion_strength,
                'adjacency_matrix': adjacency_matrix,
                'is_queue': True,
                'queue_idx': queue_idx,
                'queue_id': queue_name,
                'queue_metadata': queue_metadata[queue_idx]
            }

            groups.append(group_entry)

            # Update agent-to-group mapping
            for agent_id in member_ids:
                agent_to_group[agent_id] = group_entry['group_id']

        return {
            'groups': groups,
            'agent_to_group': agent_to_group
        }

    def _merge_group_graphs(self, queue_groups: Dict, non_queue_groups: Dict) -> Dict:
        """
        Merge queue and non-queue group graphs.

        Args:
            queue_groups: Group graph for queue agents
            non_queue_groups: Group graph for cohesive agents

        Returns:
            merged_graph: Combined group graph
        """
        merged_groups = []
        merged_agent_to_group = {}

        # Add queue groups first
        for group in queue_groups['groups']:
            # Create a copy to avoid modifying original
            group_copy = group.copy()
            group_copy['group_id'] = len(merged_groups)
            merged_groups.append(group_copy)

            for agent_id in group_copy['member_ids']:
                merged_agent_to_group[agent_id] = group_copy['group_id']

        # Add non-queue groups
        for group in non_queue_groups['groups']:
            # Create a copy to avoid modifying original
            group_copy = group.copy()
            group_copy['group_id'] = len(merged_groups)
            group_copy['is_queue'] = False  # Mark as non-queue
            merged_groups.append(group_copy)

            for agent_id in group_copy['member_ids']:
                merged_agent_to_group[agent_id] = group_copy['group_id']

        return {
            'groups': merged_groups,
            'agent_to_group': merged_agent_to_group
        }


def print_group_summary(group_graph: Dict, agent_ids: List[int]):
    print(f"\n{'='*40}")
    print(f"Family/Gathering Group Summary")
    print(f"{'='*40}")
    print(f"Total spawned agents : {len(agent_ids)}")
    print(f"Total cohesive groups: {len(group_graph['groups'])}")
    print(f"Grouped agents       : {len(group_graph['agent_to_group'])}")
    
    for group in group_graph['groups']:
        print(f"  - Group {group['group_id']}: {group['member_ids']} ({group['formation_type']})")
    print()