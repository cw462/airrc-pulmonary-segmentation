"""GT-only unrooted skeleton branches and physical-length coverage.

All candidate edges are 26-neighbor edges. Adjacent high-degree voxels are
contracted into junction clusters. Local degree-two returns wholly inside a
junction are absorbed. A cluster with exactly two external paths is a bend,
not a true bifurcation: concatenate those paths using its shortest internal
26-neighbor connection. True junction-internal edges never contribute length.
This explicit corner treatment prevents 26-neighbor triangles from splitting
a bend into spurious short branches. It does not use the prediction.
"""

from collections import defaultdict
import heapq
import math

import numpy as np
from scipy.spatial import cKDTree


ALGORITHM_VERSION = 'gt-unrooted-26-junction-v1'
MIN_BRANCH_LENGTH_MM = 3.0


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def edge_lengths(coords_zyx, edges, spacing_xyz, direction=None):
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if not len(edges):
        return np.empty(0, dtype=float)
    matrix = np.asarray(direction if direction is not None else np.eye(3), dtype=float).reshape(3, 3)
    matrix = matrix @ np.diag(np.asarray(spacing_xyz, dtype=float))
    delta_xyz = (coords_zyx[edges[:, 1]] - coords_zyx[edges[:, 0]])[:, ::-1]
    return np.linalg.norm(delta_xyz @ matrix.T, axis=1)


def _inside_path(start, target, allowed, adjacency, coords, spacing, direction):
    if start == target:
        return []
    queue, distances, previous = [(0.0, start)], {start: 0.0}, {}
    while queue:
        distance, current = heapq.heappop(queue)
        if distance != distances[current]:
            continue
        if current == target:
            route = []
            while current != start:
                prior = previous[current]
                route.append((prior, current))
                current = prior
            return route[::-1]
        for neighbor in adjacency[current]:
            if neighbor not in allowed:
                continue
            length = float(edge_lengths(coords, [(current, neighbor)], spacing, direction)[0])
            candidate = distance + length
            if candidate < distances.get(neighbor, float('inf')):
                distances[neighbor], previous[neighbor] = candidate, current
                heapq.heappush(queue, (candidate, neighbor))
    raise ValueError('A contracted junction cluster is not internally connected.')


def extract_reference_branches(coords_zyx, spacing_xyz=(1, 1, 1), direction=None,
                               minimum_length_mm=MIN_BRANCH_LENGTH_MM):
    coords = np.asarray(coords_zyx, dtype=np.int32).reshape(-1, 3)
    if len(coords) != len(np.unique(coords, axis=0)):
        raise ValueError('Duplicate GT skeleton coordinates.')
    if np.any(np.asarray(spacing_xyz) <= 0):
        raise ValueError('Physical spacing must be positive.')
    n = len(coords)
    pairs = (cKDTree(coords).query_pairs(1.0, p=np.inf, output_type='ndarray')
             if n else np.empty((0, 2), dtype=np.int64))
    pairs = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
    if len(pairs):
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    adjacency = [[] for _ in range(n)]
    for left, right in pairs:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    for neighbors in adjacency:
        neighbors.sort()
    degrees = np.asarray([len(values) for values in adjacency])
    junction = degrees >= 3
    uf = UnionFind(n)
    for left, right in pairs:
        if junction[left] and junction[right]:
            uf.union(int(left), int(right))
    clusters = defaultdict(list)
    for index in np.flatnonzero(junction):
        clusters[uf.find(int(index))].append(int(index))
    nodes, node_map = [], np.full(n, -1, dtype=np.int32)
    for root in sorted(clusters):
        members = clusters[root]
        node_map[members] = len(nodes)
        nodes.append({'kind': 'junction', 'members': members})
    for index in np.flatnonzero(degrees == 1):
        node_map[index] = len(nodes)
        nodes.append({'kind': 'endpoint', 'members': [int(index)]})
    # Absorb local return triangles, not long loops leaving the junction.
    absorbed = 0
    changed = True
    while changed:
        changed = False
        for index in np.flatnonzero((degrees == 2) & (node_map < 0)):
            left, right = adjacency[index]
            node = int(node_map[left])
            if node >= 0 and node == node_map[right] and nodes[node]['kind'] == 'junction':
                node_map[index] = node
                nodes[node]['members'].append(int(index))
                absorbed += 1
                changed = True
    visited, segments = set(), []

    def key(left, right):
        return (min(left, right), max(left, right))

    for source, node in enumerate(nodes):
        for left in sorted(node['members']):
            for right in adjacency[left]:
                edge = key(left, right)
                if edge in visited:
                    continue
                visited.add(edge)
                if node_map[right] == source:
                    continue
                edges = [(left, right)]
                prior, current = left, right
                while node_map[current] < 0:
                    if degrees[current] != 2:
                        raise ValueError('Unexpected non-degree-two path interior.')
                    following = next(value for value in adjacency[current] if value != prior)
                    edge = key(current, following)
                    if edge in visited:
                        raise ValueError('GT path revisited an edge before reaching a graph node.')
                    visited.add(edge)
                    edges.append((current, following))
                    prior, current = current, following
                segments.append({'start': source, 'end': int(node_map[current]), 'edges': edges})
    branches = []
    # Components with no endpoint/junction have no start node above.
    for left in range(n):
        for right in adjacency[left]:
            if key(left, right) in visited:
                continue
            if node_map[left] >= 0 or degrees[left] != 2:
                raise ValueError('An untraced edge is not part of a pure cycle.')
            edges = [(left, right)]
            visited.add(key(left, right))
            prior, current = left, right
            while current != left:
                if degrees[current] != 2 or node_map[current] >= 0:
                    raise ValueError('A purported pure cycle contains a junction/endpoint.')
                following = next(value for value in adjacency[current] if value != prior)
                if key(current, following) in visited:
                    raise ValueError('Pure cycle closed on an unexpected edge.')
                visited.add(key(current, following))
                edges.append((current, following))
                prior, current = current, following
            branches.append({'edges': edges, 'exclusion_reason': 'cycle_component'})
    if len(visited) != len(pairs):
        raise ValueError('Some GT graph edges were not accounted for.')
    incidence = defaultdict(list)
    for index, segment in enumerate(segments):
        incidence[segment['start']].append((index, segment['edges'][0][0]))
        incidence[segment['end']].append((index, segment['edges'][-1][1]))
    segment_uf = UnionFind(len(segments))
    connectors, transit_nodes = {}, set()
    for node_id, node in enumerate(nodes):
        links = incidence[node_id]
        if node['kind'] == 'junction' and len(links) == 2:
            (first, first_voxel), (second, second_voxel) = links
            segment_uf.union(first, second)
            connectors[node_id] = _inside_path(first_voxel, second_voxel, set(node['members']),
                                               adjacency, coords, spacing_xyz, direction)
            transit_nodes.add(node_id)
    groups = defaultdict(list)
    for index in range(len(segments)):
        groups[segment_uf.find(index)].append(index)
    for root in sorted(groups):
        members = groups[root]
        edges, boundaries, used_transit = [], [], set()
        for index in members:
            segment = segments[index]
            edges.extend(segment['edges'])
            for node_id in (segment['start'], segment['end']):
                if node_id in transit_nodes:
                    used_transit.add(node_id)
                else:
                    boundaries.append(node_id)
        for node_id in sorted(used_transit):
            edges.extend(connectors[node_id])
        if len(boundaries) not in (0, 2):
            raise ValueError(f'Maximal GT path has {len(boundaries)} boundaries instead of two.')
        branches.append({'edges': edges, 'exclusion_reason': '' if boundaries else 'cycle_component'})
    for index in np.flatnonzero(degrees == 0):
        branches.append({'edges': [], 'point_index': int(index), 'exclusion_reason': 'degenerate_isolated_voxel'})
    for node_id, node in enumerate(nodes):
        if node['kind'] == 'junction' and not incidence[node_id]:
            branches.append({'edges': [], 'point_index': min(node['members']),
                             'exclusion_reason': 'degenerate_junction_component'})
    for index, branch in enumerate(branches):
        branch['branch_id'] = f'branch_{index + 1:06d}'
        branch['edges'] = np.asarray(branch['edges'], dtype=np.int32).reshape(-1, 2)
        branch['edge_lengths_mm'] = edge_lengths(coords, branch['edges'], spacing_xyz, direction)
        branch['length_mm'] = float(math.fsum(branch['edge_lengths_mm'].tolist()))
        if not branch['exclusion_reason']:
            if branch['length_mm'] <= 0:
                branch['exclusion_reason'] = 'degenerate_path'
            elif branch['length_mm'] < minimum_length_mm:
                branch['exclusion_reason'] = 'short_branch'
    return {
        'coords_zyx': coords, 'raw26_edges': pairs, 'node_map': node_map,
        'nodes': nodes, 'branches': branches,
        'junction_cluster_count': sum(node['kind'] == 'junction' and len(incidence[index]) >= 3
                                      for index, node in enumerate(nodes)),
        'suppressed_degree2_cluster_count': len(transit_nodes),
        'absorbed_local_return_voxels': absorbed,
        'minimum_length_mm': float(minimum_length_mm),
    }


def branch_coverages(graph, prediction_mask):
    coords = graph['coords_zyx']
    covered = (prediction_mask[tuple(coords.T)].astype(float) if len(coords) else np.empty(0))
    details = []
    for branch in graph['branches']:
        edges, lengths = branch['edges'], branch['edge_lengths_mm']
        length = branch['length_mm']
        contributions = lengths * (covered[edges[:, 0]] + covered[edges[:, 1]]) * 0.5 if len(edges) else []
        numerator = float(math.fsum(contributions))
        # Stable sums and a roundoff-only range guard. Detection thresholds
        # remain exactly 0.80 and 0.90; no spatial or threshold tolerance.
        coverage = min(1.0, max(0.0, numerator / length)) if length > 0 else float('nan')
        eligible = not branch['exclusion_reason']
        details.append({
            'branch_id': branch['branch_id'], 'length_mm': length, 'coverage': coverage,
            'detected80': bool(coverage >= 0.80) if eligible else None,
            'detected90': bool(coverage >= 0.90) if eligible else None,
            'exclusion_reason': branch['exclusion_reason'],
        })
    valid = [item for item in details if not item['exclusion_reason']]
    short = [item for item in details if item['exclusion_reason'] == 'short_branch']
    count = len(valid)
    detected80 = sum(item['detected80'] for item in valid)
    detected90 = sum(item['detected90'] for item in valid)
    return {
        'branch_detection_rate_80': detected80 / count if count else float('nan'),
        'branch_detection_rate_90': detected90 / count if count else float('nan'),
        'branch_coverage_mean': float(np.mean([item['coverage'] for item in valid])) if valid else float('nan'),
        'gt_branch_count': count, 'detected_branch_count_80': detected80, 'detected_branch_count_90': detected90,
        'excluded_short_branch_count': len(short),
        'excluded_short_branch_total_length_mm': float(sum(item['length_mm'] for item in short)),
        'excluded_cycle_component_count': sum(item['exclusion_reason'] == 'cycle_component' for item in details),
        'degenerate_structure_count': sum(item['exclusion_reason'].startswith('degenerate') for item in details),
        'junction_cluster_count': graph['junction_cluster_count'],
        'suppressed_degree2_cluster_count': graph['suppressed_degree2_cluster_count'],
    }, details
