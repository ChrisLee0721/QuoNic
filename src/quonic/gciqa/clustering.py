"""Geometric clustering for GCIQA.

Performs K-means clustering on conformations from quantum measurements
to find the most-probable conformation centers.

Example::

    from quonic.gciqa import geometric_clustering

    centers, labels = geometric_clustering(
        conformations=[...],  # list of coordinate dicts
        n_clusters=5,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ClusterResult:
    """Result of geometric clustering.

    Attributes:
        centers: Cluster center coordinates.
        labels: Cluster label for each conformation.
        cluster_sizes: Number of conformations in each cluster.
        largest_cluster: Index of the largest cluster.
        convergence_radius: Max distance from largest cluster center to its members.
    """

    centers: list[dict[str, tuple[float, float, float]]]
    labels: list[int]
    cluster_sizes: list[int]
    largest_cluster: int
    convergence_radius: float


def geometric_clustering(
    conformations: list[dict[str, tuple[float, float, float]]],
    n_clusters: int = 5,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
) -> ClusterResult:
    """K-means clustering on molecular conformations.

    Each conformation is represented as a dict mapping atom names
    to (x, y, z) coordinates. The distance between conformations
    is the RMSD (root-mean-square deviation) of atomic positions.

    Args:
        conformations: List of conformations (each a dict of atom -> coords).
        n_clusters: Number of clusters.
        max_iterations: Max K-means iterations.
        tolerance: Convergence tolerance for center movement.

    Returns:
        ClusterResult with centers, labels, and statistics.

    Raises:
        ValueError: If fewer conformations than clusters.
    """
    if len(conformations) < n_clusters:
        raise ValueError(
            f"Need at least {n_clusters} conformations, got {len(conformations)}"
        )

    n = len(conformations)

    # Flatten conformations to vectors for distance computation
    atom_names = sorted(conformations[0].keys())
    vectors = []
    for conf in conformations:
        vec = []
        for atom in atom_names:
            vec.extend(conf[atom])
        vectors.append(vec)

    # Initialize centers using K-means++ initialization
    center_indices = _kmeans_pp_init(vectors, n_clusters)

    # K-means iterations
    labels = [0] * n
    for iteration in range(max_iterations):
        # Assign each point to nearest center
        changed = False
        for i in range(n):
            best_cluster = 0
            best_dist = float('inf')
            for c_idx, c in enumerate(center_indices):
                d = _rmsd(vectors[i], vectors[c])
                if d < best_dist:
                    best_dist = d
                    best_cluster = c_idx
            if labels[i] != best_cluster:
                labels[i] = best_cluster
                changed = True

        if not changed:
            break

        # Update centers
        new_center_indices = []
        for c in range(n_clusters):
            members = [i for i in range(n) if labels[i] == c]
            if not members:
                # Empty cluster - keep old center
                new_center_indices.append(center_indices[c])
                continue
            # Compute mean position
            mean_vec = [0.0] * len(vectors[0])
            for m in members:
                for j in range(len(mean_vec)):
                    mean_vec[j] += vectors[m][j]
            for j in range(len(mean_vec)):
                mean_vec[j] /= len(members)
            # Find closest actual point to mean
            best_idx = members[0]
            best_dist = _rmsd(vectors[members[0]], mean_vec)
            for m in members[1:]:
                d = _rmsd(vectors[m], mean_vec)
                if d < best_dist:
                    best_dist = d
                    best_idx = m
            new_center_indices.append(best_idx)

        # Check convergence
        max_move = 0.0
        for c in range(n_clusters):
            move = _rmsd(vectors[center_indices[c]], vectors[new_center_indices[c]])
            max_move = max(max_move, move)

        center_indices = new_center_indices

        if max_move < tolerance:
            break

    # Build result
    centers = []
    for c_idx in center_indices:
        center = {}
        for j, atom in enumerate(atom_names):
            start = j * 3
            center[atom] = (
                vectors[c_idx][start],
                vectors[c_idx][start+1],
                vectors[c_idx][start+2],
            )
        centers.append(center)

    cluster_sizes = [0] * n_clusters
    for l in labels:
        cluster_sizes[l] += 1

    largest = cluster_sizes.index(max(cluster_sizes))

    # Convergence radius: max RMSD from largest cluster center to its members
    largest_members = [i for i in range(n) if labels[i] == largest]
    conv_radius = 0.0
    for m in largest_members:
        d = _rmsd(vectors[m], vectors[center_indices[largest]])
        conv_radius = max(conv_radius, d)

    return ClusterResult(
        centers=centers,
        labels=labels,
        cluster_sizes=cluster_sizes,
        largest_cluster=largest,
        convergence_radius=conv_radius,
    )


def compute_rmsd(
    conf1: dict[str, tuple[float, float, float]],
    conf2: dict[str, tuple[float, float, float]],
) -> float:
    """Compute RMSD between two conformations.

    Args:
        conf1: First conformation.
        conf2: Second conformation.

    Returns:
        RMSD in Angstrom.
    """
    atoms = sorted(set(conf1.keys()) & set(conf2.keys()))
    if not atoms:
        return float('inf')

    sum_sq = 0.0
    for atom in atoms:
        x1, y1, z1 = conf1[atom]
        x2, y2, z2 = conf2[atom]
        sum_sq += (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2

    return math.sqrt(sum_sq / len(atoms))


def _rmsd(v1: list[float], v2: list[float]) -> float:
    """RMSD between two flat vectors."""
    n = len(v1)
    if n == 0:
        return 0.0
    sum_sq = sum((v1[i] - v2[i])**2 for i in range(n))
    return math.sqrt(sum_sq / (n // 3))  # Divide by number of atoms


def _kmeans_pp_init(vectors: list[list[float]], k: int) -> list[int]:
    """K-means++ initialization.

    Selects k initial centers that are spread out.

    Args:
        vectors: Data points.
        k: Number of centers.

    Returns:
        List of indices into vectors.
    """
    import random

    n = len(vectors)
    centers = [random.randint(0, n-1)]

    for _ in range(k - 1):
        # Compute distance from each point to nearest center
        distances = []
        for i in range(n):
            min_d = min(_rmsd(vectors[i], vectors[c]) for c in centers)
            distances.append(min_d ** 2)

        # Weighted random selection
        total = sum(distances)
        if total < 1e-12:
            centers.append(random.randint(0, n-1))
            continue

        r = random.random() * total
        cumulative = 0.0
        for i in range(n):
            cumulative += distances[i]
            if cumulative >= r:
                centers.append(i)
                break

    return centers
