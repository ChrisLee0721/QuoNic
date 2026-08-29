"""Hierarchical coarse-graining with mixed compression ratios.

Applies different compression ratios to different spatial zones:
- Core (binding site): fine-grained (low compression)
- Middle region: medium compression
- Periphery: coarse-grained (high compression)

This allows GCIQA to handle large molecules while preserving
detail where it matters most.
"""

from __future__ import annotations

from typing import Any

from .coarsegrain import (
    _ATOMIC_MASSES,
    CoarseGraining,
    _build_cg_from_groups,
    _coarse_grain_spatial,
    _distance,
)


def hierarchical_coarse_grain(
    atoms: list[str],
    coords: list[tuple[float, float, float]],
    n_super_atoms: int = 20,
    binding_site_center: tuple[float, float, float] | None = None,
    binding_site_radius: float = 10.0,
    compression_ratios: dict[str, float] | None = None,
) -> CoarseGraining:
    """Multi-resolution coarse-graining with zone-based compression.

    Atoms are divided into spatial zones based on distance from the
    binding site center. Each zone gets a different number of super-atoms
    based on its compression ratio.

    Args:
        atoms: Element symbols.
        coords: Cartesian coordinates (Angstrom).
        n_super_atoms: Total target super-atoms across all zones.
        binding_site_center: Center of the binding site. If None, falls
            back to uniform spatial clustering.
        binding_site_radius: Radius of the core zone (Angstrom).
        compression_ratios: Dict with keys "core", "middle", "periphery"
            and compression ratio values. Defaults to {50, 300, 3000}.

    Returns:
        CoarseGraining with mixed-resolution super-atoms.
    """
    if binding_site_center is None:
        return _coarse_grain_spatial(atoms, coords,
                                     [_ATOMIC_MASSES.get(a, 12.0) for a in atoms],
                                     n_super_atoms)

    if compression_ratios is None:
        compression_ratios = {"core": 50, "middle": 300, "periphery": 3000}

    masses = [_ATOMIC_MASSES.get(a, 12.0) for a in atoms]
    n = len(atoms)

    # Assign atoms to zones by distance from binding site center
    core_atoms = []
    middle_atoms = []
    periphery_atoms = []

    r_core = binding_site_radius
    r_middle = binding_site_radius * 3

    for i in range(n):
        d = _distance(coords[i], binding_site_center)
        if d <= r_core:
            core_atoms.append(i)
        elif d <= r_middle:
            middle_atoms.append(i)
        else:
            periphery_atoms.append(i)

    # Compute super-atoms per zone based on compression ratios
    r_core_ratio = compression_ratios.get("core", 50)
    r_middle_ratio = compression_ratios.get("middle", 300)
    r_peri_ratio = compression_ratios.get("periphery", 3000)

    n_core = max(1, len(core_atoms) // int(r_core_ratio)) if core_atoms else 0
    n_middle = max(1, len(middle_atoms) // int(r_middle_ratio)) if middle_atoms else 0
    n_peri = max(1, len(periphery_atoms) // int(r_peri_ratio)) if periphery_atoms else 0

    # Scale to match target n_super_atoms
    total_raw = n_core + n_middle + n_peri
    if total_raw > 0 and total_raw != n_super_atoms:
        scale = n_super_atoms / total_raw
        n_core = max(1, round(n_core * scale)) if core_atoms else 0
        n_middle = max(1, round(n_middle * scale)) if middle_atoms else 0
        n_peri = max(1, round(n_peri * scale)) if periphery_atoms else 0

    # Build groups for each zone
    all_groups: dict[int, list[int]] = {}
    group_idx = 0

    if core_atoms and n_core > 0:
        core_groups = _spatial_cluster(
            atoms, coords, masses, core_atoms, n_core
        )
        for members in core_groups.values():
            all_groups[group_idx] = members
            group_idx += 1

    if middle_atoms and n_middle > 0:
        middle_groups = _spatial_cluster(
            atoms, coords, masses, middle_atoms, n_middle
        )
        for members in middle_groups.values():
            all_groups[group_idx] = members
            group_idx += 1

    if periphery_atoms and n_peri > 0:
        peri_groups = _spatial_cluster(
            atoms, coords, masses, periphery_atoms, n_peri
        )
        for members in peri_groups.values():
            all_groups[group_idx] = members
            group_idx += 1

    if not all_groups:
        return _coarse_grain_spatial(atoms, coords, masses, n_super_atoms)

    return _build_cg_from_groups(atoms, coords, masses, all_groups)


def _spatial_cluster(
    atoms: list[str],
    coords: list[tuple[float, float, float]],
    masses: list[float],
    atom_indices: list[int],
    n_clusters: int,
) -> dict[int, list[int]]:
    """Spatial clustering on a subset of atoms using greedy seeding.

    Uses numpy for vectorized distance computation when available.
    """
    if n_clusters >= len(atom_indices):
        return {i: [i] for i in atom_indices}

    try:
        import numpy as np
        return _spatial_cluster_numpy(coords, atom_indices, n_clusters, np)
    except ImportError:
        pass

    # Pure Python fallback
    seeds = [atom_indices[0]]
    for _ in range(n_clusters - 1):
        max_dist = -1
        best = -1
        for i in atom_indices:
            if i in seeds:
                continue
            min_d = min(_distance(coords[i], coords[s]) for s in seeds)
            if min_d > max_dist:
                max_dist = min_d
                best = i
        if best >= 0:
            seeds.append(best)

    groups: dict[int, list[int]] = {s: [] for s in seeds}
    for i in atom_indices:
        best_seed = min(seeds, key=lambda s: _distance(coords[i], coords[s]))
        groups[best_seed].append(i)

    return groups


def _spatial_cluster_numpy(
    coords: list[tuple[float, float, float]],
    atom_indices: list[int],
    n_clusters: int,
    np: Any,
) -> dict[int, list[int]]:
    """Numpy-accelerated spatial clustering for a subset of atoms."""
    import numpy as np

    idx = np.array(atom_indices)
    X = np.array([coords[i] for i in idx], dtype=np.float32)  # (m, 3)
    m = len(idx)

    # Greedy farthest-point seeding
    seeds_local = [0]
    min_dists = np.full(m, np.inf, dtype=np.float32)

    for _ in range(n_clusters - 1):
        s = seeds_local[-1]
        d = np.sqrt(np.sum((X - X[s]) ** 2, axis=1))
        min_dists = np.minimum(min_dists, d)
        min_dists[seeds_local] = -1.0
        best = int(np.argmax(min_dists))
        if min_dists[best] <= 0:
            break
        seeds_local.append(best)

    # Assign to nearest seed
    seed_coords = X[seeds_local]
    diff = X[:, None, :] - seed_coords[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    assignments = np.argmin(dists, axis=1)

    # Map back to original indices
    seeds_global = [idx[s] for s in seeds_local]
    groups: dict[int, list[int]] = {s: [] for s in seeds_global}
    for i in range(m):
        groups[seeds_global[assignments[i]]].append(idx[i])

    return groups
