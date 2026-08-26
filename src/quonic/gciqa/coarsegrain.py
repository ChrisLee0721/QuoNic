"""Coarse-graining for GCIQA.

Maps full molecular systems (hundreds/thousands of atoms) to a smaller
set of super-atoms that can be encoded in a manageable number of qubits.

Strategies:
    - residue: Each residue/group → 1 super-atom (center of mass)
    - fragment: Split by molecular fragments
    - spatial: Cluster nearby atoms by distance

Example::

    from quonic.gciqa.coarsegrain import coarse_grain, CoarseGraining

    cg = coarse_grain(
        atoms=["C", "C", "O", "H", "H", "H"],
        coords=[(0,0,0), (1.5,0,0), (3,0,0), (0,1,0), (1.5,1,0), (3,1,0)],
        strategy="spatial",
        n_super_atoms=2,
    )
    print(cg.super_coords)  # 2 super-atom positions
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CoarseGraining:
    """Result of coarse-graining a molecular system.

    Attributes:
        super_atoms: Element labels for super-atoms.
        super_coords: Coordinates of super-atoms.
        atom_to_super: Mapping from atom index → super-atom index.
        super_to_atoms: Mapping from super-atom index → list of atom indices.
        masses: Atomic masses used for center-of-mass calculation.
    """

    super_atoms: list[str]
    super_coords: list[tuple[float, float, float]]
    atom_to_super: list[int]
    super_to_atoms: list[list[int]]
    masses: list[float]

    @property
    def n_super_atoms(self) -> int:
        return len(self.super_atoms)

    @property
    def n_full_atoms(self) -> int:
        return len(self.atom_to_super)

    def super_to_dict(self) -> dict[str, tuple[float, float, float]]:
        """Convert super-atom coordinates to dict format for constraints."""
        return {
            f"{i}": coord
            for i, coord in enumerate(self.super_coords)
        }

    def expand_conformation(
        self, super_coords: dict[str, tuple[float, float, float]]
    ) -> dict[str, tuple[float, float, float]]:
        """Expand a super-atom conformation back to full atom positions.

        Each full atom is placed at its super-atom's position (zero-th order
        approximation). For better accuracy, use relative offsets.

        Args:
            super_coords: Super-atom positions {super_idx: (x, y, z)}.

        Returns:
            Full atom positions {atom_idx: (x, y, z)}.
        """
        full_coords = {}
        for atom_idx, super_idx in enumerate(self.atom_to_super):
            key = f"{super_idx}"
            if key in super_coords:
                full_coords[f"{atom_idx}"] = super_coords[key]
        return full_coords


# Atomic masses (most common isotope)
_ATOMIC_MASSES: dict[str, float] = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "P": 30.974,
    "S": 32.065, "Cl": 35.453, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Fe": 55.845, "Cu": 63.546, "Zn": 65.38, "Br": 79.904, "I": 126.904,
}


def coarse_grain(
    atoms: list[str],
    coords: list[tuple[float, float, float]],
    strategy: str = "spatial",
    n_super_atoms: int | None = None,
    cutoff: float = 5.0,
    residue_ids: list[int] | None = None,
) -> CoarseGraining:
    """Coarse-grain a molecular system.

    Args:
        atoms: Element symbols.
        coords: Cartesian coordinates (Angstrom).
        strategy: "spatial", "residue", or "fragment".
        n_super_atoms: Target number of super-atoms (for spatial strategy).
        cutoff: Distance cutoff for spatial clustering (Angstrom).
        residue_ids: Per-atom residue/group IDs (for residue strategy).

    Returns:
        CoarseGraining with mapping and super-atom positions.
    """
    masses = [_ATOMIC_MASSES.get(a, 12.0) for a in atoms]

    if strategy == "residue":
        return _coarse_grain_residue(atoms, coords, masses, residue_ids)
    elif strategy == "spatial":
        return _coarse_grain_spatial(atoms, coords, masses, n_super_atoms, cutoff)
    elif strategy == "fragment":
        return _coarse_grain_fragment(atoms, coords, masses, cutoff)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")


def _coarse_grain_residue(
    atoms: list[str],
    coords: list[tuple[float, float, float]],
    masses: list[float],
    residue_ids: list[int] | None = None,
) -> CoarseGraining:
    """Group atoms by residue ID, compute center of mass for each."""
    n = len(atoms)

    if residue_ids is None:
        # Each atom is its own residue
        residue_ids = list(range(n))

    # Group atoms by residue
    residue_groups: dict[int, list[int]] = {}
    for i, rid in enumerate(residue_ids):
        residue_groups.setdefault(rid, []).append(i)

    return _build_cg_from_groups(atoms, coords, masses, residue_groups)


def _coarse_grain_spatial(
    atoms: list[str],
    coords: list[tuple[float, float, float]],
    masses: list[float],
    n_super_atoms: int | None = None,
    cutoff: float = 5.0,
) -> CoarseGraining:
    """Cluster atoms by spatial proximity using greedy seeding."""
    n = len(atoms)

    if n_super_atoms is None:
        n_super_atoms = max(1, n // 10)

    if n_super_atoms >= n:
        # Each atom is its own super-atom
        groups = {i: [i] for i in range(n)}
        return _build_cg_from_groups(atoms, coords, masses, groups)

    # Greedy seed selection: pick first seed, then farthest from existing seeds
    seeds = [0]
    for _ in range(n_super_atoms - 1):
        max_dist = -1
        best = -1
        for i in range(n):
            if i in seeds:
                continue
            min_d = min(
                _distance(coords[i], coords[s]) for s in seeds
            )
            if min_d > max_dist:
                max_dist = min_d
                best = i
        if best >= 0:
            seeds.append(best)

    # Assign each atom to nearest seed
    groups: dict[int, list[int]] = {s: [] for s in seeds}
    for i in range(n):
        best_seed = min(seeds, key=lambda s: _distance(coords[i], coords[s]))
        groups[best_seed].append(i)

    return _build_cg_from_groups(atoms, coords, masses, groups)


def _coarse_grain_fragment(
    atoms: list[str],
    coords: list[tuple[float, float, float]],
    masses: list[float],
    cutoff: float = 2.0,
) -> CoarseGraining:
    """Group atoms by connectivity (distance-based fragments)."""
    n = len(atoms)

    # Build adjacency based on distance cutoff
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _distance(coords[i], coords[j]) < cutoff:
                adj[i].append(j)
                adj[j].append(i)

    # BFS to find connected components
    visited = [False] * n
    groups: dict[int, list[int]] = {}
    for i in range(n):
        if visited[i]:
            continue
        component = []
        stack = [i]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    stack.append(neighbor)
        groups[component[0]] = component

    return _build_cg_from_groups(atoms, coords, masses, groups)


def _build_cg_from_groups(
    atoms: list[str],
    coords: list[tuple[float, float, float]],
    masses: list[float],
    groups: dict[int, list[int]],
) -> CoarseGraining:
    """Build CoarseGraining from atom groups."""
    n = len(atoms)

    # Sort groups by their first atom index
    sorted_keys = sorted(groups.keys())
    n_super = len(sorted_keys)

    super_atoms = []
    super_coords = []
    atom_to_super = [0] * n
    super_to_atoms = []

    for super_idx, key in enumerate(sorted_keys):
        members = groups[key]
        super_to_atoms.append(members)

        # Center of mass
        total_mass = sum(masses[i] for i in members)
        cx = sum(coords[i][0] * masses[i] for i in members) / total_mass
        cy = sum(coords[i][1] * masses[i] for i in members) / total_mass
        cz = sum(coords[i][2] * masses[i] for i in members) / total_mass

        super_atoms.append(f"CG{super_idx}")
        super_coords.append((cx, cy, cz))

        for atom_idx in members:
            atom_to_super[atom_idx] = super_idx

    return CoarseGraining(
        super_atoms=super_atoms,
        super_coords=super_coords,
        atom_to_super=atom_to_super,
        super_to_atoms=super_to_atoms,
        masses=masses,
    )


def _distance(c1: tuple[float, float, float], c2: tuple[float, float, float]) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def binding_site_super_atoms(
    cg: CoarseGraining,
    pocket_center: tuple[float, float, float],
    pocket_radius: float,
) -> list[int]:
    """Find super-atoms within a binding pocket.

    Args:
        cg: Coarse-graining result.
        pocket_center: Center of binding pocket.
        pocket_radius: Radius of pocket.

    Returns:
        List of super-atom indices within the pocket.
    """
    result = []
    for i, coord in enumerate(cg.super_coords):
        if _distance(coord, pocket_center) <= pocket_radius:
            result.append(i)
    return result
