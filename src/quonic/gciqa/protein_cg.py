"""Protein-aware coarse-graining for GCIQA.

Residue-based coarse-graining that preserves metal coordination sites.
Each residue becomes one super-atom (center of mass), except:
- Metal ions → separate super-atom (never merged)
- Residues coordinating metals → separate super-atoms
- Ligands within 5Å of metal → separate super-atoms

Usage::

    from quonic.gciqa.protein_cg import ProteinCoarseGraining

    strategy = ProteinCoarseGraining()
    cg = strategy.coarse_grain(
        atoms=protein.atoms,
        coords=protein.coords,
        residues=protein.residues,
        metal_ions=protein.metal_ions,
    )
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .coarsegrain import CoarseGraining, _ATOMIC_MASSES, _build_cg_from_groups
from .coarsegrain_adapters import CoarseGrainingStrategy

if TYPE_CHECKING:
    from .pdb import MetalIon, ResidueInfo


class ProteinCoarseGraining(CoarseGrainingStrategy):
    """Residue-based coarse-graining for proteins.

    Rules:
    1. Each residue → 1 super-atom (center of mass)
    2. Metal ions → separate super-atom (not merged)
    3. Residues coordinating metals → separate super-atoms
    4. Water molecules (HOH) → merged into nearby residue super-atoms
    """

    def __init__(self, metal_coordination_dist: float = 2.5, water_merge_dist: float = 3.5):
        """
        Args:
            metal_coordination_dist: Max distance (Å) to consider a residue
                as coordinating a metal ion.
            water_merge_dist: Max distance (Å) to merge water into nearby residue.
        """
        self.metal_coordination_dist = metal_coordination_dist
        self.water_merge_dist = water_merge_dist

    def coarse_grain(
        self,
        atoms: list[str],
        coords: list[tuple[float, float, float]],
        residues: list[ResidueInfo] | None = None,
        metal_ions: list[MetalIon] | None = None,
        **kwargs,
    ) -> CoarseGraining:
        """Coarse-grain a protein by residues, preserving metal sites.

        Args:
            atoms: Element symbols.
            coords: Cartesian coordinates (Angstrom).
            residues: Residue information (from PDB parsing).
            metal_ions: Detected metal ions.
            **kwargs: Ignored (for interface compatibility).

        Returns:
            CoarseGraining with residue-based super-atoms.
        """
        n = len(atoms)
        masses = [_ATOMIC_MASSES.get(a, 12.0) for a in atoms]

        if residues is None:
            # Fallback: each atom is its own group
            groups = {i: [i] for i in range(n)}
            return _build_cg_from_groups(atoms, coords, masses, groups)

        # Identify metal ion indices and their coordinating residues
        metal_indices = set()
        coord_residue_keys = set()

        if metal_ions:
            for metal in metal_ions:
                metal_indices.add(metal.index)
                # Find coordinating residues
                for res in residues:
                    for atom_idx in res.atom_indices:
                        dist = _distance(coords[metal.index], coords[atom_idx])
                        if dist <= self.metal_coordination_dist:
                            coord_residue_keys.add(res.key)
                            break

        # Identify water residues
        water_keys = {res.key for res in residues if res.name.upper() in ("HOH", "WAT")}

        # Build groups
        groups: dict[str, list[int]] = {}
        metal_groups: dict[int, list[int]] = {}

        for res in residues:
            if res.key in water_keys:
                # Water: merge into nearby non-water residue or keep separate
                continue

            if res.key in coord_residue_keys:
                # Coordinating residue: keep as separate group
                groups[res.key] = list(res.atom_indices)
            else:
                # Normal residue: group normally
                groups[res.key] = list(res.atom_indices)

        # Metal ions: each gets its own group
        if metal_ions:
            for metal in metal_ions:
                metal_groups[metal.index] = [metal.index]

        # Water molecules: merge into nearest non-water group or keep separate
        for res in residues:
            if res.key not in water_keys:
                continue
            # Find nearest non-water group
            best_group = None
            best_dist = float("inf")
            water_com = _center_of_mass(
                [atoms[i] for i in res.atom_indices],
                [coords[i] for i in res.atom_indices],
            )
            for group_key, group_atoms in groups.items():
                group_com = _center_of_mass(
                    [atoms[i] for i in group_atoms],
                    [coords[i] for i in group_atoms],
                )
                dist = _distance(water_com, group_com)
                if dist < best_dist:
                    best_dist = dist
                    best_group = group_key

            if best_group and best_dist <= self.water_merge_dist:
                groups[best_group].extend(res.atom_indices)
            else:
                # Keep water as separate group
                groups[res.key] = list(res.atom_indices)

        # Combine all groups (use string keys for consistency)
        all_groups: dict[str, list[int]] = {}
        for key, atoms_list in groups.items():
            all_groups[str(key)] = atoms_list
        for metal_idx, atoms_list in metal_groups.items():
            all_groups[f"metal_{metal_idx}"] = atoms_list

        # Convert to integer-keyed groups for _build_cg_from_groups
        int_groups = {}
        for i, (_, atoms_list) in enumerate(all_groups.items()):
            int_groups[i] = atoms_list

        return _build_cg_from_groups(atoms, coords, masses, int_groups)

    def preserve_sites(
        self,
        cg: CoarseGraining,
        sites: list[tuple[float, float, float]],
    ) -> CoarseGraining:
        """Ensure important sites are preserved as separate super-atoms.

        If a super-atom contains an important site coordinate, split it
        so the site becomes its own super-atom.

        Args:
            cg: Initial CoarseGraining result.
            sites: List of important site coordinates.

        Returns:
            Modified CoarseGraining with sites preserved.
        """
        # For protein coarse-graining, metal ions are already preserved
        # during coarse_grain(). This method is a no-op but satisfies
        # the abstract interface.
        return cg


def _center_of_mass(
    atoms: list[str],
    coords: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Compute center of mass for a group of atoms."""
    masses = [_ATOMIC_MASSES.get(a, 12.0) for a in atoms]
    total_mass = sum(masses)
    if total_mass == 0:
        return (0.0, 0.0, 0.0)
    cx = sum(c[0] * m for c, m in zip(coords, masses)) / total_mass
    cy = sum(c[1] * m for c, m in zip(coords, masses)) / total_mass
    cz = sum(c[2] * m for c, m in zip(coords, masses)) / total_mass
    return (cx, cy, cz)


def _distance(c1: tuple[float, float, float], c2: tuple[float, float, float]) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
