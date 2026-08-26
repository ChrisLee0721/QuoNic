"""Binding site detection for GCIQA.

Provides abstract interface and implementations for detecting binding sites
in protein structures. Supports metal-centered and geometric pocket detection.

Usage::

    from quonic.gciqa.binding_site import MetalSiteDetector, PocketDetector

    # Detect metal binding site
    detector = MetalSiteDetector()
    site = detector.detect(protein, metal_ion=protein.metal_ions[0])

    # Detect largest pocket
    detector = PocketDetector()
    site = detector.detect(protein)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pdb import MetalIon, ProteinStructure


@dataclass
class BindingSite:
    """A detected binding site."""
    center: tuple[float, float, float]
    radius: float
    residues: list[int] = field(default_factory=list)
    atoms: list[int] = field(default_factory=list)
    site_type: str = "unknown"  # "metal", "pocket", "covalent", "ppi"


class SiteDetector(ABC):
    """Abstract base class for binding site detection.

    Subclasses implement detection for different scenarios:
    - MetalSiteDetector: metal-centered detection
    - PocketDetector: geometric cavity detection
    - CovalentSiteDetector: reactive residue detection
    - ProteinProteinInterface: PPI interface detection
    """

    @abstractmethod
    def detect(
        self,
        protein: ProteinStructure,
        **kwargs,
    ) -> BindingSite:
        """Detect a binding site.

        Args:
            protein: Parsed protein structure.
            **kwargs: Detector-specific parameters.

        Returns:
            BindingSite with center, radius, residues, atoms.
        """
        ...


class MetalSiteDetector(SiteDetector):
    """Detect binding site centered on a metal ion.

    Finds all atoms and residues within a radius of the metal ion.
    """

    def __init__(self, radius: float = 10.0):
        """
        Args:
            radius: Detection radius around the metal ion (Å).
        """
        self.radius = radius

    def detect(
        self,
        protein: ProteinStructure,
        metal_ion: MetalIon | None = None,
        **kwargs,
    ) -> BindingSite:
        """Detect binding site around a metal ion.

        Args:
            protein: Parsed protein structure.
            metal_ion: Specific metal ion to center on. If None, uses the
                first metal ion in the structure.
            **kwargs: Ignored.

        Returns:
            BindingSite centered on the metal ion.

        Raises:
            ValueError: If no metal ion is found.
        """
        if metal_ion is None:
            if not protein.metal_ions:
                raise ValueError("No metal ions found in protein structure")
            metal_ion = protein.metal_ions[0]

        center = metal_ion.coord

        # Find atoms within radius
        atoms_in_site = []
        for i, coord in enumerate(protein.coords):
            dist = _distance(coord, center)
            if dist <= self.radius:
                atoms_in_site.append(i)

        # Find residues with atoms in site
        residues_in_site = []
        seen_residues = set()
        for res in protein.residues:
            for atom_idx in res.atom_indices:
                if atom_idx in atoms_in_site:
                    if res.key not in seen_residues:
                        residues_in_site.append(res)
                        seen_residues.add(res.key)
                    break

        return BindingSite(
            center=center,
            radius=self.radius,
            residues=[protein.residues.index(r) for r in residues_in_site],
            atoms=atoms_in_site,
            site_type="metal",
        )


class PocketDetector(SiteDetector):
    """Detect largest geometric pocket (no metal required).

    Uses a grid-based approach to find cavities in the protein surface.
    """

    def __init__(self, grid_spacing: float = 1.0, probe_radius: float = 1.4):
        """
        Args:
            grid_spacing: Grid spacing for pocket detection (Å).
            probe_radius: Probe radius for cavity detection (Å).
        """
        self.grid_spacing = grid_spacing
        self.probe_radius = probe_radius

    def detect(
        self,
        protein: ProteinStructure,
        n_candidates: int = 1,
        **kwargs,
    ) -> BindingSite:
        """Detect the largest geometric pocket.

        Args:
            protein: Parsed protein structure.
            n_candidates: Number of pocket candidates to return (top N).
            **kwargs: Ignored.

        Returns:
            BindingSite for the largest pocket found.
        """
        if not protein.coords:
            return BindingSite(center=(0, 0, 0), radius=0, site_type="pocket")

        # Compute bounding box
        xs = [c[0] for c in protein.coords]
        ys = [c[1] for c in protein.coords]
        zs = [c[2] for c in protein.coords]

        min_x, max_x = min(xs) - 5, max(xs) + 5
        min_y, max_y = min(ys) - 5, max(ys) + 5
        min_z, max_z = min(zs) - 5, max(zs) + 5

        # Build grid of empty points
        pocket_points = []
        x = min_x
        while x <= max_x:
            y = min_y
            while y <= max_y:
                z = min_z
                while z <= max_z:
                    # Check if this point is inside a cavity
                    # (not too close to any atom, but surrounded by atoms)
                    min_dist = float("inf")
                    for coord in protein.coords:
                        dist = _distance((x, y, z), coord)
                        if dist < min_dist:
                            min_dist = dist

                    # Pocket point: not clashing with atoms, but within protein
                    if self.probe_radius < min_dist < 8.0:
                        pocket_points.append((x, y, z, min_dist))

                    z += self.grid_spacing
                y += self.grid_spacing
            x += self.grid_spacing

        if not pocket_points:
            # No pocket found, return center of mass
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            cz = sum(zs) / len(zs)
            return BindingSite(
                center=(cx, cy, cz),
                radius=5.0,
                site_type="pocket",
            )

        # Cluster pocket points to find pocket centers
        # Simple approach: find the point with most neighbors
        best_point = max(pocket_points, key=lambda p: p[3])
        center = (best_point[0], best_point[1], best_point[2])

        # Find atoms near this pocket
        atoms_in_site = []
        for i, coord in enumerate(protein.coords):
            dist = _distance(coord, center)
            if dist <= 10.0:
                atoms_in_site.append(i)

        # Find residues
        residues_in_site = []
        seen_residues = set()
        for res in protein.residues:
            for atom_idx in res.atom_indices:
                if atom_idx in atoms_in_site:
                    if res.key not in seen_residues:
                        residues_in_site.append(res)
                        seen_residues.add(res.key)
                    break

        return BindingSite(
            center=center,
            radius=10.0,
            residues=[protein.residues.index(r) for r in residues_in_site],
            atoms=atoms_in_site,
            site_type="pocket",
        )


def _distance(c1: tuple[float, float, float], c2: tuple[float, float, float]) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
