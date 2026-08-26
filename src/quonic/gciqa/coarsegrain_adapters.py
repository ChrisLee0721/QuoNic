"""Abstract coarse-graining strategy interface for GCIQA.

Provides a base class for domain-specific coarse-graining implementations.
The existing spatial/residue/fragment strategies in coarsegrain.py remain
unchanged — this module adds an extensibility layer for protein, nucleic acid,
and membrane systems.

Usage::

    from quonic.gciqa.coarsegrain_adapters import CoarseGrainingStrategy

    class MyStrategy(CoarseGrainingStrategy):
        def coarse_grain(self, atoms, coords, **context):
            ...
        def preserve_sites(self, cg, sites):
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .coarsegrain import CoarseGraining


class CoarseGrainingStrategy(ABC):
    """Abstract base class for domain-specific coarse-graining.

    Subclasses implement coarse-graining for different molecular types:
    - ProteinCoarseGraining: residue-based, preserves metal sites
    - NucleicAcidCoarseGraining: base-pair/strand-based
    - MembraneCoarseGraining: lipid molecule-based
    - GenericCoarseGraining: spatial clustering (wraps existing coarse_grain)
    """

    @abstractmethod
    def coarse_grain(
        self,
        atoms: list[str],
        coords: list[tuple[float, float, float]],
        **context,
    ) -> CoarseGraining:
        """Perform coarse-graining on the molecular system.

        Args:
            atoms: Element symbols.
            coords: Cartesian coordinates (Angstrom).
            **context: Domain-specific context (residues, chains, metal_ions, etc.)

        Returns:
            CoarseGraining result with super-atom mappings.
        """
        ...

    @abstractmethod
    def preserve_sites(
        self,
        cg: CoarseGraining,
        sites: list[tuple[float, float, float]],
    ) -> CoarseGraining:
        """Ensure important sites (metal centers, binding pockets) are preserved.

        Splits super-atoms that contain important sites so that each site
        becomes its own super-atom.

        Args:
            cg: Initial CoarseGraining result.
            sites: List of important site coordinates.

        Returns:
            Modified CoarseGraining with sites preserved as separate super-atoms.
        """
        ...
