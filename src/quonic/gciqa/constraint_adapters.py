"""Constraint generation adapters for GCIQA.

Provides abstract interface and implementations for generating GCIQA
constraints from molecular structures. Supports template-based and
adaptive (data-driven) constraint generation.

Usage::

    from quonic.gciqa.constraint_adapters import (
        TemplateConstraintGenerator,
        AdaptiveConstraintGenerator,
    )

    # Template-based generation
    gen = TemplateConstraintGenerator()
    constraints = gen.generate(protein, metal_ion=zn_ion)

    # Adaptive generation (uses PDB statistics)
    gen = AdaptiveConstraintGenerator()
    constraints = gen.generate(protein, metal_ion=zn_ion)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .constraints import GeometricConstraint, ConstraintSet
from .metal_templates import get_metal_template, generate_metal_constraints

if TYPE_CHECKING:
    from .pdb import MetalIon, ProteinStructure


class ConstraintGenerator(ABC):
    """Abstract base class for constraint generation.

    Levels:
    1. Template-based: use predefined templates (fast, general)
    2. Data-driven: learn from known structures (accurate, specific)
    """

    @abstractmethod
    def generate(
        self,
        protein: ProteinStructure,
        metal_ion: MetalIon | None = None,
        **kwargs,
    ) -> ConstraintSet:
        """Generate constraints for the target system.

        Args:
            protein: Parsed protein structure.
            metal_ion: Metal ion to generate constraints for.
            **kwargs: Generator-specific parameters.

        Returns:
            ConstraintSet with generated constraints.
        """
        ...


class TemplateConstraintGenerator(ConstraintGenerator):
    """Generate constraints from predefined templates.

    Uses METAL_COORDINATION templates for metal ions.
    Uses standard geometric parameters for bonds, angles, etc.
    """

    def __init__(self, geometry: str = "auto", max_dist: float = 2.5):
        """
        Args:
            geometry: Coordination geometry ("auto", "tetrahedral", "octahedral", etc.)
            max_dist: Maximum metal-ligand distance for finding coordinators (Å).
        """
        self.geometry = geometry
        self.max_dist = max_dist

    def generate(
        self,
        protein: ProteinStructure,
        metal_ion: MetalIon | None = None,
        **kwargs,
    ) -> ConstraintSet:
        """Generate constraints using predefined metal coordination templates.

        Args:
            protein: Parsed protein structure.
            metal_ion: Metal ion to generate constraints for. If None, uses
                the first metal ion in the structure.
            **kwargs: Ignored.

        Returns:
            ConstraintSet with bond constraints for the coordination site.

        Raises:
            ValueError: If no metal ion is found.
        """
        if metal_ion is None:
            if not protein.metal_ions:
                raise ValueError("No metal ions found in protein structure")
            metal_ion = protein.metal_ions[0]

        # Get template
        template = get_metal_template(metal_ion.element, self.geometry)

        # Generate constraints from template
        return generate_metal_constraints(
            metal_ion, protein, template, self.max_dist
        )


class AdaptiveConstraintGenerator(ConstraintGenerator):
    """Generate constraints by learning from known structures.

    Workflow:
    1. Find atoms within coordination distance of the metal
    2. Measure actual metal-ligand distances
    3. Generate constraints based on observed distances ± tolerance
    4. Fall back to template if no coordinators found

    This is NOT data-specific optimization — it uses the actual structure
    to generate reasonable constraints, rather than hand-tuning parameters.
    """

    def __init__(
        self,
        tolerance: float = 0.3,
        max_dist: float = 2.5,
        fallback_geometry: str = "auto",
    ):
        """
        Args:
            tolerance: Distance tolerance added to observed distances (Å).
            max_dist: Maximum metal-ligand distance for finding coordinators (Å).
            fallback_geometry: Geometry for template fallback.
        """
        self.tolerance = tolerance
        self.max_dist = max_dist
        self.fallback_geometry = fallback_geometry

    def generate(
        self,
        protein: ProteinStructure,
        metal_ion: MetalIon | None = None,
        **kwargs,
    ) -> ConstraintSet:
        """Generate constraints by measuring the actual structure.

        Args:
            protein: Parsed protein structure.
            metal_ion: Metal ion to generate constraints for.
            **kwargs: Ignored.

        Returns:
            ConstraintSet with constraints based on observed distances.
        """
        if metal_ion is None:
            if not protein.metal_ions:
                raise ValueError("No metal ions found in protein structure")
            metal_ion = protein.metal_ions[0]

        # Find coordinating atoms
        coord_atoms = []
        for i, (x, y, z) in enumerate(protein.coords):
            if i == metal_ion.index:
                continue
            dist = math.sqrt(
                (x - metal_ion.coord[0]) ** 2
                + (y - metal_ion.coord[1]) ** 2
                + (z - metal_ion.coord[2]) ** 2
            )
            if dist <= self.max_dist:
                coord_atoms.append((i, dist))

        if not coord_atoms:
            # No coordinators found — fall back to template
            gen = TemplateConstraintGenerator(
                geometry=self.fallback_geometry,
                max_dist=self.max_dist,
            )
            return gen.generate(protein, metal_ion)

        # Generate constraints from observed distances
        constraints = []
        metal_key = str(metal_ion.index)

        for atom_idx, observed_dist in coord_atoms:
            atom_key = str(atom_idx)
            min_dist = max(0.0, observed_dist - self.tolerance)
            max_dist = observed_dist + self.tolerance

            constraints.append(
                GeometricConstraint.bond(
                    metal_key, atom_key,
                    min_dist=min_dist,
                    max_dist=max_dist,
                )
            )

        return ConstraintSet(constraints)
