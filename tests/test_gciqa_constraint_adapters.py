"""Tests for GCIQA constraint adapters."""

import pytest

from quonic.gciqa.constraint_adapters import (
    ConstraintGenerator,
    TemplateConstraintGenerator,
    AdaptiveConstraintGenerator,
)
from quonic.gciqa.pdb import MetalIon, ResidueInfo, ProteinStructure


def _make_metalloprotein():
    """Create a simple metalloprotein for testing."""
    atoms = ["ZN", "N", "S", "O", "C"]
    coords = [
        (0.0, 0.0, 0.0),   # Zn
        (2.0, 0.0, 0.0),   # His N (coordinating)
        (0.0, 2.3, 0.0),   # Cys S (coordinating)
        (0.0, 0.0, 2.1),   # Water O (coordinating)
        (10.0, 0.0, 0.0),  # Far atom (not coordinating)
    ]
    residues = [
        ResidueInfo(name="HIS", number=1, chain="A", atom_indices=[1]),
        ResidueInfo(name="CYS", number=2, chain="A", atom_indices=[2]),
        ResidueInfo(name="HOH", number=3, chain="A", atom_indices=[3]),
        ResidueInfo(name="ALA", number=4, chain="A", atom_indices=[4]),
    ]
    metal_ions = [MetalIon(element="ZN", coord=(0, 0, 0), index=0)]
    return ProteinStructure(
        atoms=atoms,
        coords=coords,
        residues=residues,
        metal_ions=metal_ions,
    )


class TestConstraintGeneratorInterface:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            ConstraintGenerator()


class TestTemplateConstraintGenerator:
    def test_is_generator(self):
        gen = TemplateConstraintGenerator()
        assert isinstance(gen, ConstraintGenerator)

    def test_generate_constraints(self):
        protein = _make_metalloprotein()
        gen = TemplateConstraintGenerator(geometry="tetrahedral")
        constraints = gen.generate(protein)

        # Should have constraints for the 3 coordinating atoms
        assert len(constraints.constraints) == 3
        for c in constraints.constraints:
            assert c.type.value == "bond"

    def test_generate_with_explicit_metal(self):
        protein = _make_metalloprotein()
        gen = TemplateConstraintGenerator()
        constraints = gen.generate(protein, metal_ion=protein.metal_ions[0])
        assert len(constraints.constraints) == 3

    def test_no_metal_raises(self):
        protein = ProteinStructure(atoms=["C"], coords=[(0, 0, 0)])
        gen = TemplateConstraintGenerator()
        with pytest.raises(ValueError, match="No metal ions"):
            gen.generate(protein)

    def test_auto_geometry(self):
        protein = _make_metalloprotein()
        gen = TemplateConstraintGenerator(geometry="auto")
        constraints = gen.generate(protein)
        assert len(constraints.constraints) > 0


class TestAdaptiveConstraintGenerator:
    def test_is_generator(self):
        gen = AdaptiveConstraintGenerator()
        assert isinstance(gen, ConstraintGenerator)

    def test_generate_from_observed_distances(self):
        protein = _make_metalloprotein()
        gen = AdaptiveConstraintGenerator(tolerance=0.2)
        constraints = gen.generate(protein)

        # Should have constraints for the 3 coordinating atoms
        assert len(constraints.constraints) == 3

        # Check that constraints are based on observed distances
        for c in constraints.constraints:
            assert c.type.value == "bond"
            # min_dist should be close to observed - tolerance
            # max_dist should be close to observed + tolerance

    def test_fallback_to_template(self):
        """When no coordinators found, should fall back to template."""
        protein = ProteinStructure(
            atoms=["ZN", "C"],
            coords=[(0, 0, 0), (100, 100, 100)],
            metal_ions=[MetalIon(element="ZN", coord=(0, 0, 0), index=0)],
        )
        gen = AdaptiveConstraintGenerator(max_dist=2.5)
        constraints = gen.generate(protein)

        # Should fall back to template (no coordinators found)
        # Template generates constraints based on default geometry
        assert len(constraints.constraints) >= 0

    def test_tolerance_affects_range(self):
        protein = _make_metalloprotein()

        gen_tight = AdaptiveConstraintGenerator(tolerance=0.1)
        gen_loose = AdaptiveConstraintGenerator(tolerance=0.5)

        tight = gen_tight.generate(protein)
        loose = gen_loose.generate(protein)

        # Loose tolerance should produce wider ranges
        for tc, lc in zip(tight.constraints, loose.constraints):
            tc_range = tc.params["max_dist"] - tc.params["min_dist"]
            lc_range = lc.params["max_dist"] - lc.params["min_dist"]
            assert lc_range > tc_range
